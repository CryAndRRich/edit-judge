"""Real-ESRGAN x4plus (Wang et al., ICCV 2021 workshop; BSD-3), the classic upscaler of lists 4 and 7.

Weights: RealESRGAN_x4plus.pth of release v0.1.0 of gh:xinntao/Real-ESRGAN (67,040,989 bytes; no checksum is
published, so the size is the only check). Loaded with the `spandrel` package (0.4.2; not on Kaggle by default, so
it gets its own environment), not the `realesrgan` and `basicsr` packages, which are from 2022 and break on new
torchvision. As in Steven's Enhance run of 2026-11-22 (his standalone script, class RealESRGAN): RGB in [0, 1],
float16 on CUDA, tiles of 256 px with a 16 px overlap, x4 output. No prompt, no mask; deterministic.

- enhance (ops light, enhance): the x4 output is returned and the runner shrinks it back to the input size
  (LANCZOS), so the model acts as a sharpener and every output stays comparable pixel for pixel.
- formatting (op format): runs only on samples whose plan asks for a resize with a scale above 1 ("make it
  bigger"); the x4 output is resized (LANCZOS) to the input size times the plan's scale, with the long side capped
  at 2,048 px, and kept at that size (`keep_size` False). Cut-out samples and other formatting plans are n/a.
"""

from ..base import Editor
from .lama import download

URL = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
SIZE_BYTES = 67040989
NATIVE_SCALE = 4
TILE = 256
TILE_PAD = 16
MAX_LONG_SIDE = 2048
ENHANCE_OPS = ("light", "enhance")


class RealESRGAN(Editor):
    name = "realesrgan"
    model_id = URL
    tasks = ("enhance", "formatting")
    uses_mask = False
    install = ("spandrel==0.4.2",)
    settings = {"weights": "RealESRGAN_x4plus.pth (release v0.1.0)", "native_scale": NATIVE_SCALE,
                "dtype": "fp16", "tile": TILE, "tile_pad": TILE_PAD, "prompt": "ignored",
                "enhance_output": "x4, shrunk back to the input size by the runner (LANCZOS)",
                "format_output": "x4 resized to the plan's scale (LANCZOS), long side at most 2048 px"}

    def applies(self, sample, plan) -> bool:
        """Enhance samples always; formatting samples only when the plan asks to enlarge the image."""
        if sample.get("op") in ENHANCE_OPS:
            return True
        return plan is not None and plan.get("action") == "resize" and (plan.get("scale") or 0) > 1

    def load(self, device: str) -> None:
        import torch
        from spandrel import ModelLoader

        self.torch, self.device = torch, device
        self.dtype = torch.float16 if device == "cuda" else torch.float32
        model = ModelLoader().load_from_file(str(download(URL, SIZE_BYTES))).to(device).eval()
        self.model = model.half() if self.dtype == torch.float16 else model

    def upscale(self, image):
        """The x4 output of `image` (PIL RGB), made tile by tile."""
        import numpy as np
        from PIL import Image

        torch = self.torch
        pixels = torch.from_numpy(np.asarray(image, dtype=np.float32) / 255).permute(2, 0, 1)[None]
        pixels = pixels.to(self.device, self.dtype)
        height, width = image.height, image.width
        scale = NATIVE_SCALE
        result = torch.zeros(1, 3, height * scale, width * scale, device=self.device, dtype=self.dtype)
        with torch.inference_mode():
            for top in range(0, height, TILE):
                for left in range(0, width, TILE):
                    pad_top, pad_left = max(top - TILE_PAD, 0), max(left - TILE_PAD, 0)
                    pad_bottom = min(top + TILE + TILE_PAD, height)
                    pad_right = min(left + TILE + TILE_PAD, width)
                    tile = self.model(pixels[:, :, pad_top:pad_bottom, pad_left:pad_right])
                    bottom, right = min(top + TILE, height), min(left + TILE, width)
                    tile = tile[:, :, (top - pad_top) * scale:(bottom - pad_top) * scale,
                                (left - pad_left) * scale:(right - pad_left) * scale]
                    result[:, :, top * scale:bottom * scale, left * scale:right * scale] = tile
        array = torch.nan_to_num(result[0].float()).clamp(0, 1).permute(1, 2, 0).cpu().numpy()
        return Image.fromarray((array * 255).round().astype(np.uint8))

    def edit(self, image, mask, prompt: str, seed: int = 0):
        from PIL import Image

        self.keep_size = self.op in ENHANCE_OPS
        upscaled = self.upscale(image)
        if self.keep_size:              # enhance: the runner shrinks the x4 output back to the input size
            return upscaled
        factor = min(float(self.plan["scale"]), MAX_LONG_SIDE / max(image.size))
        target = (max(1, round(image.width * factor)), max(1, round(image.height * factor)))
        return upscaled.resize(target, Image.Resampling.LANCZOS)
