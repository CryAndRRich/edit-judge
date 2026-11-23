"""MI-GAN (Sargsyan et al., ICCV 2023; MIT): a mobile inpainting GAN distilled from Co-Mod-GAN, a second lightweight
GAN baseline next to LaMa.

The TorchScript export used by IOPaint (Sanster/models release "migan", 27 MB, MD5 checked), loaded with
`torch.jit.load`, without the iopaint package (its pins clash with Kaggle). Pre- and post-processing as in
IOPaint's MIGAN model: the model works on 512 x 512 only, so the region around the mask (with a 128 px margin) is
cropped, scaled so its long side is 512, padded to a square by reflection; the input is [0.5 - mask, image * (1 -
mask)] in [-1, 1]; the output is cropped, scaled back and pasted into the crop (the runner then pastes the masked
region onto the original, as for every model). No prompt; deterministic.
"""

from ..base import Editor
from .lama import download

URL = "https://github.com/Sanster/models/releases/download/migan/migan_traced.pt"
SIZE_BYTES = 27048675
MD5 = "76eb3b1a71c400ee3290524f7a11b89c"
SIZE = 512
MARGIN = 128


class MIGAN(Editor):
    name = "migan"
    model_id = f"{URL} (md5 {MD5[:8]})"
    tasks = ("remove",)
    settings = {"size": SIZE, "crop_margin": MARGIN}

    def load(self, device: str) -> None:
        import hashlib

        import torch
        path = download(URL, SIZE_BYTES)
        if hashlib.md5(path.read_bytes()).hexdigest() != MD5:
            raise RuntimeError(f"{path}: MD5 does not match {MD5}")
        self.torch, self.device = torch, device
        self.model = torch.jit.load(str(path), map_location=device).eval()

    def edit(self, image, mask, prompt: str, seed: int = 0):
        import numpy as np
        from PIL import Image

        torch = self.torch
        x1, y1, x2, y2 = mask.getbbox()
        x1, y1 = max(0, x1 - MARGIN), max(0, y1 - MARGIN)
        x2, y2 = min(image.width, x2 + MARGIN), min(image.height, y2 + MARGIN)
        crop, crop_mask = image.crop((x1, y1, x2, y2)), mask.crop((x1, y1, x2, y2))
        scale = SIZE / max(crop.size)
        size = (max(1, round(crop.width * scale)), max(1, round(crop.height * scale)))
        small = np.asarray(crop.resize(size, Image.Resampling.BICUBIC), dtype=np.float32) / 127.5 - 1
        holes = (np.asarray(crop_mask.resize(size, Image.Resampling.NEAREST)) > 120).astype(np.float32)
        pad = ((0, SIZE - size[1]), (0, SIZE - size[0]))
        small = np.pad(small, (*pad, (0, 0)), mode="symmetric")
        holes = np.pad(holes, pad, mode="symmetric")
        pixels = torch.from_numpy(small).permute(2, 0, 1)[None].to(self.device)
        holes_t = torch.from_numpy(holes)[None, None].to(self.device)
        with torch.inference_mode():
            out = self.model(torch.cat([0.5 - holes_t, pixels * (1 - holes_t)], dim=1))
        out = (out[0].permute(1, 2, 0) * 127.5 + 127.5).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
        filled = Image.fromarray(out[:size[1], :size[0]]).resize(crop.size, Image.Resampling.BICUBIC)
        result = image.copy()
        result.paste(Image.composite(filled, crop, crop_mask), (x1, y1))
        return result
