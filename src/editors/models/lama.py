"""LaMa big-lama (Suvorov et al., WACV 2022), the classic eraser of list 2.

The TorchScript export of simple-lama-inpainting (release v0.1.0, Apache 2.0) loaded with `torch.jit.load`, without
the pip package, whose pins (numpy < 2, pillow < 10) clash with Kaggle. Preprocessing as in that package: RGB and
mask in [0, 1], the mask binarized (white = fill), both padded to a multiple of 8 by reflection; the output is
cropped back. It runs at the image's own size (at most 1,024 px here), with no prompt, and is deterministic.
"""

from ..base import Editor

URL = "https://github.com/enesmsahin/simple-lama-inpainting/releases/download/v0.1.0/big-lama.pt"
SIZE_BYTES = 205803670
MODULO = 8


def download(url: str, size: int):
    """The file of `url` in EJ_MODELS_DIR (downloaded once; a file of the wrong size is fetched again)."""
    import urllib.request

    from ...paths import MODELS_DIR
    path = MODELS_DIR / url.rsplit("/", 1)[1]
    if not path.exists() or path.stat().st_size != size:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".part")
        urllib.request.urlretrieve(url, tmp)
        if tmp.stat().st_size != size:
            raise RuntimeError(f"{url}: got {tmp.stat().st_size} bytes, expected {size}")
        tmp.replace(path)
    return path


class Lama(Editor):
    name = "lama"
    model_id = URL
    tasks = ("remove",)
    settings = {"size": "original", "pad_modulo": MODULO}
    download_gb = 0.21

    def fetch(self) -> None:
        download(URL, SIZE_BYTES)

    def load(self, device: str) -> None:
        import torch
        self.torch, self.device = torch, device
        self.model = torch.jit.load(str(download(URL, SIZE_BYTES)), map_location=device).eval()

    def edit(self, image, mask, prompt: str, seed: int = 0):
        import numpy as np
        from PIL import Image

        torch = self.torch
        height, width = image.height, image.width
        pad_h, pad_w = -height % MODULO, -width % MODULO
        pixels = np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255
        holes = (np.asarray(mask, dtype=np.float32)[None] > 0).astype(np.float32)
        pixels = np.pad(pixels, ((0, 0), (0, pad_h), (0, pad_w)), mode="symmetric")
        holes = np.pad(holes, ((0, 0), (0, pad_h), (0, pad_w)), mode="symmetric")
        with torch.inference_mode():
            result = self.model(torch.from_numpy(pixels)[None].to(self.device),
                                torch.from_numpy(holes)[None].to(self.device))
        result = result[0].permute(1, 2, 0).cpu().numpy()[:height, :width]
        return Image.fromarray(np.clip(result * 255, 0, 255).astype(np.uint8))
