"""HVI-CIDNet (Yan et al., CVPR 2025; MIT): low-light enhancement in the HVI colour space (a learnable intensity
collapse of HSV) with a dual-branch U-Net and lightweight cross-attention; 1.98 M parameters. List 4 (enhance, op
"light" only: it brightens whatever it gets, so samples.py gives it only the low-light instructions).

The repository has no package to install: its code (net/CIDNet.py and helpers) is downloaded at COMMIT into
EJ_MODELS_DIR and put on the path. Its imports beyond torch are einops and huggingface_hub (PyTorchModelHubMixin),
both already on Kaggle, so it runs in the main environment; the pins of requirements.txt (torch 1.13, Pillow 9.5,
scipy 1.7) are for training and metrics and are not needed. The weights are the "generalization" checkpoint, which
the authors recommend for real photos, pinned to REVISION. eval_hf.py loads them with its own `from_pretrained`
(hf_hub_download of model.safetensors, unpinned, then `load_state_dict(strict=False)`); the 191 keys of the file
match `CIDNet()` (default channels [36, 36, 72, 144], heads [1, 2, 4, 8], norm False = the repository's
config.json) exactly, so they are loaded here with strict=True at the pinned revision.

Settings of eval_hf.py (its defaults, as the team's Enhance run): alpha_s = alpha_i = gamma = 1, set on the model's
RGB_HVI module as eval_hf.py does (`trans.alpha_s`, `trans.alpha`, `trans.gated = trans.gated2 = True`); the input
in [0, 1] is reflect-padded to a multiple of 8, the output clamped to [0, 1], cropped back and rounded to 8 bits
(eval_hf.py's ToPILImage truncates instead: at most 1/255 apart). No prompt, no mask; deterministic. fp32 (the
network is tiny; about 1.2 GB peak at 1,024 px in the team's run).
"""

import sys

from ..base import Editor
from .fetch import repository_code

REPO = "Fediory/HVI-CIDNet"
COMMIT = "eb43d7d91e9a336c66856824ff9e4603ae41f408"
MODEL = "Fediory/HVI-CIDNet-Generalization"
REVISION = "51481ef2546f870060c43eb6d6525399f5b3d2d3"
WEIGHTS = "model.safetensors"
SIZE_BYTES = 7920332
MODULO = 8
ALPHA_S = 1.0       # saturation scale
ALPHA_I = 1.0       # intensity (brightness) scale
GAMMA = 1.0


class HVICIDNet(Editor):
    name = "hvi-cidnet"
    model_id = f"{MODEL}@{REVISION[:7]} (code {COMMIT[:7]})"
    tasks = ("enhance",)
    ops = ("light",)
    uses_mask = False
    settings = {"weights": "generalization", "alpha_s": ALPHA_S, "alpha_i": ALPHA_I, "gamma": GAMMA,
                "pad_modulo": MODULO, "dtype": "fp32"}

    def load(self, device: str) -> None:
        import os

        import torch
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file

        sys.path.insert(0, str(repository_code(REPO, COMMIT)))
        from net.CIDNet import CIDNet

        weights_path = hf_hub_download(MODEL, WEIGHTS, revision=REVISION)
        if os.path.getsize(weights_path) != SIZE_BYTES:
            raise RuntimeError(f"{weights_path}: {os.path.getsize(weights_path)} bytes, expected {SIZE_BYTES}")
        model = CIDNet()
        model.load_state_dict(load_file(weights_path), strict=True)
        model.trans.alpha_s = ALPHA_S
        model.trans.alpha = ALPHA_I
        model.trans.gated = True
        model.trans.gated2 = True
        self.model = model.to(device).eval()
        self.torch, self.device = torch, device

    def edit(self, image, mask, prompt: str, seed: int = 0):
        import numpy as np
        from PIL import Image

        torch = self.torch
        height, width = image.height, image.width
        pixels = torch.from_numpy(np.asarray(image.convert("RGB"), dtype=np.float32) / 255)
        pixels = pixels.permute(2, 0, 1)[None].to(self.device)
        pixels = torch.nn.functional.pad(pixels, (0, -width % MODULO, 0, -height % MODULO), mode="reflect")
        with torch.inference_mode():
            result = self.model(pixels ** GAMMA)
        result = result.clamp(0, 1)[0, :, :height, :width].permute(1, 2, 0).cpu().numpy()
        return Image.fromarray((result * 255).round().astype(np.uint8))
