"""GFPGAN v1.4 (Wang et al., CVPR 2021; Apache 2.0), the classic face restorer of list 5.

Weights: GFPGANv1.4.pth of release v1.3.0 of gh:TencentARC/GFPGAN (348,632,874 bytes; no checksum is published, so
the size is the only check). Code: the PyPI packages gfpgan 1.3.8, basicsr 1.4.2 and facexlib 0.3.0 (all from
2022-2023, the latest releases). gfpgan and facexlib are wheels, installed in their own environment with --no-deps:
their pins pull in tb-nightly (which overwrites Kaggle's tensorboard), lmdb and yapf, none needed for inference;
facexlib needs only filterpy on top of Kaggle's packages. basicsr is only an sdist, and building it failed on Kaggle
(preflight of 2026-11-24, metadata generation), so it is not installed: its sdist (pure Python; it contains the
generated version.py; sha256 checked) is unpacked into EJ_MODELS_DIR and put on the path. Its import-time
dependencies (cv2, scipy, yaml, distutils through setuptools) are on Kaggle. basicsr imports
torchvision.transforms.functional_tensor, removed in torchvision 0.17: the name is pointed at
torchvision.transforms.functional (same rgb_to_grayscale) before the import.

Inference as in inference_gfpgan.py with -v 1.4 (GFPGANer, arch "clean", channel multiplier 2, weight 0.5, which
the clean architecture ignores): RetinaFace (ResNet-50) finds the faces, each is aligned to 512 px, restored, and
pasted back over a ParseNet face mask. facexlib downloads those two models (detection_Resnet50_Final.pth,
parsing_parsenet.pth) into <cwd>/gfpgan/weights, so they are loaded from EJ_MODELS_DIR and their sizes checked.
Everything runs in fp32. The StyleGAN decoder adds random noise, so the torch generator is seeded per image.

Deviation: upscale 1 and no background upsampler (the script's defaults are x2 with Real-ESRGAN x2plus on the
background). The runner shrinks every output back to the input size, so the default would only add a
Real-ESRGAN-sharpened background (already the realesrgan editor) to GFPGAN's face restoration; with upscale 1 only
the faces change. Photos without a detected face come back unchanged. No prompt, no mask.
"""

import sys

from ..base import Editor
from .fetch import release_file
from .lama import download

URL = "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth"
SIZE_BYTES = 348632874
FACE_MODELS = {"detection_Resnet50_Final.pth": 109497761,      # facexlib release v0.1.0
               "parsing_parsenet.pth": 85331193}               # facexlib release v0.2.2
FACE_URLS = {"detection_Resnet50_Final.pth":
             "https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth",
             "parsing_parsenet.pth":
             "https://github.com/xinntao/facexlib/releases/download/v0.2.2/parsing_parsenet.pth"}
BASICSR_URL = "https://files.pythonhosted.org/packages/source/b/basicsr/basicsr-1.4.2.tar.gz"
BASICSR_SHA256 = "b89b595a87ef964cda9913b4d99380ddb6554c965577c0c10cb7b78e31301e87"
BASICSR_BYTES = 172524
UPSCALE = 1
WEIGHT = 0.5


def patch_torchvision() -> None:
    """Let basicsr's `from torchvision.transforms.functional_tensor import rgb_to_grayscale` work on torchvision
    0.17 and later, where that module is gone."""
    try:
        import torchvision.transforms.functional_tensor  # noqa: F401
    except ImportError:
        import torchvision.transforms.functional as functional
        sys.modules["torchvision.transforms.functional_tensor"] = functional


def basicsr_code():
    """The basicsr 1.4.2 sdist unpacked once into EJ_MODELS_DIR; returns the folder to put on the path."""
    import hashlib
    import tarfile

    from ...paths import MODELS_DIR
    folder = MODELS_DIR / "basicsr-1.4.2"
    if not (folder / "basicsr" / "version.py").exists():
        archive = release_file(BASICSR_URL, MODELS_DIR, BASICSR_BYTES)
        digest = hashlib.sha256(archive.read_bytes()).hexdigest()
        if digest != BASICSR_SHA256:
            archive.unlink()
            raise RuntimeError(f"{archive}: sha256 {digest}, expected {BASICSR_SHA256}")
        with tarfile.open(archive) as tar:
            tar.extractall(MODELS_DIR, filter="data")
        archive.unlink()
    return folder


class GFPGAN(Editor):
    name = "gfpgan"
    local_dirs = ("gfpgan", "basicsr-1.4.2")
    download_gb = 0.55
    model_id = URL
    tasks = ("restore",)
    uses_mask = False
    install = ("--no-deps gfpgan==1.3.8 facexlib==0.3.0",
               "filterpy==1.4.5")
    settings = {"weights": "GFPGANv1.4.pth (release v1.3.0)", "arch": "clean", "channel_multiplier": 2,
                "upscale": UPSCALE, "bg_upsampler": None, "weight": WEIGHT, "detector": "retinaface_resnet50",
                "parse": "parsenet", "dtype": "fp32", "prompt": "ignored"}

    def fetch(self) -> None:
        from ...paths import MODELS_DIR
        download(URL, SIZE_BYTES)
        basicsr_code()
        for file_name, size in FACE_MODELS.items():     # where facexlib looks for them (load())
            release_file(FACE_URLS[file_name], MODELS_DIR / "gfpgan" / "weights", size)

    def load(self, device: str) -> None:
        import contextlib

        import torch

        from ...paths import MODELS_DIR
        sys.path.insert(0, str(basicsr_code()))
        patch_torchvision()
        from gfpgan import GFPGANer

        self.torch = torch
        weights = download(URL, SIZE_BYTES)
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        with contextlib.chdir(MODELS_DIR):            # facexlib saves its models under ./gfpgan/weights
            self.restorer = GFPGANer(model_path=str(weights), upscale=UPSCALE, arch="clean", channel_multiplier=2,
                                     bg_upsampler=None, device=torch.device(device))
        for file_name, size in FACE_MODELS.items():
            path = MODELS_DIR / "gfpgan" / "weights" / file_name
            found = path.stat().st_size
            if found != size:
                path.unlink()
                raise RuntimeError(f"{path}: got {found} bytes, expected {size}")

    def edit(self, image, mask, prompt: str, seed: int = 0):
        import numpy as np
        from PIL import Image

        self.torch.manual_seed(seed)
        bgr = np.ascontiguousarray(np.asarray(image)[:, :, ::-1])
        restored = self.restorer.enhance(bgr, has_aligned=False, only_center_face=False, paste_back=True,
                                         weight=WEIGHT)[2]
        return Image.fromarray(np.ascontiguousarray(restored[:, :, ::-1]))
