"""DDColor (Kang et al., ICCV 2023; Apache 2.0): automatic colourisation with a ConvNeXt-L encoder and a
multi-scale query-based colour decoder, the colouriser of list 6 (change_color, op "colorize" only).

The repository has no package to install: its code is downloaded at COMMIT into EJ_MODELS_DIR and put on the path.
Its standalone `ddcolor` package ("No basicsr Required" in the README) still imports the model's building blocks from
the bundled, slimmed-down `basicsr/archs/ddcolor_arch_utils` (torch, numpy, cv2 and torchvision.utils only; timm is
optional), so nothing is installed and it runs in the main environment; the pins of requirements.txt (numpy 1.24,
opencv 4.7) are not needed for inference. The weights are the `ddcolor_modelscope` checkpoint, which MODEL_ZOO.md
recommends for photos outside ImageNet, pinned to REVISION; they are loaded straight into the repository's `DDColor`
class from config.json (instead of the README's PyTorchModelHubMixin subclass, which only passes the config).

Settings of scripts/infer.py and the repository's ColorizationPipeline: input 512 x 512 (the photo is squashed to a
square), the model sees only the lightness (L) channel; the predicted ab chroma is upscaled back to the full size
(nearest, the F.interpolate default the repository uses) and combined with the original L channel. No prompt, no
mask; deterministic. Deviation: fp32 always (0.9 GB; fp16 is untested for its spectral-norm and transformer layers).
"""

import sys

from ..base import Editor
from .fetch import forget_modules, repository_code

REPO = "piddnad/DDColor"
COMMIT = "2adb63f2656ac41cbdf7b894cddd94121a3faf13"
MODEL = "piddnad/ddcolor_modelscope"
REVISION = "060f67494e31883a4b13cb27f889f3154847ada4"
WEIGHTS = "pytorch_model.bin"
SIZE_BYTES = 911914869
INPUT_SIZE = 512


class DDColor(Editor):
    name = "ddcolor"
    repos = (MODEL,)
    local_dirs = (f"DDColor-{COMMIT}",)
    download_gb = 0.95
    model_id = f"{MODEL}@{REVISION[:7]} (code {COMMIT[:7]})"
    tasks = ("change_color",)
    ops = ("colorize",)
    uses_mask = False
    settings = {"checkpoint": "ddcolor_modelscope", "input_size": INPUT_SIZE, "chroma_upscale": "nearest",
                "dtype": "fp32"}

    def fetch(self) -> None:
        from huggingface_hub import hf_hub_download
        repository_code(REPO, COMMIT)
        hf_hub_download(MODEL, "config.json", revision=REVISION)
        hf_hub_download(MODEL, WEIGHTS, revision=REVISION)

    def load(self, device: str) -> None:
        import json
        import os

        import torch
        from huggingface_hub import hf_hub_download

        folder = str(repository_code(REPO, COMMIT))
        forget_modules("basicsr", "ddcolor")           # another editor's basicsr (CodeFormer, GFPGAN)
        if folder in sys.path:
            sys.path.remove(folder)
        sys.path.insert(0, folder)                    # first, so its own slim basicsr is the one found
        from ddcolor import ColorizationPipeline
        from ddcolor import DDColor as DDColorNet

        config_path = hf_hub_download(MODEL, "config.json", revision=REVISION)
        weights_path = hf_hub_download(MODEL, WEIGHTS, revision=REVISION)
        if os.path.getsize(weights_path) != SIZE_BYTES:
            raise RuntimeError(f"{weights_path}: {os.path.getsize(weights_path)} bytes, expected {SIZE_BYTES}")
        with open(config_path) as handle:
            config = json.load(handle)
        model = DDColorNet(**config)
        state = torch.load(weights_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)
        self.pipeline = ColorizationPipeline(model, input_size=INPUT_SIZE, device=torch.device(device))

    def edit(self, image, mask, prompt: str, seed: int = 0):
        import numpy as np
        from PIL import Image

        bgr = np.ascontiguousarray(np.asarray(image.convert("RGB"))[:, :, ::-1])
        result = self.pipeline.process(bgr)
        return Image.fromarray(np.ascontiguousarray(result[:, :, ::-1]))
