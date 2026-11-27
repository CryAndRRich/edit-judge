"""InstructIR (Conde et al., ECCV 2024; MIT): a NAFNet restoration network conditioned on the embedding of a written
instruction; one model for 7 degradations (noise, blur, rain, haze, low light, enhancement, upscaling). The authors
say it generalises poorly to real photos.

The repository has no package to install: its code is downloaded at COMMIT (the head of the default branch
`gradio`, what a plain `git clone` gave the team's Kaggle test) into EJ_MODELS_DIR and its top-level packages
`models` and `text` are imported from there. The weights im_instructir-7d.pt and lm_instructir-7d.pt come from
the HF repository at REVISION (the same files, by SHA-256, as the copies in the code archive). The text encoder is
TaylorAI/bge-micro-v2 (MIT), pinned at ENCODER_REVISION: the repository's LanguageModel takes no revision, so the
encoder is downloaded into a local folder whose path still contains "TaylorAI/bge-micro-v2" (LanguageModel picks
mean pooling by that substring) and that folder is handed to it. Its requirements (torch, transformers, PyYAML)
are on Kaggle, so it runs in the main environment.

Settings of the authors' gradio_demo.py and configs/eval5d.yml, as in the team's Kaggle test: fp32, the image
reflect-padded to a multiple of 16 (the network's 4 downsamplings; it would zero-pad itself otherwise) and cropped
back, the prompt is the instruction. NaN or inf in the output are set to 0 or 1 (nan_to_num); an all-black output
is marked "black" by the runner.
"""

import sys

from ..base import Editor
from .fetch import repository_code

REPO = "mv-lab/InstructIR"
COMMIT = "5959746f21d2eb17c27aec5818536cf24f289138"
MODEL = "marcosv/InstructIR"
REVISION = "faa7dcfaa04721d7e0e52d4dbef2abb5f5b5d07a"
ENCODER = "TaylorAI/bge-micro-v2"
ENCODER_REVISION = "3edf6d7de0faa426b09780416fe61009f26ae589"
IMAGE_WEIGHTS = "im_instructir-7d.pt"
HEAD_WEIGHTS = "lm_instructir-7d.pt"
MULTIPLE = 16
REPO_PACKAGES = ("models", "text")


def is_repo_module(module_name: str) -> bool:
    return any(module_name == package or module_name.startswith(package + ".") for package in REPO_PACKAGES)


def import_repository(folder):
    """The repository's `models.instructir` and `text.models`. Its packages have generic top-level names, so any
    other `models` / `text` already imported (another repository's) is set aside first, the folder is on sys.path
    only during the import, and the repository's modules are dropped from sys.modules afterwards (the returned
    module objects keep working)."""
    import importlib

    from ...paths import MODELS_DIR

    saved = {key: sys.modules.pop(key) for key in list(sys.modules) if is_repo_module(key)}
    # the other repositories' folders leave sys.path during the import: InstructIR's `models` has no
    # __init__.py, and Python prefers a regular `models` package anywhere on the path (VOSR's) to it
    # (2026-11-28, one process for every editor: "No module named 'models.instructir'" after VOSR)
    others = [entry for entry in sys.path if entry.startswith(str(MODELS_DIR)) and entry != str(folder)]
    for entry in others:
        sys.path.remove(entry)
    sys.path.insert(0, str(folder))
    try:
        instructir = importlib.import_module("models.instructir")
        text_models = importlib.import_module("text.models")
    finally:
        sys.path.remove(str(folder))
        sys.path.extend(others)
        for key in [key for key in sys.modules if is_repo_module(key)]:
            del sys.modules[key]
        sys.modules.update(saved)
    return instructir, text_models


def pad_to_multiple(array, factor: int):
    """Reflect-pad an HxWxC array so both sides are multiples of `factor`; returns it and the original size."""
    import numpy as np
    height, width = array.shape[:2]
    pad_height, pad_width = (-height) % factor, (-width) % factor
    if pad_height or pad_width:
        array = np.pad(array, ((0, pad_height), (0, pad_width), (0, 0)), mode="reflect")
    return array, (height, width)


class InstructIR(Editor):
    name = "instructir"
    repos = (MODEL,)
    local_dirs = (f"InstructIR-{COMMIT}", ENCODER.split("/")[0])
    download_gb = 0.2
    model_id = f"{MODEL}@{REVISION[:7]} (code {COMMIT[:7]}, text encoder {ENCODER}@{ENCODER_REVISION[:7]})"
    tasks = ("enhance",)
    uses_mask = False
    settings = {"weights": f"{IMAGE_WEIGHTS}, {HEAD_WEIGHTS}", "config": "configs/eval5d.yml",
                "text_encoder": ENCODER, "dtype": "fp32", "padding": f"reflect to a multiple of {MULTIPLE}",
                "prompt": "instruction"}

    def fetch(self) -> None:
        from huggingface_hub import hf_hub_download, snapshot_download

        from ...paths import MODELS_DIR
        repository_code(REPO, COMMIT)
        hf_hub_download(MODEL, IMAGE_WEIGHTS, revision=REVISION)
        hf_hub_download(MODEL, HEAD_WEIGHTS, revision=REVISION)
        snapshot_download(ENCODER, revision=ENCODER_REVISION,
                          local_dir=MODELS_DIR / ENCODER / ENCODER_REVISION[:7],
                          allow_patterns=["*.json", "*.txt", "model.safetensors"])

    def load(self, device: str) -> None:
        import os
        from types import SimpleNamespace

        import torch
        import yaml
        from huggingface_hub import hf_hub_download, snapshot_download

        from ...paths import MODELS_DIR

        folder = repository_code(REPO, COMMIT)
        instructir, text_models = import_repository(folder)

        def namespace(value):
            return SimpleNamespace(**{key: namespace(item) for key, item in value.items()}) \
                if isinstance(value, dict) else value

        with open(folder / "configs" / "eval5d.yml") as handle:
            config = namespace(yaml.safe_load(handle))
        if config.llm.model != ENCODER:
            raise RuntimeError(f"eval5d.yml names the text encoder {config.llm.model}, expected {ENCODER}")

        image_weights = hf_hub_download(MODEL, IMAGE_WEIGHTS, revision=REVISION)
        head_weights = hf_hub_download(MODEL, HEAD_WEIGHTS, revision=REVISION)
        self.model = instructir.create_model(
            input_channels=config.model.in_ch, width=config.model.width, enc_blks=config.model.enc_blks,
            middle_blk_num=config.model.middle_blk_num, dec_blks=config.model.dec_blks,
            txtdim=config.model.textdim)
        self.model.load_state_dict(torch.load(image_weights, map_location="cpu", weights_only=True), strict=True)
        self.model.to(device).eval()

        # the path must keep the substring "TaylorAI/bge-micro-v2": LanguageModel chooses mean pooling by it
        encoder_dir = MODELS_DIR / ENCODER / ENCODER_REVISION[:7]
        snapshot_download(ENCODER, revision=ENCODER_REVISION, local_dir=encoder_dir,
                          allow_patterns=["*.json", "*.txt", "model.safetensors"])
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        self.language_model = text_models.LanguageModel(model=str(encoder_dir))   # stays on the CPU, as upstream
        if ENCODER not in self.language_model.model_name or "clip" in self.language_model.model_name:
            raise RuntimeError(f"text encoder path {encoder_dir} breaks LanguageModel's pooling choice")
        self.head = text_models.LMHead(embedding_dim=config.llm.model_dim, hidden_dim=config.llm.embd_dim,
                                       num_classes=config.llm.nclasses)
        self.head.load_state_dict(torch.load(head_weights, map_location="cpu", weights_only=True), strict=True)
        self.head.to(device).eval()
        self.torch, self.device = torch, device

    def edit(self, image, mask, prompt: str, seed: int = 0):
        import numpy as np
        from PIL import Image

        torch = self.torch
        array, (height, width) = pad_to_multiple(np.asarray(image.convert("RGB")), MULTIPLE)
        tensor = torch.from_numpy(array.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(self.device)
        with torch.no_grad():
            text_embedding, _ = self.head(self.language_model(prompt).to(self.device))
            output = self.model(tensor, text_embedding)[0, :, :height, :width].float().cpu()
        output = torch.nan_to_num(output, nan=0.0, posinf=1.0, neginf=0.0).clamp(0, 1)
        return Image.fromarray((output * 255).round().byte().permute(1, 2, 0).numpy())
