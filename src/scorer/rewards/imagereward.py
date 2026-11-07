"""ImageReward (Xu et al., NeurIPS 2023): BLIP ViT-L with a cross-attention text encoder and an MLP, trained on
137k expert comparisons of text-to-image results. It reads the prompt (truncated to 35 tokens) and the generated
image only, never the original: a text-to-image reward model used as a baseline for edits.

Code: THUDM/ImageReward at COMMIT, downloaded as source (not pip-installed: its setup.py and CLIP's need
pkg_resources, which current setuptools no longer has, and the package pulls in CLIP, diffusers and fairscale for
parts not used here). Only the model modules are imported (ImageReward.py and models/BLIP/), without the package's
__init__; fairscale's checkpoint_wrapper, imported by BLIP's ViT but used only for gradient checkpointing in
training, gets a stub when fairscale is not installed. `load` does what `ImageReward.load` does (same files, same
`load_state_dict(strict=False)`), and `score` is the batched form of `ImageReward.score` (same tokenizer settings,
preprocessing, MLP and normalisation by the reward mean and deviation), for speed.
"""

import importlib
import importlib.util
import sys
import types
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

from ...paths import MODELS_DIR
from .base import RewardModel

COMMIT = "78f8051059b34fef8e0032ccb40561c23a2ddbbb"
CODE_URL = f"https://github.com/THUDM/ImageReward/archive/{COMMIT}.zip"
WEIGHTS = "zai-org/ImageReward"        # ImageReward.pt and med_config.json (the repo was THUDM/ImageReward)


def download_code(folder: Path) -> Path:
    """The source tree of the pinned commit (downloaded once)."""
    code = folder / f"ImageReward-{COMMIT}"
    if not code.exists():
        folder.mkdir(parents=True, exist_ok=True)
        archive = folder / f"{COMMIT}.zip"
        urlretrieve(CODE_URL, archive)
        with zipfile.ZipFile(archive) as source:
            source.extractall(folder)
        archive.unlink()
    return code


def stub_module(name: str, **attributes) -> None:
    module = types.ModuleType(name)
    module.__dict__.update(attributes)
    sys.modules[name] = module


def import_image_reward(code: Path):
    """The ImageReward class of the source tree, imported without ImageReward/__init__.py and models/__init__.py
    (they import CLIP, diffusers and the training code)."""
    package = code / "ImageReward"
    stub_module("ImageReward", __path__=[str(package)])
    stub_module("ImageReward.models", __path__=[str(package / "models")])
    if importlib.util.find_spec("fairscale") is None:
        def checkpoint_wrapper(module, *args, **kwargs):
            raise NotImplementedError("gradient checkpointing (fairscale) is not available: inference only")
        for name in ("fairscale", "fairscale.nn", "fairscale.nn.checkpoint"):
            stub_module(name, __path__=[])
        stub_module("fairscale.nn.checkpoint.checkpoint_activations", checkpoint_wrapper=checkpoint_wrapper)
    return importlib.import_module("ImageReward.ImageReward").ImageReward


class ImageRewardModel(RewardModel):
    name = "imagereward"
    model_id = f"THUDM/ImageReward ImageReward-v1.0 (code {COMMIT[:7]})"
    batch_size = 32
    install = ("transformers>=4.56,<5 timm",)     # BLIP's text encoder and ViT; torch is the machine's

    def load(self, device: str) -> None:
        import torch
        from huggingface_hub import hf_hub_download

        image_reward = import_image_reward(download_code(MODELS_DIR / "imagereward"))
        state_dict = torch.load(hf_hub_download(WEIGHTS, "ImageReward.pt"), map_location="cpu")
        model = image_reward(device=device, med_config=hf_hub_download(WEIGHTS, "med_config.json"))
        result = model.load_state_dict(state_dict, strict=False)
        missing = [key for key in result.missing_keys if not key.endswith("position_ids")]
        if missing:
            raise SystemExit(f"ImageReward checkpoint misses {len(missing)} parameters, e.g. {missing[:3]}")
        self.device = device
        self.model = model.to(device).eval()

    def score(self, batch):
        import torch

        texts = [item.text for item, _, _ in batch]
        with torch.inference_mode():
            tokens = self.model.blip.tokenizer(texts, padding="max_length", truncation=True, max_length=35,
                                               return_tensors="pt").to(self.device)
            pixels = torch.stack([self.model.preprocess(edit) for _, _, edit in batch]).to(self.device)
            image_embeds = self.model.blip.visual_encoder(pixels)
            image_mask = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=self.device)
            output = self.model.blip.text_encoder(tokens.input_ids, attention_mask=tokens.attention_mask,
                                                  encoder_hidden_states=image_embeds,
                                                  encoder_attention_mask=image_mask, return_dict=True)
            rewards = self.model.mlp(output.last_hidden_state[:, 0, :].float())
            rewards = (rewards - self.model.mean) / self.model.std
        return [{"score": round(float(value), 5)} for value in rewards.squeeze(-1).tolist()]
