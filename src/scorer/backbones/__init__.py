"""Frozen encoders whose embeddings feed the scorer heads and the similarity baselines. Register each as
name -> "module:Class" (module relative to this package); modules are imported only when a backbone is used, so
torch loads only then. To add one, subclass backbones.base.Backbone in its own file and register it here."""

import importlib

BACKBONES: dict[str, str] = {
    "clip-l14": ".clip:ClipL14",            # OpenAI CLIP ViT-L/14: the CLIP-score baseline
    "pickscore": ".clip:PickScore",         # CLIP-H fine-tuned on human preferences (Pick-a-Pic)
    "siglip": ".clip:SigLIP",               # stronger image-text alignment, 384 px
    "hpsv2": ".hpsv2:HPSv21",               # OpenCLIP ViT-H-14 fine-tuned on human preferences (HPD v2)
    "dinov2": ".dino:DINOv2Base",           # image only: structure and identity preservation
}


def get_backbone(name: str):
    target = BACKBONES.get(name)
    if target is None:
        raise SystemExit(f"no backbone {name!r}; registered: {sorted(BACKBONES)}")
    module, class_name = target.split(":")
    return getattr(importlib.import_module(module, __name__), class_name)()
