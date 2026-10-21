"""Frozen encoders whose embeddings feed the scorer heads. Register each as name -> "module:Class" (module relative
to this package); modules are imported only when a backbone is used, so torch loads only then. To add one,
subclass backbones.base.Backbone in its own file and register it here."""

import importlib

BACKBONES: dict[str, str] = {
    "clip-l14": ".clip:ClipL14",            # the CLIP-score baseline of the project goal
    "pickscore": ".clip:PickScore",         # CLIP-H fine-tuned on human preferences (Pick-a-Pic)
    "siglip": ".clip:SigLIP",               # stronger image-text alignment, 384 px
    "dinov2": ".dino:DINOv2Base",           # image only: structure and identity preservation
}


def get_backbone(name: str):
    target = BACKBONES.get(name)
    if target is None:
        raise SystemExit(f"no backbone {name!r}; registered: {sorted(BACKBONES)}")
    module, cls = target.split(":")
    return getattr(importlib.import_module(module, __name__), cls)()
