"""Sensitive-content detectors (nudity, sexual content).

Register each detector here as name -> "module:Class" (module relative to this package), e.g.
    DETECTORS = {"nudenet": ".nudenet:NudeNet"}
and write the class in its own file (nudenet.py), subclassing src.filters.base.Detector with task = "nsfw".
Candidates: NudeNet (ONNX), a ViT NSFW classifier from Hugging Face, OpenNSFW2 (needs TensorFlow), CLIP zero-shot.
"""

DETECTORS: dict[str, str] = {}
