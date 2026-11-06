"""Sensitive-content detectors (nudity, sexual content). Each scores P(NSFW) for an image.

The six models of the team's NSFW benchmark (docs/filters-benchmark.md) and the CLIP zero-shot baseline, as
name -> "module:Class". `nemo` is LAION's CLIP-based NSFW detector (used to filter LAION-5B) in NeMo Curator, and
`opennsfw2` is the detector RealEdit (CVPR 2025) used.
"""

DETECTORS: dict[str, str] = {
    "falconsai": ".falconsai:Falconsai",          # ViT classifier (Transformers)
    "owen": ".owen:OwenElliott",                  # timm classifier, SFW / NSFW / NSFL
    "nudenet": ".nudenet:NudeNet",                # exposed-body-part detector (ONNX)
    "opennsfw2": ".opennsfw2:OpenNSFW2",          # Yahoo Open-NSFW (TensorFlow)
    "mobilenet": ".mobilenet:MobileNetNSFW",      # GantMan MobileNetV2, 5 classes (TensorFlow)
    "nemo": ".nemo:NemoCurator",                  # NVIDIA NeMo Curator: CLIP ViT-L/14 + LAION NSFW head
    "clip": ".clip:ClipNSFW",                     # CLIP zero-shot (baseline)
}
