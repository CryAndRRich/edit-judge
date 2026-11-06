"""Sensitive-content detectors (nudity, sexual content). Each scores P(NSFW) for an image.

The six models of the team's NSFW benchmark (docs/filters-benchmark.md), as name -> "module:Class":
"""

DETECTORS: dict[str, str] = {
    "falconsai": ".falconsai:Falconsai",          # ViT classifier (Transformers)
    "owen": ".owen:OwenElliott",                  # timm classifier, SFW / NSFW / NSFL
    "nudenet": ".nudenet:NudeNet",                # exposed-body-part detector (ONNX)
    "opennsfw2": ".opennsfw2:OpenNSFW2",          # Yahoo Open-NSFW (TensorFlow)
    "mobilenet": ".mobilenet:MobileNetNSFW",      # GantMan MobileNetV2, 5 classes (TensorFlow)
    "nemo": ".nemo:NemoCurator",                  # NVIDIA NeMo Curator: CLIP ViT-L/14 embedding + NSFW scorer
}
