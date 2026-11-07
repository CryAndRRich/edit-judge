"""Watermark detectors (logos, signatures and stock-photo marks added to an image). Each scores P(watermark).

The three models of the team's watermark benchmark (../docs/filters-benchmark.md) and three baselines (LAION's
detector, CLIP zero-shot, OCR), as name -> "module:Class". Checkpoints are loaded with torch.load(weights_only=True);
Hugging Face ones at pinned revisions.
"""

DETECTORS: dict[str, str] = {
    "convnext": ".convnext:ConvNextWatermark",          # boomb0om ConvNeXt-tiny classifier
    "efficientnet": ".efficientnet:EfficientNetWatermark",  # qwertyforce EfficientNetV2-B0 classifier
    "yolo": ".yolo:YoloWatermark",                      # qfisch YOLOv8n logo/text box detector
    "laion": ".laion:LaionWatermark",                   # LAION-5B watermark classifier (baseline)
    "clip": ".clip:ClipWatermark",                      # CLIP zero-shot (baseline)
    "ocr": ".ocr:OcrWatermark",                         # EasyOCR text detection (baseline)
}
