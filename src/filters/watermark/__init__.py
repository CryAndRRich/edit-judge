"""Watermark detectors (logos, signatures and stock-photo marks added to an image). Each scores P(watermark).

The three models of the team's watermark benchmark (docs/filters-benchmark.md), as name -> "module:Class". The
checkpoints are downloaded at pinned Hugging Face revisions and loaded with torch.load(weights_only=True).
"""

DETECTORS: dict[str, str] = {
    "convnext": ".convnext:ConvNextWatermark",          # boomb0om ConvNeXt-tiny classifier
    "efficientnet": ".efficientnet:EfficientNetWatermark",  # qwertyforce EfficientNetV2-B0 classifier
    "yolo": ".yolo:YoloWatermark",                      # qfisch YOLOv8n logo/text box detector
}
