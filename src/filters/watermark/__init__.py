"""Watermark detectors (logos, signatures and stock-photo marks added to an image).

Register each detector here as name -> "module:Class" (module relative to this package), e.g.
    DETECTORS = {"laion": ".laion:LaionWatermark"}
and write the class in its own file, subclassing src.filters.base.Detector with task = "watermark".
Candidates: the LAION watermark classifier, an OCR text detector, CLIP zero-shot.
"""

DETECTORS: dict[str, str] = {}
