import re

from ..base import Detector

WORD = re.compile(r"[A-Za-z0-9]{3,}")


class OcrWatermark(Detector):
    """OCR text baseline: EasyOCR (CRAFT text detector, Baek et al., CVPR 2019, and a CRNN recogniser), English.
    Score = highest recognition confidence of a text with at least 3 letters or digits, 0 when there is none. It
    flags any visible text, so captions and signs in a photo count too: a lower bound for text watermarks."""
    task, name, model_id, batch_size = "watermark", "ocr", "easyocr (CRAFT + CRNN, English)", 1
    install = ("easyocr",)

    def load(self, device):
        import easyocr
        import numpy as np
        self.np = np
        self.reader = easyocr.Reader(["en"], gpu=device == "cuda", verbose=False)

    def score(self, images):
        out = []
        for im in images:
            found = self.reader.readtext(self.np.asarray(im), detail=1)
            out.append(max((float(conf) for _, text, conf in found if WORD.search(text)), default=0.0))
        return out
