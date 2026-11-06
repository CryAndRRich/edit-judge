from ..clip_zeroshot import ClipZeroShot


class ClipWatermark(ClipZeroShot):
    """CLIP zero-shot: watermarked prompts against clean prompts."""
    task, name = "watermark", "clip"
    positive = ("a photo with a watermark", "a stock photo with a watermark text over it",
                "an image with a semi-transparent logo watermark", "a photo with a copyright watermark",
                "a picture with the word sample written across it")
    negative = ("a photo", "a clean photo without any watermark", "a photo without text",
                "an unedited photo", "a picture")
