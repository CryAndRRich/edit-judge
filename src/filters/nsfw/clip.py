from ..clip_zeroshot import ClipZeroShot


class ClipNSFW(ClipZeroShot):
    """CLIP zero-shot: NSFW prompts against safe prompts."""
    task, name = "nsfw", "clip"
    positive = ("a nude photo", "a photo of a naked person", "pornography", "a sexually explicit image",
                "a photo of a person in lingerie", "explicit adult content")
    negative = ("a photo", "a photo of a fully clothed person", "a safe for work image", "a family photo",
                "a landscape photo", "a drawing")
