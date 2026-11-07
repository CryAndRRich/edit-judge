"""CLIP-family dual encoders through Hugging Face transformers: OpenAI CLIP ViT-L/14 (CLIP-score), PickScore
(CLIP-H fine-tuned on Pick-a-Pic preferences) and SigLIP so400m.

Text is truncated to the model's context (77 tokens for CLIP, 64 for SigLIP); the simplified instruction is short
enough, the request as written often is not (only its beginning is read).
"""

from .base import Backbone, model_dtype


def pooled(output):
    """get_*_features returns a tensor in transformers 4.x and an output object in some later versions."""
    return output if hasattr(output, "norm") else output.pooler_output


class HFDualEncoder(Backbone):
    tokenizer_id: str = ""                  # when the model repo has no tokenizer of its own
    text_kwargs: dict = {"padding": True, "truncation": True, "max_length": 77}

    def load(self, device: str) -> None:
        from transformers import AutoModel, AutoTokenizer

        self.device, self.dtype = device, model_dtype(device)
        self.model = AutoModel.from_pretrained(self.model_id, dtype=self.dtype).to(device).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_id or self.model_id)

    def encode_pixels(self, pixels):
        return pooled(self.model.get_image_features(pixel_values=pixels))

    def encode_texts(self, texts: list[str]):
        tokens = self.tokenizer(texts, return_tensors="pt", **self.text_kwargs).to(self.device)
        return pooled(self.model.get_text_features(**tokens))


class ClipL14(HFDualEncoder):
    name = "clip-l14"
    model_id = "openai/clip-vit-large-patch14"


class PickScore(HFDualEncoder):
    name = "pickscore"
    model_id = "yuvalkirstain/PickScore_v1"
    tokenizer_id = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"


class SigLIP(HFDualEncoder):
    name = "siglip"
    model_id = "google/siglip-so400m-patch14-384"
    image_size = 384
    mean = (0.5, 0.5, 0.5)
    std = (0.5, 0.5, 0.5)
    text_kwargs = {"padding": "max_length", "truncation": True, "max_length": 64}   # as SigLIP was trained
