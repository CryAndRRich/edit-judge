"""CLIP-family dual encoders through Hugging Face transformers: OpenAI CLIP ViT-L/14, PickScore and SigLIP.

float16 on a GPU, float32 elsewhere. Text is truncated to the model's context (77 tokens for CLIP, 64 for SigLIP),
so a long request keeps only its beginning (for RealEdit and forum posts: the title and the first sentences).
"""

import numpy as np

from .base import Backbone


def _pooled(out):
    """get_*_features returns a tensor in transformers 4.x and an output object in some later versions."""
    return out if hasattr(out, "norm") else out.pooler_output


class HFDualEncoder(Backbone):
    processor_id: str = ""                  # when the model repo has no processor of its own
    text_kwargs: dict = {"padding": True, "truncation": True, "max_length": 77}

    def load(self, device: str) -> None:
        import torch
        from transformers import AutoModel, AutoProcessor

        self.device = device
        self.dtype = torch.float16 if device == "cuda" else torch.float32
        self.model = AutoModel.from_pretrained(self.model_id, dtype=self.dtype).to(device).eval()
        self.processor = AutoProcessor.from_pretrained(self.processor_id or self.model_id)

    def preprocess(self, image):
        return self.processor(images=image, return_tensors="pt")["pixel_values"][0]

    def _normalised(self, x) -> np.ndarray:
        x = _pooled(x).float()
        return (x / x.norm(dim=-1, keepdim=True)).cpu().numpy()

    def encode_images(self, pixels):
        import torch
        with torch.inference_mode():
            return self._normalised(self.model.get_image_features(pixel_values=pixels.to(self.device, self.dtype)))

    def encode_texts(self, texts: list[str]):
        import torch
        tokens = self.processor(text=texts, return_tensors="pt", **self.text_kwargs).to(self.device)
        with torch.inference_mode():
            return self._normalised(self.model.get_text_features(**tokens))


class ClipL14(HFDualEncoder):
    name = "clip-l14"
    model_id = "openai/clip-vit-large-patch14"


class PickScore(HFDualEncoder):
    name = "pickscore"
    model_id = "yuvalkirstain/PickScore_v1"
    processor_id = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
    batch_size = 32


class SigLIP(HFDualEncoder):
    name = "siglip"
    model_id = "google/siglip-so400m-patch14-384"
    text_kwargs = {"padding": "max_length", "truncation": True, "max_length": 64}   # as SigLIP was trained
    batch_size = 32
