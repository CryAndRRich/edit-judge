"""DINOv2 (self-supervised ViT): image-only embeddings that track structure and identity better than CLIP, used
for "did the edit keep what was not asked to change". The CLS token after the final layer norm."""

import numpy as np

from .base import Backbone


class DINOv2Base(Backbone):
    name = "dinov2"
    model_id = "facebook/dinov2-base"
    has_text = False

    def load(self, device: str) -> None:
        import torch
        from transformers import AutoImageProcessor, AutoModel

        self.device = device
        self.dtype = torch.float16 if device == "cuda" else torch.float32
        self.model = AutoModel.from_pretrained(self.model_id, dtype=self.dtype).to(device).eval()
        self.processor = AutoImageProcessor.from_pretrained(self.model_id)

    def preprocess(self, image):
        return self.processor(images=image, return_tensors="pt")["pixel_values"][0]

    def encode_images(self, pixels) -> np.ndarray:
        import torch
        with torch.inference_mode():
            x = self.model(pixel_values=pixels.to(self.device, self.dtype)).pooler_output.float()
        return (x / x.norm(dim=-1, keepdim=True)).cpu().numpy()
