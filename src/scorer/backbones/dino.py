"""DINOv2 (self-supervised ViT): image-only embeddings that track structure and identity better than CLIP, used
for "did the edit keep what was not asked to change". The CLS token after the final layer norm (pooler_output),
on the whole padded image at 224 px (16 x 16 patches) instead of the processor's resize and centre crop."""

from .base import IMAGENET_MEAN, IMAGENET_STD, Backbone, model_dtype


class DINOv2Base(Backbone):
    name = "dinov2"
    model_id = "facebook/dinov2-base"
    has_text = False
    mean = IMAGENET_MEAN
    std = IMAGENET_STD

    def load(self, device: str) -> None:
        from transformers import AutoModel

        self.device, self.dtype = device, model_dtype(device)
        self.model = AutoModel.from_pretrained(self.model_id, dtype=self.dtype).to(device).eval()

    def encode_pixels(self, pixels):
        return self.model(pixel_values=pixels).pooler_output
