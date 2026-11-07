"""The interface every frozen encoder implements.

Images reach a backbone already decoded once for all backbones (embed.py): RGB, padded to a square on grey, at
SQUARE_SIZE px, as a uint8 batch on the GPU. `prepare` resizes that batch to the backbone's input size and
normalises it with the backbone's mean and deviation on the GPU, so no backbone crops the image (a centre crop
would hide edits near the border) and no CPU time is spent per backbone. Embeddings come back L2-normalised, as
float32 numpy arrays.
"""

import numpy as np

OPENAI_MEAN = (0.48145466, 0.4578275, 0.40821073)        # CLIP, PickScore, HPSv2
OPENAI_STD = (0.26862954, 0.26130258, 0.27577711)
IMAGENET_MEAN = (0.485, 0.456, 0.406)                    # DINOv2
IMAGENET_STD = (0.229, 0.224, 0.225)


class Backbone:
    name: str = ""          # short unique name; also the folder in work/scorer/embeddings/
    model_id: str = ""      # Hugging Face ID (and file) of the weights
    has_text: bool = True   # False for image-only encoders (no request embedding)
    image_size: int = 224
    mean: tuple = OPENAI_MEAN
    std: tuple = OPENAI_STD

    def load(self, device: str) -> None:
        """Load the model (float16 on a GPU) and its tokenizer onto `device`. Called once."""
        raise NotImplementedError

    def encode_pixels(self, pixels):
        """A normalised (n, 3, image_size, image_size) tensor -> (n, dim) embeddings, any scale."""
        raise NotImplementedError

    def encode_texts(self, texts: list[str]):
        """Texts -> (n, dim) embeddings in the image space, any scale; None if has_text is False."""
        return None

    def prepare(self, batch):
        """uint8 (n, 3, S, S) square images on the device -> the model's normalised input."""
        import torch
        import torch.nn.functional as functional

        pixels = batch.float() / 255.0
        if pixels.shape[-1] != self.image_size:
            pixels = functional.interpolate(pixels, size=(self.image_size, self.image_size), mode="bicubic",
                                            align_corners=False, antialias=True).clamp(0, 1)
        mean = torch.tensor(self.mean, device=pixels.device).view(1, 3, 1, 1)
        std = torch.tensor(self.std, device=pixels.device).view(1, 3, 1, 1)
        return ((pixels - mean) / std).to(self.dtype)

    def encode_images(self, batch) -> np.ndarray:
        import torch
        with torch.inference_mode():
            return normalised(self.encode_pixels(self.prepare(batch)))

    def encode_text_batch(self, texts: list[str]) -> np.ndarray:
        import torch
        with torch.inference_mode():
            return normalised(self.encode_texts(texts))


def normalised(vectors) -> np.ndarray:
    vectors = vectors.float()
    return (vectors / vectors.norm(dim=-1, keepdim=True)).cpu().numpy()


def model_dtype(device: str):
    import torch
    return torch.float16 if device == "cuda" else torch.float32
