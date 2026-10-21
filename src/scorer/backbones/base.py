"""The interface every frozen encoder implements.

`preprocess` runs in DataLoader workers (decoding and resizing 1,024 px JPEGs is the slow part), the encode methods
in the main process on the GPU. Embeddings are returned L2-normalised, as float32 numpy arrays.
"""

from PIL import Image


class Backbone:
    name: str = ""          # short unique name; also the folder in data/scorer/embeddings/
    model_id: str = ""      # Hugging Face ID
    has_text: bool = True   # False for image-only encoders (no request embedding)
    batch_size: int = 64

    def load(self, device: str) -> None:
        """Load the model and its processor onto `device`. Called once, before preprocess."""
        raise NotImplementedError

    def preprocess(self, image: Image.Image):
        """One RGB image -> the model's pixel tensor (no batch dimension)."""
        raise NotImplementedError

    def encode_images(self, pixels):
        """A batch of pixel tensors -> (n, dim) normalised embeddings."""
        raise NotImplementedError

    def encode_texts(self, texts: list[str]):
        """Request texts -> (n, dim) normalised embeddings in the image space; None if has_text is False."""
        return None
