"""HPSv2.1 (Wu et al., 2023): OpenCLIP ViT-H-14 fine-tuned on human preferences between text-to-image results
(HPD v2). The HPS score is the cosine between the image and the prompt embeddings (`hpsv2/img_score.py` at
tgxs002/HPSv2@866735e: image_features @ text_features.T of the normalised features). Loaded with open_clip_torch
instead of the `hpsv2` package (which bundles an old copy of open_clip): the same ViT-H-14 architecture, then the
HPS_v2.1 state dict from the Hugging Face repo xswu/HPSv2."""

from .base import Backbone, model_dtype

CHECKPOINT = ("xswu/HPSv2", "HPS_v2.1_compressed.pt")


class HPSv21(Backbone):
    name = "hpsv2"
    model_id = "xswu/HPSv2/HPS_v2.1_compressed.pt (open_clip ViT-H-14)"

    def load(self, device: str) -> None:
        import open_clip
        import torch
        from huggingface_hub import hf_hub_download

        self.device, self.dtype = device, model_dtype(device)
        model = open_clip.create_model("ViT-H-14", pretrained=None)
        # The checkpoint is a dict saved by the authors' training code (not only tensors), hence weights_only=False
        checkpoint = torch.load(hf_hub_download(*CHECKPOINT), map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["state_dict"])
        self.model = model.to(device=device, dtype=self.dtype).eval()
        self.tokenizer = open_clip.get_tokenizer("ViT-H-14")

    def encode_pixels(self, pixels):
        return self.model.encode_image(pixels)

    def encode_texts(self, texts: list[str]):
        return self.model.encode_text(self.tokenizer(texts).to(self.device))
