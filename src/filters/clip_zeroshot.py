"""CLIP zero-shot detector (Radford et al., ICML 2021), shared by both tasks: no training, only text prompts.

The image is compared with the mean text embedding of the positive prompts and of the negative prompts; the score
is the softmax probability of the positive side, with CLIP's logit scale (100). Model: OpenCLIP ViT-L/14 with the
OpenAI weights, the same encoder as LAION's NSFW detector (the `nemo` model), so the two differ only in the head.
"""

from .base import Detector


class ClipZeroShot(Detector):
    model_id = "open_clip ViT-L-14 openai"
    batch_size = 32
    install = ("open_clip_torch",)
    positive: tuple[str, ...] = ()
    negative: tuple[str, ...] = ()

    def load(self, device):
        import open_clip
        import torch
        self.torch, self.device = torch, device
        self.model, _, self.preprocess = open_clip.create_model_and_transforms("ViT-L-14", pretrained="openai")
        self.model.to(device).eval()
        tokenizer = open_clip.get_tokenizer("ViT-L-14")
        with torch.inference_mode():
            sides = []
            for prompts in (self.positive, self.negative):
                side = self.model.encode_text(tokenizer(list(prompts)).to(device)).float()
                side = (side / side.norm(dim=-1, keepdim=True)).mean(0)
                sides.append(side / side.norm())
            self.text = torch.stack(sides)                    # [positive, negative]

    def score(self, images):
        batch = self.torch.stack([self.preprocess(image) for image in images]).to(self.device)
        with self.torch.inference_mode():
            embeddings = self.model.encode_image(batch).float()
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
            return (100.0 * embeddings @ self.text.T).softmax(-1)[:, 0].tolist()
