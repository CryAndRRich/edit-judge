"""BiRefNet (Zheng et al., CAAI AIR 2024; MIT), the background remover of ../docs/editors.md list 7: a bilateral
reference network (Swin-L backbone) that predicts a high-resolution foreground matte of the main subject.

The general-use weights (stored in fp16) and their remote code (birefnet.py, BiRefNet_config.py) come from the
Hugging Face repository at REVISION, loaded with `AutoModelForImageSegmentation.from_pretrained(...,
trust_remote_code=True)`; `revision` pins the code too, as it lives in the same repository. That revision contains
the author's fixes for transformers >= 4.56 ("is_encoder_decoder"), timm >= 1.0.23 and transformers 5. The code
needs timm, kornia and einops (on Kaggle), so it runs in the main environment; the repository's training pin
`torch==2.5.1` is skipped.

Settings of the model card: the image resized to 1024 x 1024, ImageNet normalisation, `.half()` on the GPU (float32
on the CPU), the sigmoid of the last output (taken in float32) as the matte, resized back to the image size.
Output: the image composited on white by that matte (RGB, the input size). It runs on cut-out samples only (other
formatting ops are n/a); there is no mask and the prompt is ignored, so it always cuts out the most salient
subject. Deterministic.
"""

from ..base import Editor

MODEL = "ZhengPeng7/BiRefNet"
REVISION = "e2bf8e4460fc8fa32bba5ea4d94b3233d367b0e4"
SIZE = 1024
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
BACKGROUND = (255, 255, 255)


class BiRefNet(Editor):
    name = "birefnet"
    repos = (MODEL,)
    download_gb = 0.45
    model_id = f"{MODEL}@{REVISION[:7]}"
    tasks = ("formatting",)
    ops = ("cutout",)
    uses_mask = False
    settings = {"size": SIZE, "normalisation": "imagenet", "background": "white", "prompt": "ignored"}

    def fetch(self) -> None:
        from huggingface_hub import snapshot_download
        snapshot_download(MODEL, revision=REVISION)

    def load(self, device: str) -> None:
        import torch
        from transformers import AutoModelForImageSegmentation

        self.torch, self.device = torch, device
        self.dtype = torch.float16 if device == "cuda" else torch.float32
        model = AutoModelForImageSegmentation.from_pretrained(MODEL, revision=REVISION, trust_remote_code=True)
        self.model = model.to(device=device, dtype=self.dtype).eval()

    def edit(self, image, mask, prompt: str, seed: int = 0):
        import numpy as np
        from PIL import Image

        torch = self.torch
        resized = image.resize((SIZE, SIZE), Image.Resampling.BILINEAR)
        pixels = (np.asarray(resized, dtype=np.float32) / 255 - np.array(MEAN, dtype=np.float32)) \
            / np.array(STD, dtype=np.float32)
        batch = torch.from_numpy(pixels.transpose(2, 0, 1).copy())[None].to(self.device, self.dtype)
        with torch.inference_mode():
            logits = self.model(batch)[-1]
        matte = logits.float().sigmoid()[0, 0].cpu().numpy()
        matte = Image.fromarray(np.clip(matte * 255, 0, 255).round().astype(np.uint8), mode="L")
        matte = matte.resize(image.size, Image.Resampling.BICUBIC)
        return Image.composite(image, Image.new("RGB", image.size, BACKGROUND), matte)
