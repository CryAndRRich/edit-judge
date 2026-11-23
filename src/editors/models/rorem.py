"""RORem (Li et al., CVPR 2025): SDXL inpainting fine-tuned for object removal on human-checked pairs.

hf:LetsThink/RORem is a whole SDXL inpainting pipeline (unet, vae, both text encoders), so
`AutoPipelineForInpainting` loads it directly. Settings of the authors' "RORem Diffuser" example (GitHub README):
512 x 512, 50 steps, strength 0.99, guidance 1.0, their quality prompt (with guidance 1.0 the negative prompt has no
effect; it is kept as written). Fixes to that example: no `variant="fp16"` (the repository has no fp16 files: the
weights are stored once, already in half precision), and no mask dilation of its own (grounding.py dilates every
mask the same way for every model).
"""

from ..base import Editor

MODEL = "LetsThink/RORem"
REVISION = "22d02de974c8dc92c717cd3431f8b0fe0eeecc3b"
RESOLUTION = 512
PROMPT = "4K, high quality, masterpiece, Highly detailed, Sharp focus, Professional, photorealistic, realistic"
NEGATIVE = "low quality, worst, bad proportions, blurry, extra finger, Deformed, disfigured, unclear background"


class RORem(Editor):
    name = "rorem"
    model_id = f"{MODEL}@{REVISION[:7]}"
    tasks = ("remove",)
    settings = {"size": RESOLUTION, "steps": 50, "strength": 0.99, "guidance": 1.0, "prompt": "authors' quality prompt"}

    def load(self, device: str) -> None:
        import torch
        from diffusers import AutoPipelineForInpainting

        self.torch, self.device = torch, device
        dtype = torch.float16 if device == "cuda" else torch.float32
        self.pipe = AutoPipelineForInpainting.from_pretrained(MODEL, revision=REVISION, torch_dtype=dtype).to(device)
        self.pipe.set_progress_bar_config(disable=True)

    def edit(self, image, mask, prompt: str, seed: int = 0):
        from PIL import Image

        size = (RESOLUTION, RESOLUTION)
        generator = self.torch.Generator(device=self.device).manual_seed(seed)
        return self.pipe(prompt=PROMPT, negative_prompt=NEGATIVE, height=RESOLUTION, width=RESOLUTION,
                         image=image.resize(size, Image.Resampling.BICUBIC),
                         mask_image=mask.resize(size, Image.Resampling.NEAREST),
                         guidance_scale=1.0, num_inference_steps=50, strength=0.99, generator=generator).images[0]
