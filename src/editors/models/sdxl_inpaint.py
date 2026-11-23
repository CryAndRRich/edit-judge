"""SDXL Inpainting 0.1 (diffusers, 09/2023; OpenRAIL++), the inpainting model of list 3 (Add and Replace).

`AutoPipelineForInpainting` with the fp16 weights. Settings of the model card: 20 steps, guidance 8.0, strength 0.99
(the card says 1.0 lowers quality). The card works at 1024 x 1024; here the image keeps its aspect ratio, with the
long side at 1,024 px and both sides a multiple of 8, so it is not distorted. The prompt is the plan's description
of what to paint (plan.py): for Add the mask is the plan's box, for Replace the segmented object. The VAE runs in
fp16 (its config has `force_upcast: false`, as RORem's and ObjectClear's): if it overflows (black or NaN images,
MODEL_LIST.md), the runner records them as "black"; the fix is then to load madebyollin/sdxl-vae-fp16-fix.

Settings version 2: strength 1.0 for the background op. At 0.99 the masked background keeps a trace of the original
(the run of 2026-11-21: the sky stayed sky, no mushroom cloud was painted); a new background must not depend on
the old one. Add and replace keep the card's 0.99.
"""

from ..base import Editor

MODEL = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
LONG_SIDE = 1024
STRENGTH = {"background": 1.0}      # by op; others 0.99 (the card)


def working_size(width: int, height: int, long_side: int = LONG_SIDE) -> tuple[int, int]:
    scale = long_side / max(width, height)
    return max(8, round(width * scale / 8) * 8), max(8, round(height * scale / 8) * 8)


class SDXLInpaint(Editor):
    name = "sdxl-inpaint"
    model_id = f"{MODEL} (fp16 variant)"
    tasks = ("add_replace",)
    settings_version = 2
    settings = {"long_side": LONG_SIDE, "steps": 20, "guidance": 8.0, "strength": 0.99,
                "strength_by_op": STRENGTH, "prompt": "plan"}

    def load(self, device: str) -> None:
        import torch
        from diffusers import AutoPipelineForInpainting

        self.torch, self.device = torch, device
        fp16 = device == "cuda"
        self.pipe = AutoPipelineForInpainting.from_pretrained(
            MODEL, torch_dtype=torch.float16 if fp16 else torch.float32, variant="fp16" if fp16 else None).to(device)
        self.pipe.set_progress_bar_config(disable=True)

    def edit(self, image, mask, prompt: str, seed: int = 0):
        from PIL import Image

        width, height = working_size(*image.size)
        generator = self.torch.Generator(device=self.device).manual_seed(seed)
        return self.pipe(prompt=prompt, image=image.resize((width, height), Image.Resampling.BICUBIC),
                         mask_image=mask.resize((width, height), Image.Resampling.NEAREST), height=height,
                         width=width, guidance_scale=8.0, num_inference_steps=20,
                         strength=STRENGTH.get(self.op, 0.99),
                         generator=generator).images[0]
