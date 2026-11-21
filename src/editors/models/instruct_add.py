"""Mask-free object addition: InstructPix2Pix-style models trained to add objects from an instruction alone, so they
decide where the new thing goes (our VLM places new things badly: MODEL_LIST.md list 3).

- Paint by Inpaint (Wasserman et al., CVPR 2025; MIT): trained on PIPE, 1M pairs made by removing objects from
  real photos and reversing the edit; `add-finetuned-mb` is the addition model fine-tuned on MagicBrush's
  additions. Settings of the authors' README: Euler Ancestral, 50 steps, guidance 7, image guidance 1.5.
- EraseDraw (Canberk et al., ECCV 2024; MIT): trained on pairs made by erasing objects; v0.2 of the authors' HF
  weights. Settings of its model card: Euler Ancestral (its own scheduler config), 10 steps, image guidance 1,
  guidance 7.5 (the pipeline default).
Both are SD 1.5 InstructPix2Pix models (`StableDiffusionInstructPix2PixPipeline`, fp16, no safety checker), so
the image is resized to a long side of 512 px (both sides a multiple of 8), the resolution they were trained at.
They run on the add samples only (op "add"); replace and background samples are "n/a". The prompt is the simplified
instruction ("Add a camel behind us."); the output is not pasted back (the model chose the region).
"""

from ..base import Editor

LONG_SIDE = 512


def working_size(width: int, height: int) -> tuple[int, int]:
    scale = LONG_SIDE / max(width, height)
    return max(8, round(width * scale / 8) * 8), max(8, round(height * scale / 8) * 8)


class InstructAdd(Editor):
    repo = ""
    revision = ""
    steps, guidance, image_guidance = 50, 7.0, 1.5
    tasks = ("add_replace",)
    ops = ("add",)
    uses_mask = False

    def __init__(self):
        self.model_id = f"{self.repo}@{self.revision[:7]}"
        self.settings = {"long_side": LONG_SIDE, "steps": self.steps, "guidance": self.guidance,
                         "image_guidance": self.image_guidance, "scheduler": "EulerAncestral",
                         "prompt": "instruction"}

    def load(self, device: str) -> None:
        import torch
        from diffusers import EulerAncestralDiscreteScheduler, StableDiffusionInstructPix2PixPipeline

        self.torch, self.device = torch, device
        self.pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            self.repo, revision=self.revision, torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            safety_checker=None, requires_safety_checker=False).to(device)
        self.pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.set_progress_bar_config(disable=True)

    def edit(self, image, mask, prompt: str, seed: int = 0):
        from PIL import Image

        generator = self.torch.Generator(device=self.device).manual_seed(seed)
        return self.pipe(prompt, image=image.resize(working_size(*image.size), Image.Resampling.BICUBIC),
                         num_inference_steps=self.steps, guidance_scale=self.guidance,
                         image_guidance_scale=self.image_guidance, generator=generator).images[0]


class PaintByInpaint(InstructAdd):
    name = "paint-by-inpaint"
    repo = "paint-by-inpaint/add-finetuned-mb"
    revision = "bb8c6752143ec9fadebe6d9612d38d524f41d35a"
    steps, guidance, image_guidance = 50, 7.0, 1.5


class EraseDraw(InstructAdd):
    name = "erasedraw"
    repo = "alpercanberk/erasedraw-v0.2"
    revision = "652876527032dd72d5b9d47cd9ed31670aa1f13d"
    steps, guidance, image_guidance = 10, 7.5, 1.0
