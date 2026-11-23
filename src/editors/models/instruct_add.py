"""Mask-free instruction editors: InstructPix2Pix-style models (SD 1.5, `StableDiffusionInstructPix2PixPipeline`,
fp16, no safety checker) that get the image and the simplified instruction alone and decide where to edit.
`InstructEdit` is the shared base (every task and op, the generalists of MODEL_LIST.md list 1); `InstructAdd`
limits it to the add samples (models trained to add objects, which place the new thing themselves: our VLM places
new things badly, list 3).

- Paint by Inpaint (Wasserman et al., CVPR 2025; MIT): trained on PIPE, 1M pairs made by removing objects from
  real photos and reversing the edit; `add-finetuned-mb` is the addition model fine-tuned on MagicBrush's
  additions. Settings of the authors' README: Euler Ancestral, 50 steps, guidance 7, image guidance 1.5.
- EraseDraw (Canberk et al., ECCV 2024; MIT): trained on pairs made by erasing objects; v0.2 of the authors' HF
  weights. Settings of its model card: Euler Ancestral (its own scheduler config), 10 steps, image guidance 1,
  guidance 7.5 (the pipeline default).
- InstructPix2Pix (Brooks et al., CVPR 2023; MIT), list 1 classic: the fp16 variant files (2.1 GB); the authors'
  defaults (edit_cli.py and the pipeline): 100 steps, guidance 7.5, image guidance 1.5, Euler Ancestral (the repo's
  own scheduler config). The HF README example (10 steps, image guidance 1) is a quick demo and is not used.
- MagicBrush (Zhang et al., NeurIPS 2023; CreativeML OpenRAIL-M), list 1 classic: vinesmsuic/magicbrush-jul7, the
  community diffusers port of the authors' "recent best" checkpoint (MagicBrush-epoch-52-step-4999.ckpt, linked
  from the authors' README; not the paper's epoch-000168), fp32 .bin only (4.3 GB), cast to fp16. Settings of the
  port's card: 20 steps, guidance 7, image guidance 1.5 (the authors' script: 100 steps, 7.5, 1.5).
- RealEdit (Sushko et al., CVPR 2025; code MIT, weights "cc"), list 1 classic: InstructPix2Pix fine-tuned on the
  RealEdit Reddit edits, the source of most of our samples. fp32 .bin only (4.3 GB), cast to fp16. Settings of the
  card and the authors' notebook: 50 steps, image guidance 2, guidance 7.5 (the pipeline default; the authors'
  inference.py passes an unused `text_guidance_scale`, so it ran at 7.5 too), long side 512 (inference.py).
  Caveat: it was trained on RealEdit's train split, whose photos may include some of our samples.
The .bin weights need torch >= 2.6 (transformers refuses torch.load below it).
All of them are SD 1.5 InstructPix2Pix models (`StableDiffusionInstructPix2PixPipeline`, fp16, no safety checker), so
the image is resized to a long side of 512 px (both sides a multiple of 8), the resolution they were trained at.
The addition models run on the add samples only (op "add"); replace and background samples are "n/a". The prompt is
the simplified instruction ("Add a camel behind us."); the output is not pasted back (the model chose the region).
"""

from .. import TASKS
from ..base import Editor

LONG_SIDE = 512


def working_size(width: int, height: int) -> tuple[int, int]:
    scale = LONG_SIDE / max(width, height)
    return max(8, round(width * scale / 8) * 8), max(8, round(height * scale / 8) * 8)


class InstructEdit(Editor):
    repo = ""
    revision = ""
    variant = None          # "fp16" when the repository ships *.fp16.* files next to the fp32 ones
    steps, guidance, image_guidance = 50, 7.0, 1.5
    tasks = TASKS
    ops = None
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
            self.repo, revision=self.revision, variant=self.variant,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            safety_checker=None, requires_safety_checker=False).to(device)
        self.pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.set_progress_bar_config(disable=True)

    def edit(self, image, mask, prompt: str, seed: int = 0):
        from PIL import Image

        generator = self.torch.Generator(device=self.device).manual_seed(seed)
        return self.pipe(prompt, image=image.resize(working_size(*image.size), Image.Resampling.BICUBIC),
                         num_inference_steps=self.steps, guidance_scale=self.guidance,
                         image_guidance_scale=self.image_guidance, generator=generator).images[0]


class InstructAdd(InstructEdit):
    tasks = ("add_replace",)
    ops = ("add",)


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


class InstructPix2Pix(InstructEdit):
    name = "instruct-pix2pix"
    repo = "timbrooks/instruct-pix2pix"
    revision = "31519b5cb02a7fd89b906d88731cd4d6a7bbf88d"
    variant = "fp16"
    steps, guidance, image_guidance = 100, 7.5, 1.5


class MagicBrush(InstructEdit):
    name = "magicbrush"
    repo = "vinesmsuic/magicbrush-jul7"
    revision = "d75a9f18439f67dbdc0ab113036f491ce2256c53"
    steps, guidance, image_guidance = 20, 7.0, 1.5


class RealEdit(InstructEdit):
    name = "realedit"
    repo = "peter-sushko/RealEdit"
    revision = "49ae0e0507fe7a5ee30397aadd491414aecea45f"
    steps, guidance, image_guidance = 50, 7.5, 2.0
