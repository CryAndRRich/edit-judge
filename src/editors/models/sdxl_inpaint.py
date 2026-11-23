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

List 6 (change_color, op recolor, 2026-11-24): the same settings; the mask is the segmented thing whose colour
changes and the prompt the thing in its new colour (plan_v4). List 7 (formatting, the "extend" action):
`SDXLOutpaint`, below.
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
    tasks = ("add_replace", "change_color")
    ops = ("add", "replace", "background", "recolor")
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


EXTEND_SCALE = 1.5          # new canvas size relative to the old one when the plan gives no usable scale
MAX_EXTEND_SCALE = 3.0
OVERLAP = 16                # pixels of the original repainted along the seam, so the new part blends in


def extended_canvas(image, direction: str, scale: float):
    """(canvas with the image placed and the new border filled by its edge pixels, mask of the new border plus
    OVERLAP px of the seam, box of the image on the canvas)."""
    from PIL import Image, ImageDraw

    grow_x = direction in ("left", "right", "horizontal", "all")
    grow_y = direction in ("top", "bottom", "vertical", "all")
    if not (grow_x or grow_y):
        grow_x = grow_y = True
    width = round(image.width * scale) if grow_x else image.width
    height = round(image.height * scale) if grow_y else image.height
    left = {"left": width - image.width, "right": 0}.get(direction, (width - image.width) // 2)
    top = {"top": height - image.height, "bottom": 0}.get(direction, (height - image.height) // 2)
    canvas = image.resize((width, height), Image.Resampling.BICUBIC)   # a rough fill, repainted anyway
    canvas.paste(image, (left, top))
    mask = Image.new("L", (width, height), 255)
    inner = [left + (OVERLAP if left else 0), top + (OVERLAP if top else 0),
             left + image.width - (OVERLAP if left + image.width < width else 0),
             top + image.height - (OVERLAP if top + image.height < height else 0)]
    ImageDraw.Draw(mask).rectangle([inner[0], inner[1], inner[2] - 1, inner[3] - 1], fill=0)
    return canvas, mask, (left, top, left + image.width, top + image.height)


class SDXLOutpaint(SDXLInpaint):
    """Outpainting with SDXL Inpainting 0.1 (MODEL_LIST.md list 7: our own padding, the card does not document this
    use): the canvas grows by the plan's scale (EXTEND_SCALE when it gives none, at most MAX_EXTEND_SCALE) in the
    plan's direction, the new border (and OVERLAP px of the seam) is painted from the plan's prompt (the whole scene)
    at strength 1.0, and the original pixels are put back where they were. The output keeps the new size."""
    name = "sdxl-outpaint"
    tasks = ("formatting",)
    ops = ("format",)
    uses_mask = False
    settings_version = 1
    settings = {"long_side": LONG_SIDE, "steps": 20, "guidance": 8.0, "strength": 1.0, "overlap": OVERLAP,
                "default_scale": EXTEND_SCALE, "max_scale": MAX_EXTEND_SCALE, "prompt": "plan"}

    def applies(self, sample, plan) -> bool:
        return sample["op"] == "format" and plan is not None and plan.get("action") == "extend"

    def edit(self, image, mask, prompt: str, seed: int = 0):
        from PIL import Image

        scale = self.plan.get("scale") or EXTEND_SCALE
        scale = min(max(scale, 1.1), MAX_EXTEND_SCALE)
        canvas, border, _ = extended_canvas(image, self.plan.get("direction") or "all", scale)
        width, height = working_size(*canvas.size)
        generator = self.torch.Generator(device=self.device).manual_seed(seed)
        painted = self.pipe(prompt=self.plan.get("prompt") or prompt,
                            image=canvas.resize((width, height), Image.Resampling.BICUBIC),
                            mask_image=border.resize((width, height), Image.Resampling.NEAREST), height=height,
                            width=width, guidance_scale=8.0, num_inference_steps=20, strength=1.0,
                            generator=generator).images[0].resize(canvas.size, Image.Resampling.LANCZOS)
        self.keep_size = False
        return Image.composite(painted, canvas, border)     # the original's pixels outside the border and seam
