"""Plain code (PIL), the "sometimes the right answer" baseline of MODEL_LIST.md list 7: the format action of the
sample's plan (plan_v4, op "format") carried out exactly, with no model.

Actions: crop (to the plan's box), rotate (by the plan's degrees, counter-clockwise as plan_v4 asks, or clockwise
when the plan's direction says "clockwise": Qwen sometimes gives 90 with that direction; multiples of 90 turn the
whole image, other angles rotate about the centre on the same canvas, corners filled with white), flip (horizontal or
vertical), resize (by the plan's scale, LANCZOS, long side capped at 2,048 px) and grayscale. Other actions
(extend, cutout, other) and plans without the value they need are n/a. Crop, a quarter turn and resize change the
size, so their outputs keep it (`keep_size` False). CPU only; deterministic.
"""

from ..base import Editor

ACTIONS = ("crop", "rotate", "flip", "resize", "grayscale")
MAX_LONG_SIDE = 2048
FILL = (255, 255, 255)


def carry_out(image, plan: dict):
    """(edited image, whether the size changed on purpose) for one plan."""
    from PIL import Image, ImageOps

    action = plan["action"]
    if action == "crop":
        return image.crop(tuple(plan["box"])), True
    if action == "rotate":
        degrees = float(plan["degrees"])
        direction = plan.get("direction") or ""
        if "clockwise" in direction and not direction.startswith(("counter", "anti")):
            degrees = -abs(degrees)
        degrees %= 360
        if degrees % 90 == 0:
            return image.rotate(degrees, expand=True), degrees % 180 != 0
        return image.rotate(degrees, resample=Image.Resampling.BICUBIC, fillcolor=FILL), False
    if action == "flip":
        vertical = (plan.get("direction") or "horizontal").startswith("vert")
        return (ImageOps.flip(image) if vertical else ImageOps.mirror(image)), False
    if action == "resize":
        factor = min(float(plan["scale"]), MAX_LONG_SIDE / max(image.size))
        size = (max(1, round(image.width * factor)), max(1, round(image.height * factor)))
        return image.resize(size, Image.Resampling.LANCZOS), True
    if action == "grayscale":
        return image.convert("L").convert("RGB"), False
    raise ValueError(f"plain code cannot {action!r}")


class PlainCode(Editor):
    name = "plain-code"
    uses_plan = True
    model_id = "PIL"
    tasks = ("formatting",)
    uses_mask = False
    settings = {"actions": ACTIONS, "max_long_side": MAX_LONG_SIDE, "fill": "white"}

    def applies(self, sample, plan) -> bool:
        if sample["op"] != "format" or plan is None or plan.get("action") not in ACTIONS:
            return False
        needed = {"crop": "box", "rotate": "degrees", "resize": "scale"}.get(plan["action"])
        return needed is None or plan.get(needed) is not None

    def load(self, device: str) -> None:
        pass

    def edit(self, image, mask, prompt: str, seed: int = 0):
        edited, resized = carry_out(image, self.plan)
        self.keep_size = not resized
        return edited
