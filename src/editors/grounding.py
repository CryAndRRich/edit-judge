"""Masks for the inpainting models (MODEL_LIST.md, "Grounding").

- remove, replace: Grounding DINO base finds each target phrase of the plan (lowercase, ending with a dot; box
  threshold 0.4, text threshold 0.3) and keeps the best box of each phrase; SAM 2.1 large turns the boxes into
  masks (`input_boxes`, `multimask_output=False`), and the mask of the sample is their union.
- add: the box of the plan, filled (nothing to segment yet).
The mask is then grown by DILATE pixels (more for remove: edges and shadows must go too), so every model gets the
same mask. Both detectors run in float32 (0.9 GB each; T4 has no fast bfloat16).

Output: work/editors/masks/<sample id>.png (L, 255 = repaint) and masks.jsonl, one row per sample: {"id", "status"
(ok | no-mask | no-plan), "boxes": [[x1, y1, x2, y2, score, phrase]], "coverage" (share of the image masked)}. A
sample without a found target is "no-mask"; the editors skip it and record that.
"""

import inspect
import time

from ..common.device import check_heavy_run, pick_device
from ..common.io import append_jsonl, iter_jsonl
from ..paths import EDITORS_DIR, SAMPLES_DIR
from .plan import DEFAULT_CLIENT, DEFAULT_VERSION, load_plans
from .samples import load_task_samples

DINO_ID = "IDEA-Research/grounding-dino-base"
SAM_ID = "facebook/sam2.1-hiera-large"
BOX_THRESHOLD = 0.4
TEXT_THRESHOLD = 0.3
DILATE = {"remove": 15, "replace": 8, "add": 0}
MASKS_DIR = EDITORS_DIR / "masks"
MASKS_INDEX = MASKS_DIR / "masks.jsonl"


def mask_path(sample_id: str):
    return MASKS_DIR / f"{sample_id}.png"


def load_masks() -> dict[str, dict]:
    return {row["id"]: row for row in iter_jsonl(MASKS_INDEX)}


def dino_text(phrase: str) -> str:
    """Grounding DINO wants lowercase text that ends with a dot."""
    return phrase.lower().strip().rstrip(".") + "."


def dilate(mask, pixels: int):
    """Grow a binary mask (PIL L) by `pixels` with repeated 3 x 3 maximum filters."""
    from PIL import ImageFilter
    for _ in range(pixels):
        mask = mask.filter(ImageFilter.MaxFilter(3))
    return mask


def box_mask(size: tuple[int, int], boxes: list[list[float]]):
    from PIL import Image, ImageDraw
    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)
    for x1, y1, x2, y2 in boxes:
        draw.rectangle([x1, y1, x2, y2], fill=255)
    return mask


class Grounder:
    def load(self, device: str) -> None:
        import torch
        from transformers import (AutoModelForZeroShotObjectDetection, AutoProcessor, Sam2Model,
                                  Sam2Processor)

        self.torch, self.device = torch, device
        self.dino_processor = AutoProcessor.from_pretrained(DINO_ID)
        self.dino = AutoModelForZeroShotObjectDetection.from_pretrained(DINO_ID).to(device).eval()
        self.sam_processor = Sam2Processor.from_pretrained(SAM_ID)
        self.sam = Sam2Model.from_pretrained(SAM_ID, dtype=torch.float32).to(device).eval()
        # the name of the box threshold changed across transformers versions (box_threshold -> threshold)
        params = inspect.signature(self.dino_processor.post_process_grounded_object_detection).parameters
        self.threshold_arg = "box_threshold" if "box_threshold" in params else "threshold"

    def find(self, image, phrase: str) -> list[float] | None:
        """[x1, y1, x2, y2, score] of the best box for `phrase`, or None."""
        inputs = self.dino_processor(images=image, text=dino_text(phrase), return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            outputs = self.dino(**inputs)
        result = self.dino_processor.post_process_grounded_object_detection(
            outputs, inputs["input_ids"], **{self.threshold_arg: BOX_THRESHOLD}, text_threshold=TEXT_THRESHOLD,
            target_sizes=[image.size[::-1]])[0]
        if len(result["scores"]) == 0:
            return None
        best = int(result["scores"].argmax())
        return [*(round(float(v), 1) for v in result["boxes"][best].tolist()), round(float(result["scores"][best]), 3)]

    def segment(self, image, boxes: list[list[float]]):
        """Union of SAM 2.1's masks of `boxes`, as PIL L."""
        from PIL import Image
        inputs = self.sam_processor(images=image, input_boxes=[[box[:4] for box in boxes]],
                                    return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            outputs = self.sam(**inputs, multimask_output=False)
        masks = self.sam_processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"])[0]
        union = masks.reshape(-1, *masks.shape[-2:]).bool().any(0).numpy()
        return Image.fromarray(union.astype("uint8") * 255, mode="L")


def run(task: str, client_name: str = DEFAULT_CLIENT, version: int = DEFAULT_VERSION, limit: int | None = None,
        device: str | None = None, allow_cpu: bool = False) -> dict:
    from PIL import Image

    samples = load_task_samples(task, limit)
    plans = load_plans(client_name, version)
    done = load_masks()
    todo = [s for s in samples if s["id"] not in done]
    device = pick_device(device)
    check_heavy_run(device, len(todo), allow_cpu)
    counts = {"ok": 0, "no-mask": 0, "no-plan": 0}
    if not todo:
        return {"task": task, "masked": 0, "already_done": len(samples)}
    grounder = None
    started = time.monotonic()
    for index, sample in enumerate(todo):
        plan = plans.get(sample["id"])
        if plan is None:
            counts["no-plan"] += 1
            print(f"{sample['id']}: no plan; run `plan` first", flush=True)
            continue                                  # not recorded: a later run picks it up
        with Image.open(SAMPLES_DIR / sample["id"] / "original.jpg") as im:
            image = im.convert("RGB")
        boxes, mask = [], None
        if plan["op"] == "add":
            if plan["box"]:
                boxes = [[*plan["box"], 1.0, "plan"]]
                mask = box_mask(image.size, [plan["box"]])
        elif plan["targets"]:
            if grounder is None:
                grounder = Grounder()
                grounder.load(device)
                print(f"loaded Grounding DINO and SAM 2.1 on {device}", flush=True)
            for phrase in plan["targets"]:
                found = grounder.find(image, phrase)
                if found:
                    boxes.append([*found, phrase])
            if boxes:
                mask = grounder.segment(image, boxes)
        row = {"id": sample["id"], "op": plan["op"], "boxes": boxes}
        if mask is None or not mask.getbbox():
            row.update(status="no-mask", coverage=0.0)
        else:
            mask = dilate(mask, DILATE[plan["op"]])
            mask_path(sample["id"]).parent.mkdir(parents=True, exist_ok=True)
            mask.save(mask_path(sample["id"]))
            histogram = mask.histogram()
            row.update(status="ok", coverage=round(histogram[255] / (image.width * image.height), 4))
        counts[row["status"]] += 1
        append_jsonl(MASKS_INDEX, [row])
        print(f"mask {index + 1}/{len(todo)} {sample['id']} {row['status']} {row['coverage']:.1%}", flush=True)
    return {"task": task, **counts, "seconds": round(time.monotonic() - started, 1)}
