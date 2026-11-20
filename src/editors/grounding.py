"""Masks for the inpainting models (MODEL_LIST.md, "Grounding").

For each target of the plan (remove, replace: what goes; background: the subjects that stay), two detectors agree
on its boxes:
- Grounding DINO base finds every box of the phrase (lowercase, ending with a dot; box threshold 0.4, text
  threshold 0.3);
- the plan's own box for the target (Qwen2.5-VL's grounding) says which of them are meant: DINO boxes whose centre
  lies in Qwen's box (grown by 10%) are kept, so "the letters DB" keeps both letters and "the man on the left"
  keeps one man. When DINO finds none there, Qwen's box is used (small things such as "the beer can"); when the
  plan has no box for the target, DINO's best box.
SAM 2.1 large turns the kept boxes into masks (`input_boxes`, `multimask_output=False`); the sample's mask is their
union. Then by op:
- remove: grown by 15 px (edges and shadows must go too); replace: grown by 8 px;
- background: the subjects' mask grown by 4 px (their edges stay intact), then inverted: everything else is
  repainted;
- add: the plan's box, filled (nothing to segment yet).
Both detectors run in float32 (0.9 GB each; T4 has no fast bfloat16). Every model gets the same mask.

Output: work/editors/masks/<sample id>.png (L, 255 = repaint) and masks.jsonl, one row per sample: {"id", "op",
"status" (ok | no-mask), "plan" (the prompt version of the plan), "boxes": [[x1, y1, x2, y2, score, phrase, source (dino | qwen | plan)]], "coverage" (share
of the image masked)}. A sample where nothing is found is "no-mask"; the editors skip it and record that.
"""

import inspect
import time

from ..common.device import check_heavy_run, pick_device
from ..common.io import append_jsonl, iter_jsonl
from ..paths import EDITORS_DIR, SAMPLES_DIR
from .plan import DEFAULT_CLIENT, DEFAULT_VERSION, load_plans, prompt_id
from .samples import load_task_samples

DINO_ID = "IDEA-Research/grounding-dino-base"
SAM_ID = "facebook/sam2.1-hiera-large"
BOX_THRESHOLD = 0.4
TEXT_THRESHOLD = 0.3
QWEN_BOX_MARGIN = 0.10          # a DINO box is meant when its centre lies in Qwen's box grown by this share
MAX_BOXES = 6
DILATE = {"remove": 15, "replace": 8, "add": 0, "background": 0}
KEEP_EDGE = 4                   # background: pixels added around the kept subjects before inverting
MASKS_DIR = EDITORS_DIR / "masks"
MASKS_INDEX = MASKS_DIR / "masks.jsonl"


def mask_path(sample_id: str):
    return MASKS_DIR / f"{sample_id}.png"


def load_masks(version: int = DEFAULT_VERSION) -> dict[str, dict]:
    """Sample ID -> its mask row, only masks made from plans of prompt version `version` (a mask of an older plan
    is made again; the last row wins)."""
    return {row["id"]: row for row in iter_jsonl(MASKS_INDEX) if row.get("plan", "plan_v1") == prompt_id(version)}


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


def inside(box: list[float], region: list[float], margin: float = QWEN_BOX_MARGIN) -> bool:
    """True when the centre of `box` lies in `region` grown by `margin` of its size on every side."""
    x1, y1, x2, y2 = region
    dx, dy = (x2 - x1) * margin, (y2 - y1) * margin
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    return x1 - dx <= cx <= x2 + dx and y1 - dy <= cy <= y2 + dy


def choose_boxes(found: list[list[float]], qwen_box: list[int] | None, phrase: str) -> list[list]:
    """The boxes of one target: [x1, y1, x2, y2, score, phrase, source]. `found` are DINO's [x1, y1, x2, y2,
    score], best first."""
    if qwen_box is not None:
        kept = [[*box, phrase, "dino"] for box in found if inside(box, qwen_box)]
        return kept[:MAX_BOXES] or [[*qwen_box, 1.0, phrase, "qwen"]]
    return [[*found[0], phrase, "dino"]] if found else []


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

    def find(self, image, phrase: str) -> list[list[float]]:
        """Every box of `phrase` above the thresholds, [x1, y1, x2, y2, score], best first."""
        inputs = self.dino_processor(images=image, text=dino_text(phrase), return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            outputs = self.dino(**inputs)
        result = self.dino_processor.post_process_grounded_object_detection(
            outputs, inputs["input_ids"], **{self.threshold_arg: BOX_THRESHOLD}, text_threshold=TEXT_THRESHOLD,
            target_sizes=[image.size[::-1]])[0]
        boxes = [[*(round(float(v), 1) for v in box), round(float(score), 3)]
                 for box, score in zip(result["boxes"].tolist(), result["scores"].tolist())]
        return sorted(boxes, key=lambda box: -box[4])

    def segment(self, image, boxes: list[list[float]]):
        """Union of SAM 2.1's masks of `boxes`, as PIL L."""
        from PIL import Image
        inputs = self.sam_processor(images=image, input_boxes=[[[float(v) for v in box[:4]] for box in boxes]],
                                    return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            outputs = self.sam(**inputs, multimask_output=False)
        masks = self.sam_processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"])[0]
        union = masks.reshape(-1, *masks.shape[-2:]).bool().any(0).numpy()
        return Image.fromarray(union.astype("uint8") * 255, mode="L")


def sample_mask(grounder, image, plan: dict):
    """(mask or None, boxes) of one plan."""
    from PIL import ImageOps

    op = plan["op"]
    if op == "add":
        if not plan["box"]:
            return None, []
        return box_mask(image.size, [plan["box"]]), [[*plan["box"], 1.0, "new", "plan"]]
    boxes = []
    qwen_boxes = plan.get("target_boxes") or [None] * len(plan["targets"])
    for phrase, qwen_box in zip(plan["targets"], qwen_boxes):
        boxes += choose_boxes(grounder.find(image, phrase), qwen_box, phrase)
    if not boxes:
        return None, []
    mask = grounder.segment(image, boxes)
    if op == "background" and mask.getbbox():
        mask = ImageOps.invert(dilate(mask, KEEP_EDGE))
    return mask, boxes


def run(task: str, client_name: str = DEFAULT_CLIENT, version: int = DEFAULT_VERSION, limit: int | None = None,
        device: str | None = None, allow_cpu: bool = False) -> dict:
    from PIL import Image

    samples = load_task_samples(task, limit)
    plans = load_plans(client_name, version)
    done = load_masks(version)
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
        if grounder is None and plan["op"] != "add":
            grounder = Grounder()
            grounder.load(device)
            print(f"loaded Grounding DINO and SAM 2.1 on {device}", flush=True)
        mask, boxes = sample_mask(grounder, image, plan)
        row = {"id": sample["id"], "op": plan["op"], "plan": prompt_id(version), "boxes": boxes}
        if mask is None or not mask.getbbox():
            row.update(status="no-mask", coverage=0.0)
        else:
            mask = dilate(mask, DILATE[plan["op"]])
            mask_path(sample["id"]).parent.mkdir(parents=True, exist_ok=True)
            mask.save(mask_path(sample["id"]))
            row.update(status="ok", coverage=round(mask.histogram()[255] / (image.width * image.height), 4))
        counts[row["status"]] += 1
        append_jsonl(MASKS_INDEX, [row])
        sources = ",".join(sorted({b[-1] for b in boxes})) or "-"
        print(f"mask {index + 1}/{len(todo)} {sample['id']} {plan['op']} {row['status']} {row['coverage']:.1%} "
              f"({len(boxes)} boxes: {sources})", flush=True)
    return {"task": task, **counts, "seconds": round(time.monotonic() - started, 1)}
