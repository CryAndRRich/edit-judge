"""Edit plans: what an inpainting model needs from a request, read by an open VLM (Qwen2.5-VL-7B by default).

One call per sample, with the original image, the simplified instruction, the request as written and the op of the
sample (remove, add, replace, background; samples.py), and the prompt prompts/plan_v<k>.txt. The answer is one JSON
object. Every thing is named by a phrase (for Grounding DINO) with Qwen's own box (used to choose among DINO's boxes
and as the fallback when DINO finds nothing; grounding.py):
- targets: what to remove or replace;
- keep (plan_v3): what must not be painted over: background, the subjects that stay; add, the people or things
  near the new one (plan_v2 put the background's subjects in `targets`; grounding.py reads both);
- anchor (plan_v3, add): the existing thing the new one goes on (a face for facial hair, a head for a hat): the
  region is around it, found like a target. Qwen places existing things well and new things poorly (the smoke run
  of 2026-11-20: facial hair boxed on the chest), so a free box (`bbox_2d`) is used only without an anchor;
- prompt: what the region shows after the edit (add, replace, background).

Prompt versions per task (TASK_VERSIONS): remove keeps plan_v2, which works for it, so its plans, masks and
outputs of the second smoke run are reused; add_replace uses plan_v3.

Coordinates. Qwen2.5-VL answers in absolute pixels of the image it saw, i.e. after its processor's smart_resize.
That size is taken from the processor itself (`image_grid_thw` x the patch size), not assumed: the model's
preprocessor_config.json sets max_pixels = 12,845,056, which overrides the `size` the client passes, so the
dataset's images (at most 1,024 px) are seen at about their own size, rounded to multiples of 28. (plan_v1 assumed
the client's 640 px limit and scaled boxes up to 1.4 x too much: fixed here.) Boxes are stored in original pixels
with `seen` (the size Qwen saw), so any box can be checked.

Output: work/editors/plans/<client>/plan_v<k>.jsonl, one row per sample: {"id", "op", "targets", "target_boxes",
"keep", "keep_boxes", "anchor", "anchor_box", "box", "prompt", "seen", "raw", "problems"}, written after every batch, so a run can be stopped and started again.
A prompt version used for a run is never edited; add plan_v3.txt instead.
"""

import json
import re
import time
from pathlib import Path
from string import Template

from ..common.device import check_heavy_run, pick_device
from ..common.io import append_jsonl, iter_jsonl
from ..paths import EDITORS_DIR, SAMPLES_DIR
from .samples import load_task_samples

PROMPTS_DIR = Path(__file__).parent / "prompts"
DEFAULT_CLIENT = "qwen2.5-vl-7b"
DEFAULT_VERSION = 3
TASK_VERSIONS = {"remove": 2, "add_replace": 3}
MAX_TARGETS = 5
MAX_TOKENS = 300
MIN_BOX = 8                     # pixels: a smaller box is not a usable region
BROKEN = ("no JSON object", "invalid JSON")


def plan_version(task: str) -> int:
    """The prompt version a task uses unless --prompt-version says otherwise."""
    return TASK_VERSIONS.get(task, DEFAULT_VERSION)


def prompt_id(version: int = DEFAULT_VERSION) -> str:
    return f"plan_v{version}"


def plans_path(client_name: str = DEFAULT_CLIENT, version: int = DEFAULT_VERSION) -> Path:
    return EDITORS_DIR / "plans" / client_name / f"{prompt_id(version)}.jsonl"


def load_plans(client_name: str = DEFAULT_CLIENT, version: int = DEFAULT_VERSION) -> dict[str, dict]:
    """Sample ID -> its plan (the last row wins, so a sample planned again replaces the earlier answer)."""
    return {row["id"]: row for row in iter_jsonl(plans_path(client_name, version))}


def build_prompt(sample: dict, version: int = DEFAULT_VERSION) -> str:
    template = Template((PROMPTS_DIR / f"{prompt_id(version)}.txt").read_text(encoding="utf-8"))
    return template.substitute(instruction=" ".join(sample["instruction"].split()).replace('"', "'"),
                               request=" ".join(sample["request"].split())[:600].replace('"', "'"),
                               operation=sample["op"])


def seen_size(processor, image) -> tuple[int, int]:
    """(width, height) of `image` as the model sees it, from the processor's own resize."""
    out = processor.image_processor(images=[image], return_tensors="pt")
    _, grid_h, grid_w = (int(v) for v in out["image_grid_thw"][0])
    patch = processor.image_processor.patch_size
    return grid_w * patch, grid_h * patch


def map_box(raw, image_size: tuple[int, int], seen: tuple[int, int]) -> list[int] | None:
    """A box of the model (pixels of the seen image) in original pixels, clipped to the image; None if unusable."""
    if isinstance(raw, list) and len(raw) == 1 and isinstance(raw[0], list):
        raw = raw[0]                                  # [[x1, y1, x2, y2]]
    try:
        x1, y1, x2, y2 = (float(v) for v in raw)
    except (TypeError, ValueError):
        return None
    width, height = image_size
    sx, sy = width / seen[0], height / seen[1]
    x1, x2 = sorted((max(0.0, min(width, x1 * sx)), max(0.0, min(width, x2 * sx))))
    y1, y2 = sorted((max(0.0, min(height, y1 * sy)), max(0.0, min(height, y2 * sy))))
    if x2 - x1 < MIN_BOX or y2 - y1 < MIN_BOX:
        return None
    return [round(x1), round(y1), round(x2), round(y2)]


def parse_plan(text: str, sample: dict, image_size: tuple[int, int], seen: tuple[int, int] | None = None) -> dict:
    """The plan of one answer; every deviation from the format is recorded in `problems`. `seen` is the size the
    model saw (default: the image's own size)."""
    seen = seen or image_size
    problems, data = [], None
    match = re.search(r"\{.*\}", text, re.S)
    if match:
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            problems.append("invalid JSON")
    else:
        problems.append("no JSON object")
    data = data if isinstance(data, dict) else {}
    def things(field: str) -> tuple[list[str], list]:
        """Phrases and boxes of a list field: [{"phrase", "bbox_2d"}] (plan_v2, v3) or phrases (plan_v1)."""
        items = data.get(field) or []
        if isinstance(items, (str, dict)):
            items = [items]
        phrases, boxes = [], []
        for item in items:
            phrase = item.get("phrase") if isinstance(item, dict) else item
            phrase = " ".join(str(phrase or "").split())
            if phrase:
                phrases.append(phrase)
                boxes.append(map_box(item.get("bbox_2d") or item.get("bbox"), image_size, seen)
                             if isinstance(item, dict) else None)
        if len(phrases) > MAX_TARGETS:
            problems.append(f"{len(phrases)} {field}, first {MAX_TARGETS} kept")
        if any(b is None for b in boxes[:MAX_TARGETS]):
            problems.append(f"{field} without a box")
        return phrases[:MAX_TARGETS], boxes[:MAX_TARGETS]

    targets, target_boxes = things("targets")
    keep, keep_boxes = things("keep")
    anchors, anchor_boxes = things("anchor")
    op = sample["op"]
    prompt = " ".join(str(data.get("prompt") or "").split())
    box = anchor = anchor_box = None
    if op == "add":
        targets, target_boxes = [], []
        if anchors:
            anchor, anchor_box = anchors[0], anchor_boxes[0]
        else:
            box = map_box(data.get("bbox_2d") or data.get("bbox") or data.get("box"), image_size, seen)
            if box is None:
                problems.append("no anchor and no box")
    elif op == "background":
        if not keep and not targets:
            problems.append("nothing to keep")
    elif not targets:
        problems.append("no targets")
    if op != "remove" and not prompt:
        problems.append("no prompt")
        prompt = sample["instruction"]                # better than nothing for the inpainting prompt
    if op == "remove":
        prompt = ""
    return {"id": sample["id"], "op": op, "targets": targets, "target_boxes": target_boxes, "keep": keep,
            "keep_boxes": keep_boxes, "anchor": anchor, "anchor_box": anchor_box, "box": box, "prompt": prompt,
            "seen": list(seen), "raw": text, "problems": problems}


def run(task: str, client_name: str = DEFAULT_CLIENT, version: int | None = None, limit: int | None = None,
        device: str | None = None, allow_cpu: bool = False) -> dict:
    from PIL import Image

    from ..instructions.clients import get_client

    version = version or plan_version(task)
    samples = load_task_samples(task, limit)
    out = plans_path(client_name, version)
    done = {row["id"] for row in iter_jsonl(out)}
    todo = [s for s in samples if s["id"] not in done]
    device = pick_device(device)
    check_heavy_run(device, len(todo), allow_cpu)
    if not todo:
        return {"task": task, "planned": 0, "already_done": len(samples)}
    client = get_client(client_name)
    started = time.monotonic()
    client.load(device)
    print(f"loaded {client_name} on {device} in {time.monotonic() - started:.0f} s", flush=True)
    started, problems = time.monotonic(), 0
    for start in range(0, len(todo), client.batch_size):
        batch = todo[start:start + client.batch_size]
        images = [SAMPLES_DIR / s["id"] / "original.jpg" for s in batch]
        prompts = [build_prompt(s, version) for s in batch]
        answers = client.generate_batch(prompts, images, MAX_TOKENS)
        rows = []
        for sample, prompt, image, answer in zip(batch, prompts, images, answers):
            with Image.open(image) as im:
                size, seen = im.size, seen_size(client.processor, im.convert("RGB"))
            row = parse_plan(answer, sample, size, seen)
            if len(batch) > 1 and any(p in row["problems"] for p in BROKEN):   # again alone, without padding
                again = parse_plan(client.generate(prompt, image, MAX_TOKENS), sample, size, seen)
                row = again if len(again["problems"]) < len(row["problems"]) else row
            problems += bool(row["problems"])
            rows.append(row)
        append_jsonl(out, rows)
        print(f"planned {start + len(batch)}/{len(todo)}", flush=True)
    seconds = time.monotonic() - started
    return {"task": task, "planned": len(todo), "with_problems": problems,
            "seconds_per_sample": round(seconds / len(todo), 2), "output": str(out)}
