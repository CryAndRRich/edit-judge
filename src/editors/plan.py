"""Edit plans: what an inpainting model needs from a request, read by an open VLM (Qwen2.5-VL-7B by default).

One call per sample, with the original image, the simplified instruction, the request as written and the operation
of its category (remove, add, replace), and the prompt prompts/plan_v<k>.txt. The answer is one JSON object:
- targets: noun phrases for Grounding DINO (remove, replace);
- bbox_2d: for add, where the new thing goes. Qwen2.5-VL answers in absolute pixels of the image it saw, which the
  processor resized (smart_resize, between MIN_PIXELS and MAX_PIXELS of the client); the box is mapped back to the
  original image here and stored as `box` in original pixels;
- prompt: what to paint (add, replace).

Output: work/editors/plans/<client>/plan_v<k>.jsonl, one row per sample: {"id", "op", "targets", "box", "prompt",
"raw", "problems"}, written after every batch, so a run can be stopped and started again. A prompt version used for
a run is never edited; add plan_v2.txt instead.
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
DEFAULT_VERSION = 1
MAX_TARGETS = 3
MAX_TOKENS = 200
BROKEN = ("no JSON object", "invalid JSON")


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


def seen_size(width: int, height: int) -> tuple[int, int]:
    """(width, height) of the image as Qwen2.5-VL saw it: the processor's smart_resize with the client's pixel
    limits (src/instructions/clients/qwen_vl.py)."""
    from ..instructions.clients.qwen_vl import MAX_PIXELS, MIN_PIXELS
    try:
        from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize
    except ImportError:                       # same rule as transformers, for machines without it (tests)
        def smart_resize(h, w, factor=28, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS):
            import math
            h_bar, w_bar = max(factor, round(h / factor) * factor), max(factor, round(w / factor) * factor)
            if h_bar * w_bar > max_pixels:
                beta = math.sqrt(h * w / max_pixels)
                h_bar, w_bar = math.floor(h / beta / factor) * factor, math.floor(w / beta / factor) * factor
            elif h_bar * w_bar < min_pixels:
                beta = math.sqrt(min_pixels / (h * w))
                h_bar, w_bar = math.ceil(h * beta / factor) * factor, math.ceil(w * beta / factor) * factor
            return h_bar, w_bar
    seen_h, seen_w = smart_resize(height, width, factor=28, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS)
    return seen_w, seen_h


def parse_plan(text: str, sample: dict, image_size: tuple[int, int]) -> dict:
    """The plan of one answer; every deviation from the format is recorded in `problems`."""
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
    targets = data.get("targets") or []
    if isinstance(targets, str):
        targets = [targets]
    targets = [" ".join(str(t).split()) for t in targets if str(t).strip()]
    if len(targets) > MAX_TARGETS:
        problems.append(f"{len(targets)} targets, first {MAX_TARGETS} kept")
        targets = targets[:MAX_TARGETS]
    prompt = " ".join(str(data.get("prompt") or "").split())
    box = None
    raw_box = data.get("bbox_2d") or data.get("bbox") or data.get("box")
    if isinstance(raw_box, list) and len(raw_box) == 1 and isinstance(raw_box[0], list):
        raw_box = raw_box[0]                          # [[x1, y1, x2, y2]]
    if sample["op"] == "add":
        try:
            x1, y1, x2, y2 = (float(v) for v in raw_box)
        except (TypeError, ValueError):
            problems.append("no box")
        else:
            width, height = image_size
            seen_w, seen_h = seen_size(width, height)
            sx, sy = width / seen_w, height / seen_h
            x1, x2 = sorted((max(0.0, min(width, x1 * sx)), max(0.0, min(width, x2 * sx))))
            y1, y2 = sorted((max(0.0, min(height, y1 * sy)), max(0.0, min(height, y2 * sy))))
            if x2 - x1 < 8 or y2 - y1 < 8:
                problems.append("box too small")
            else:
                box = [round(x1), round(y1), round(x2), round(y2)]
    elif not targets:
        problems.append("no targets")
    if sample["op"] in ("add", "replace") and not prompt:
        problems.append("no prompt")
        prompt = sample["instruction"]                # better than nothing for the inpainting prompt
    if sample["op"] == "remove":
        prompt = ""
    return {"id": sample["id"], "op": sample["op"], "targets": targets if sample["op"] != "add" else [],
            "box": box, "prompt": prompt, "raw": text, "problems": problems}


def run(task: str, client_name: str = DEFAULT_CLIENT, version: int = DEFAULT_VERSION, limit: int | None = None,
        device: str | None = None, allow_cpu: bool = False) -> dict:
    from PIL import Image

    from ..instructions.clients import get_client

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
                size = im.size
            row = parse_plan(answer, sample, size)
            if len(batch) > 1 and any(p in row["problems"] for p in BROKEN):   # again alone, without padding
                again = parse_plan(client.generate(prompt, image, MAX_TOKENS), sample, size)
                row = again if len(again["problems"]) < len(row["problems"]) else row
            problems += bool(row["problems"])
            rows.append(row)
        append_jsonl(out, rows)
        print(f"planned {start + len(batch)}/{len(todo)}", flush=True)
    seconds = time.monotonic() - started
    return {"task": task, "planned": len(todo), "with_problems": problems,
            "seconds_per_sample": round(seconds / len(todo), 2), "output": str(out)}
