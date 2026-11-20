"""The interface every editing model implements, and the runner that applies one model to a task's samples.

An editor gets the original image (RGB, at most 1,024 px on the long side: the dataset is normalized), the mask of
the sample (L, 255 = repaint; grounding.py) and the inpainting prompt of the plan, and returns the edited image.
Each editor resizes to its own working size (MODEL_LIST.md's settings); the runner resizes the output back to the
original size, so every output can be compared with the original and with the human edits pixel for pixel.

Output: work/editors/<model>/<sample id>.png and work/editors/<model>/runs.jsonl, one row per attempt: {"id",
"task", "status" (ok | failed | black | no-mask), "seconds", "gpu_peak_gb", "settings", "error"}. A sample with
status ok or no-mask is not run again; failed and black ones are tried again by the next run. "black" catches the
fp16 failure of MODEL_LIST.md (black or NaN images). The first sample's time includes CUDA warm-up, so the summary
reports the median.
"""

import statistics
import time
import traceback
from datetime import datetime, timezone

from ..common.device import check_heavy_run, pick_device
from ..common.io import append_jsonl, iter_jsonl, write_json
from ..paths import EDITORS_DIR, SAMPLES_DIR
from .grounding import load_masks, mask_path
from .plan import DEFAULT_CLIENT, DEFAULT_VERSION, load_plans
from .samples import load_task_samples

SEED = 0
FINAL = ("ok", "no-mask")


class Editor:
    name: str = ""          # short unique name; also the output folder
    model_id: str = ""      # where the weights come from (Hugging Face ID and revision, release URL, commit)
    tasks: tuple[str, ...] = ()     # the tasks of src/editors/__init__.py:TASKS it runs on
    install: tuple[str, ...] = ()   # extra pip arguments, one `pip install` each (src/common/envs.py)
    isolated: bool = False
    settings: dict = {}     # the settings used, recorded with every output

    def load(self, device: str) -> None:
        """Load the model onto `device`. Import heavy libraries here, not at the top of the module."""
        raise NotImplementedError

    def edit(self, image, mask, prompt: str, seed: int = SEED):
        """The edited image (PIL RGB, any size) of `image` (PIL RGB) inside `mask` (PIL L, same size)."""
        raise NotImplementedError


def output_dir(name: str):
    return EDITORS_DIR / name


def output_path(name: str, sample_id: str):
    return output_dir(name) / f"{sample_id}.png"


def runs_path(name: str):
    return output_dir(name) / "runs.jsonl"


def load_runs(name: str) -> dict[str, dict]:
    """Sample ID -> the last attempt of the model on it."""
    return {row["id"]: row for row in iter_jsonl(runs_path(name))}


def is_black(image) -> bool:
    """True for an all-black or flat image (the fp16 failure: NaN latents decode to black)."""
    low, high = zip(*image.getextrema())
    return max(high) <= 8 or all(h - l <= 2 for l, h in zip(low, high))


def gpu_peak():
    from ..scorer.latency import gpu_peak_gb, reset_gpu_peak
    return reset_gpu_peak, gpu_peak_gb


def run(editor: Editor, task: str, limit: int | None = None, device: str | None = None, allow_cpu: bool = False,
        client_name: str = DEFAULT_CLIENT, version: int = DEFAULT_VERSION) -> dict:
    from PIL import Image

    if task not in editor.tasks:
        raise SystemExit(f"{editor.name} does not run on {task} (its tasks: {', '.join(editor.tasks)})")
    samples = load_task_samples(task, limit)
    plans, masks, runs = load_plans(client_name, version), load_masks(), load_runs(editor.name)
    todo = [s for s in samples if runs.get(s["id"], {}).get("status") not in FINAL]
    missing = [s["id"] for s in todo if s["id"] not in masks]
    if missing:
        raise SystemExit(f"{len(missing)} samples have no mask yet (e.g. {missing[0]}): run `masks` first")
    device = pick_device(device)
    to_edit = [s for s in todo if masks[s["id"]]["status"] == "ok"]
    check_heavy_run(device, len(to_edit), allow_cpu)
    out = runs_path(editor.name)
    reset_peak, peak_gb = gpu_peak()

    def record(sample, status, seconds=None, peak=None, error=None):
        append_jsonl(out, [{"id": sample["id"], "task": task, "status": status, "seconds": seconds,
                            "gpu_peak_gb": peak, "model_id": editor.model_id, "settings": editor.settings,
                            "error": error, "at": datetime.now(timezone.utc).isoformat(timespec="seconds")}])

    for sample in todo:
        if masks[sample["id"]]["status"] != "ok":
            record(sample, "no-mask")
    if not to_edit:
        return summary(editor.name, task, samples)
    started = time.monotonic()
    editor.load(device)
    load_seconds = round(time.monotonic() - started, 1)
    print(f"loaded {editor.name} on {device} in {load_seconds:.0f} s", flush=True)
    for index, sample in enumerate(to_edit):
        with Image.open(SAMPLES_DIR / sample["id"] / "original.jpg") as im:
            image = im.convert("RGB")
        with Image.open(mask_path(sample["id"])) as im:
            mask = im.convert("L")
        prompt = plans[sample["id"]]["prompt"] if sample["id"] in plans else sample["instruction"]
        reset_peak()
        started = time.perf_counter()
        try:
            edited = editor.edit(image, mask, prompt, SEED).convert("RGB")
        except Exception as error:                    # one bad sample must not stop the run
            seconds = round(time.perf_counter() - started, 2)
            record(sample, "failed", seconds, peak_gb(), f"{type(error).__name__}: {error}"[:500])
            print(f"{editor.name} {index + 1}/{len(to_edit)} {sample['id']} FAILED: {error}", flush=True)
            traceback.print_exc()
            if "out of memory" in str(error).lower():
                import torch
                torch.cuda.empty_cache()
            continue
        seconds = round(time.perf_counter() - started, 2)
        if edited.size != image.size:
            edited = edited.resize(image.size, Image.Resampling.LANCZOS)
        status = "black" if is_black(edited) else "ok"
        output_path(editor.name, sample["id"]).parent.mkdir(parents=True, exist_ok=True)
        edited.save(output_path(editor.name, sample["id"]))
        record(sample, status, seconds, peak_gb())
        print(f"{editor.name} {index + 1}/{len(to_edit)} {sample['id']} {status} {seconds:.1f} s", flush=True)
    result = summary(editor.name, task, samples)
    write_json(output_dir(editor.name) / f"{task}.meta.json", {**result, "device": device,
                                                                "load_seconds": load_seconds})
    return result


def summary(name: str, task: str, samples: list[dict] | None = None) -> dict:
    """Counts by status, median seconds per image and the highest GPU peak of a model on a task's samples."""
    ids = {s["id"] for s in samples} if samples is not None else None
    rows = [r for r in load_runs(name).values() if r["task"] == task and (ids is None or r["id"] in ids)]
    counts = {status: sum(r["status"] == status for r in rows) for status in ("ok", "failed", "black", "no-mask")}
    seconds = [r["seconds"] for r in rows if r["status"] == "ok" and r["seconds"] is not None]
    peaks = [r["gpu_peak_gb"] for r in rows if r.get("gpu_peak_gb") is not None]
    return {"model": name, "task": task, **counts,
            "median_seconds": round(statistics.median(seconds), 2) if seconds else None,
            "gpu_peak_gb": max(peaks) if peaks else None}
