"""The interface every editing model implements, and the runner that applies one model to a task's samples.

Two kinds of editor:
- mask-based (`uses_mask = True`, inpainting): gets the original image (RGB, at most 1,024 px on the long side: the
  dataset is normalized), the mask of the sample (L, 255 = repaint; grounding.py) and the plan's prompt (what the
  region shows after the edit);
- mask-free (`uses_mask = False`, instruction editors such as Paint by Inpaint): gets the image and the simplified
  instruction ("Add a camel behind us.") and decides where to edit by itself, so it runs even when no mask was
  found, and its output is not pasted back.
`ops` limits an editor to some ops of the task (e.g. the addition models: "add"); the other samples are recorded as
"n/a" and left out of its results.

Each editor resizes to its own working size (MODEL_LIST.md's settings); the runner resizes the output back to the
original size, so every output can be compared with the original and with the human edits pixel for pixel.

Paste-back (PASTE_BACK, on for mask-based models): only the masked region of the output is kept, blended into the
original over a FEATHER px soft edge. Models that work at 512 px (RORem, ObjectClear, OSOR) would otherwise return
the whole photo resampled and blurred, and the scorer would judge that blur rather than the edit; it also undoes the
small changes a diffusion model's VAE makes outside the mask. It also cuts shadows a remover erases outside the
(15 px grown) mask, the same for every model. The setting is recorded with every output.

Output: work/editors/<model>/<sample id>.png and work/editors/<model>/runs.jsonl, one row per attempt: {"id",
"task", "plan", "status" (ok | failed | black | no-mask | n/a), "mask_digest", "uses_mask", "settings_version", "seconds",
"gpu_peak_gb", "settings", "error"}. An attempt counts only while it is current: same plan version, same
`settings_version` of the editor (raised when a model's settings change, so its outputs are made again) and, for a mask-based model, the
same mask (`mask_digest`, grounding.py); an output of a changed mask is made again. A current
ok, no-mask or n/a sample is not run again; failed and black ones are tried again by the next run. "black" catches
the fp16 failure of MODEL_LIST.md (black or NaN images). The first sample's time includes CUDA warm-up, so the
summary reports the median.
"""

import statistics
import time
import traceback
from datetime import datetime, timezone

from ..common.device import check_heavy_run, pick_device
from ..common.io import append_jsonl, iter_jsonl, write_json
from ..paths import EDITORS_DIR, SAMPLES_DIR
from .grounding import load_masks, mask_path
from .plan import DEFAULT_CLIENT, DEFAULT_VERSION, load_plans, plan_version, prompt_id
from .samples import load_task_samples

SEED = 0
PASTE_BACK = True
FEATHER = 3                 # pixels of soft edge where the edit meets the original
FINAL = ("ok", "no-mask", "n/a")
STATUSES = ("ok", "failed", "black", "no-mask", "n/a")


class Editor:
    name: str = ""          # short unique name; also the output folder
    model_id: str = ""      # where the weights come from (Hugging Face ID and revision, release URL, commit)
    tasks: tuple[str, ...] = ()     # the tasks of src/editors/__init__.py:TASKS it runs on
    ops: tuple[str, ...] | None = None   # the ops it runs on (None: every op of its tasks)
    uses_mask: bool = True  # False: an instruction editor, given the instruction and no mask
    settings_version: int = 1   # raise when the settings change: outputs of an older one are made again
    op: str | None = None   # the op of the sample being edited, set by the runner before each edit()
    install: tuple[str, ...] = ()   # extra pip arguments, one `pip install` each (src/common/envs.py)
    isolated: bool = False
    settings: dict = {}     # the settings used, recorded with every output

    def load(self, device: str) -> None:
        """Load the model onto `device`. Import heavy libraries here, not at the top of the module."""
        raise NotImplementedError

    def edit(self, image, mask, prompt: str, seed: int = SEED):
        """The edited image (PIL RGB, any size) of `image` (PIL RGB). Mask-based: `mask` (PIL L, same size) is the
        region and `prompt` what it shows after the edit; mask-free: `mask` is None and `prompt` the instruction."""
        raise NotImplementedError


def output_dir(name: str):
    return EDITORS_DIR / name


def output_path(name: str, sample_id: str):
    return output_dir(name) / f"{sample_id}.png"


def runs_path(name: str):
    return output_dir(name) / "runs.jsonl"


def load_runs(name: str, version: int = DEFAULT_VERSION) -> dict[str, dict]:
    """Sample ID -> the last current attempt of the model on it: made from a plan of prompt version `version` and,
    for a mask-based model, from the sample's current mask (attempts of an older plan or mask are made again)."""
    from . import get_editor
    settings_version = get_editor(name).settings_version
    digests = {sid: row.get("digest") for sid, row in load_masks(version).items()}
    rows = {}
    for row in iter_jsonl(runs_path(name)):
        if row.get("plan", "plan_v1") == prompt_id(version) and row.get("settings_version", 1) == settings_version:
            rows[row["id"]] = row
    return {sid: row for sid, row in rows.items()
            if not row.get("uses_mask", True) or row["status"] == "n/a"
            or (sid in digests and row.get("mask_digest") == digests[sid])}


def is_black(image) -> bool:
    """True for an all-black or flat image (the fp16 failure: NaN latents decode to black)."""
    low, high = zip(*image.getextrema())
    return max(high) <= 8 or all(h - l <= 2 for l, h in zip(low, high))


def paste_back(original, edited, mask):
    """`edited` inside `mask` (softened by FEATHER px), `original` elsewhere; all three the same size."""
    from PIL import Image, ImageFilter
    soft = mask.filter(ImageFilter.GaussianBlur(FEATHER)) if FEATHER else mask
    return Image.composite(edited, original, soft)


def gpu_peak():
    from ..scorer.latency import gpu_peak_gb, reset_gpu_peak
    return reset_gpu_peak, gpu_peak_gb


def run(editor: Editor, task: str, limit: int | None = None, device: str | None = None, allow_cpu: bool = False,
        client_name: str = DEFAULT_CLIENT, version: int | None = None) -> dict:
    from PIL import Image

    version = version or plan_version(task)
    if task not in editor.tasks:
        raise SystemExit(f"{editor.name} does not run on {task} (its tasks: {', '.join(editor.tasks)})")
    samples = load_task_samples(task, limit)
    plans, masks, runs = load_plans(client_name, version), load_masks(version), load_runs(editor.name, version)
    todo = [s for s in samples if runs.get(s["id"], {}).get("status") not in FINAL]
    applies = [s for s in todo if editor.ops is None or s["op"] in editor.ops]
    if editor.uses_mask:
        missing = [s["id"] for s in applies if s["id"] not in masks]
        if missing:
            raise SystemExit(f"{len(missing)} samples have no mask yet (e.g. {missing[0]}): run `masks` first")
        to_edit = [s for s in applies if masks[s["id"]]["status"] == "ok"]
    else:
        to_edit = applies
    device = pick_device(device)
    check_heavy_run(device, len(to_edit), allow_cpu)
    out = runs_path(editor.name)
    reset_peak, peak_gb = gpu_peak()

    def record(sample, status, seconds=None, peak=None, error=None):
        digest = masks.get(sample["id"], {}).get("digest") if editor.uses_mask else None
        append_jsonl(out, [{"id": sample["id"], "task": task, "plan": prompt_id(version), "status": status,
                            "mask_digest": digest, "uses_mask": editor.uses_mask, "settings_version": editor.settings_version,
                            "seconds": seconds,
                            "gpu_peak_gb": peak, "model_id": editor.model_id,
                            "settings": {**editor.settings, "paste_back": PASTE_BACK and editor.uses_mask,
                                         "feather": FEATHER},
                            "error": error, "at": datetime.now(timezone.utc).isoformat(timespec="seconds")}])

    for sample in todo:
        if editor.ops is not None and sample["op"] not in editor.ops:
            record(sample, "n/a")
        elif editor.uses_mask and masks[sample["id"]]["status"] != "ok":
            record(sample, "no-mask")
    if not to_edit:
        return summary(editor.name, task, samples, version)
    started = time.monotonic()
    editor.load(device)
    load_seconds = round(time.monotonic() - started, 1)
    print(f"loaded {editor.name} on {device} in {load_seconds:.0f} s", flush=True)
    for index, sample in enumerate(to_edit):
        with Image.open(SAMPLES_DIR / sample["id"] / "original.jpg") as im:
            image = im.convert("RGB")
        if editor.uses_mask:
            with Image.open(mask_path(sample["id"])) as im:
                mask = im.convert("L")
            prompt = plans[sample["id"]]["prompt"] if sample["id"] in plans else sample["instruction"]
        else:
            mask, prompt = None, sample["instruction"]
        editor.op = sample["op"]
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
        status = "black" if is_black(edited) else "ok"     # judged on the model's own output
        if PASTE_BACK and editor.uses_mask:
            edited = paste_back(image, edited, mask)
        output_path(editor.name, sample["id"]).parent.mkdir(parents=True, exist_ok=True)
        edited.save(output_path(editor.name, sample["id"]))
        record(sample, status, seconds, peak_gb())
        print(f"{editor.name} {index + 1}/{len(to_edit)} {sample['id']} {status} {seconds:.1f} s", flush=True)
    result = summary(editor.name, task, samples, version)
    write_json(output_dir(editor.name) / f"{task}.meta.json", {**result, "device": device,
                                                                "load_seconds": load_seconds})
    return result


def summary(name: str, task: str, samples: list[dict] | None = None, version: int = DEFAULT_VERSION) -> dict:
    """Counts by status, median seconds per image and the highest GPU peak of a model on a task's samples."""
    ids = {s["id"] for s in samples} if samples is not None else None
    rows = [r for r in load_runs(name, version).values() if r["task"] == task and (ids is None or r["id"] in ids)]
    counts = {status: sum(r["status"] == status for r in rows) for status in STATUSES}
    seconds = [r["seconds"] for r in rows if r["status"] == "ok" and r["seconds"] is not None]
    peaks = [r["gpu_peak_gb"] for r in rows if r.get("gpu_peak_gb") is not None]
    return {"model": name, "task": task, **counts,
            "median_seconds": round(statistics.median(seconds), 2) if seconds else None,
            "gpu_peak_gb": max(peaks) if peaks else None}
