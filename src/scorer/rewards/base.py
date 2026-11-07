"""The interface every reward model or VLM judge implements, and the runner that scores edits with it.

A reward model gives each edit a score from (original image, text, edited image): higher means better. The runner
scores the edits of one split (the test split by default) in a fixed order (samples sorted by a hash of their
ID, edits by key), so a run stopped by its time limit covers a well-mixed part of the split, and the next run goes
on where it stopped:
- scores go to work/scorer/baselines/<model>[.part-I-of-N].jsonl as {"image_id", "score", ...}, appended after
  every batch; edits already in any of the model's files (from this run or from an earlier session whose output
  was copied in) are skipped;
- `max_hours` stops before the next batch once that time has passed;
- `shard=(i, n)` takes every n-th edit (one process per GPU, for models that fit one GPU);
- images are decoded in threads while the model works on the previous batch.
<model>.meta.json records the device, the time per edit and the totals.
"""

import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageOps

from ...common.dataset import iter_samples, text_of
from ...common.device import check_heavy_run, pick_device
from ...common.io import append_jsonl, iter_jsonl, write_json
from ...paths import SAMPLES_DIR, SCORER_DIR
from ..data import load_pairs

BASELINES_DIR = SCORER_DIR / "baselines"


@dataclass(frozen=True)
class EditItem:
    image_id: str           # "<sample id>/<edit key>"
    original: Path
    edit: Path
    text: str               # the simplified instruction (or the request as written)


class RewardModel:
    name: str = ""          # short unique name; also the name of the score file
    model_id: str = ""      # where the weights and code come from (Hugging Face IDs, commits)
    batch_size: int = 8
    install: tuple[str, ...] = ()   # extra pip arguments, one `pip install` each; models with the same `install`
                                    # share one environment (src/common/envs.py)
    isolated: bool = False          # True: an environment that does not see the machine's packages

    def load(self, device: str) -> None:
        """Load the model onto `device`. Import heavy libraries here, so listing models needs none of them."""
        raise NotImplementedError

    def score(self, batch: list[tuple[EditItem, Image.Image, Image.Image]]) -> list[dict]:
        """For each (item, original image, edited image), RGB: {"score": float or None, ...details}."""
        raise NotImplementedError


def sample_order(sample_id: str) -> str:
    return hashlib.sha1(f"order:{sample_id}".encode()).hexdigest()


def edit_items(split: str = "test", text_kind: str = "instruction") -> list[EditItem]:
    """Every edit with a preference pair in `split`, in the fixed run order."""
    keys = {}
    for pair in load_pairs("all", split):
        keys.setdefault(pair["id"], set()).update((pair["a"], pair["b"]))
    samples = {sample["id"]: sample for sample in iter_samples() if sample["id"] in keys}
    items = []
    for sample_id in sorted(samples, key=sample_order):
        sample = samples[sample_id]
        for key in sorted(keys[sample_id]):
            items.append(EditItem(f"{sample_id}/{key}", SAMPLES_DIR / sample_id / "original.jpg",
                                  SAMPLES_DIR / sample_id / f"{key}.jpg", text_of(sample, text_kind)))
    return items


def score_files(name: str) -> list[Path]:
    return sorted(BASELINES_DIR.glob(f"{name}.jsonl")) + sorted(BASELINES_DIR.glob(f"{name}.part-*.jsonl"))


def scored_ids(name: str) -> set[str]:
    return {row["image_id"] for path in score_files(name) for row in iter_jsonl(path)}


def read_image(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


def gpu_info() -> dict:
    try:
        import torch
        if torch.cuda.is_available():
            return {"gpus": torch.cuda.device_count(), "gpu": torch.cuda.get_device_name(0)}
    except ImportError:
        pass
    return {"gpus": 0, "gpu": None}


def run(model: RewardModel, split: str = "test", text_kind: str = "instruction", limit: int | None = None,
        device: str | None = None, allow_cpu: bool = False, shard: tuple[int, int] | None = None,
        max_hours: float | None = None, workers: int = 4) -> dict:
    device = pick_device(device)
    items = edit_items(split, text_kind)
    if shard:
        items = items[shard[0]::shard[1]]
    done = scored_ids(model.name)
    todo = [item for item in items if item.image_id not in done]
    if limit is not None:
        todo = todo[:limit]
    summary = {"model": model.name, "split": split, "text": text_kind, "edits": len(items),
               "already_done": len(items) - len([item for item in items if item.image_id not in done])}
    if not todo:
        return {**summary, "scored": 0, "left": 0, "complete": True}
    check_heavy_run(device, len(todo), allow_cpu)
    started = time.monotonic()
    model.load(device)
    load_seconds = time.monotonic() - started
    print(f"loaded {model.name} on {device} in {load_seconds:.0f} s", flush=True)

    out = BASELINES_DIR / f"{model.name}{f'.part-{shard[0]}-of-{shard[1]}' if shard else ''}.jsonl"
    batches = [todo[start:start + model.batch_size] for start in range(0, len(todo), model.batch_size)]
    started, scored, failed = time.monotonic(), 0, 0
    deadline = started + max_hours * 3600 if max_hours is not None else None

    def decode(batch_items):
        return [(item, read_image(item.original), read_image(item.edit)) for item in batch_items]

    with ThreadPoolExecutor(workers) as pool:
        pending = [pool.submit(decode, batch_items) for batch_items in batches[:2]]
        for index, batch_items in enumerate(batches):
            if deadline is not None and time.monotonic() > deadline:
                print(f"{model.name}: time limit of {max_hours:.2f} h reached", flush=True)
                break
            batch = pending[index].result()
            if index + 2 < len(batches):
                pending.append(pool.submit(decode, batches[index + 2]))
            pending[index] = None
            results = model.score(batch)
            append_jsonl(out, ({"image_id": item.image_id, **result} for (item, _, _), result in zip(batch, results)))
            scored += len(batch)
            failed += sum(result.get("score") is None for result in results)
            if (index + 1) % max(1, 50 // model.batch_size) == 0 or index + 1 == len(batches):
                rate = scored / max(time.monotonic() - started, 1e-9)
                print(f"{model.name}: {scored}/{len(todo)} edits ({100 * scored // len(todo)}%), "
                      f"{1 / rate:.2f} s/edit, about {(len(todo) - scored) / rate / 3600:.1f} h left", flush=True)
    seconds = time.monotonic() - started
    meta_path = BASELINES_DIR / f"{model.name}.meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {"edits_scored": 0, "seconds": 0.0}
    meta.update(model=model.name, model_id=model.model_id, split=split, text=text_kind, device=device,
                batch_size=model.batch_size, **gpu_info(), edits_scored=meta["edits_scored"] + scored,
                seconds=round(meta["seconds"] + seconds, 1), load_seconds=round(load_seconds, 1),
                updated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"))
    meta["seconds_per_edit"] = round(meta["seconds"] / meta["edits_scored"], 4) if meta["edits_scored"] else None
    write_json(meta_path, meta)
    left = len(items) - len(scored_ids(model.name) & {item.image_id for item in items})
    return {**summary, "scored": scored, "without_score": failed, "left": left, "complete": left == 0,
            "seconds_per_edit": round(seconds / scored, 4) if scored else None}
