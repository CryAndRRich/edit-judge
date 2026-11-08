"""The interface every reward model or VLM judge implements, and the runner that scores edits with it.

A reward model gives each edit a score from (original image, text, edited image): higher means better. The runner
scores the edits of one split (the test split by default) in a fixed order (samples sorted by a hash of their
ID, edits by key), so a run stopped by its time limit covers a well-mixed part of the split, and the next run goes
on where it stopped:
- scores go to work/scorer/baselines/<model>[.part-I-of-N].jsonl as {"image_id", "score", ...}, appended after
  every batch; edits already in any of the model's files (from this run or from an earlier session whose output
  was copied in) are skipped;
- `max_hours` stops before the next batch once that time has passed;
- `shard=(i, n)` takes every n-th edit: one process per GPU for models that fit one GPU, or one Kaggle session
  (one team member) per part for the slow VLM judges; each part has its own score and meta files;
- images are decoded in threads while the model works on the previous batch;
- a result that failed inside a batch (no score, a score that is not finite, or an answer that could not be
  parsed) is scored again alone: float16 batches with padding can give a broken output ("!!!!" from a VLM, an
  overflow) that the same input alone does not. A score that is still not finite is written as null.
<model>[.part-I-of-N].meta.json records the device, the time per edit and the totals.

Latency (../latency.py): right after loading, the runner times `latency_pairs` requests, each the two edits of one
test pair in one `score` call, the way one user request would be answered (<model>.latency.json, once per model;
a smoke run with `limit` times SMOKE_LATENCY_PAIRS and a full run times them again).
"""

import io
import json
import math
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
from ..data import load_pairs, sample_order
from ..embed import part_suffix
from ..latency import WARMUP_PAIRS, gpu_peak_gb, latency_pairs, reset_gpu_peak, time_requests, timing_summary

BASELINES_DIR = SCORER_DIR / "baselines"
PROGRESS_SECONDS = 60            # a progress line after the first 3 batches, then at least once a minute
SMOKE_LATENCY_PAIRS = 3


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
    latency_pairs: int = 50         # requests timed for the latency (fewer for slow models)

    def load(self, device: str) -> None:
        """Load the model onto `device`. Import heavy libraries here, so listing models needs none of them."""
        raise NotImplementedError

    def score(self, batch: list[tuple[EditItem, Image.Image, Image.Image]]) -> list[dict]:
        """For each (item, original image, edited image), RGB: {"score": float or None, ...details}."""
        raise NotImplementedError


def is_broken(result: dict) -> bool:
    score = result.get("score")
    return score is None or not math.isfinite(score) or result.get("parsed") is False


def score_batch(model: RewardModel, batch: list) -> tuple[list[dict], int]:
    """The model's results for a batch, with every broken result scored again alone; and how many were."""
    results = model.score(batch)
    retried = 0
    if len(batch) > 1:
        for position, result in enumerate(results):
            if is_broken(result):
                retried += 1
                again = model.score([batch[position]])[0]
                if not is_broken(again):
                    results[position] = again
    for result in results:
        if result.get("score") is not None and not math.isfinite(result["score"]):
            result["score"] = None
    return results, retried


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


def measure_latency(model: RewardModel, text_kind: str, count: int, device: str) -> dict | None:
    """Time `count` requests (two edits of one test pair each, in one `score` call) unless an earlier run timed at
    least as many; writes <model>.latency.json."""
    path = BASELINES_DIR / f"{model.name}.latency.json"
    if path.exists() and json.loads(path.read_text())["pairs"] >= count:
        return None
    requests = latency_pairs(count + WARMUP_PAIRS, text_kind)

    def answer(request):
        sample_id, keys = request["id"], request["keys"]
        original = read_image(io.BytesIO(request["bytes"]["original"]))
        model.score([(EditItem(f"{sample_id}/{key}", SAMPLES_DIR / sample_id / "original.jpg",
                               SAMPLES_DIR / sample_id / f"{key}.jpg", request["text"]),
                      original, read_image(io.BytesIO(request["bytes"][key]))) for key in keys])

    print(f"latency: timing {model.name} on {len(requests)} requests ({WARMUP_PAIRS} warm-up)", flush=True)
    reset_gpu_peak()
    result = {"model": model.name, "text": text_kind, "batch": "the 2 edits of one request",
              **timing_summary(time_requests(requests, answer)), "peak_gpu_memory_gb": gpu_peak_gb(),
              "device": device, **gpu_info(), "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    write_json(path, result)
    print(f"latency: {model.name}: median {result['median_ms']} ms, p90 {result['p90_ms']} ms, "
          f"peak GPU memory {result['peak_gpu_memory_gb']} GB", flush=True)
    return result


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
    print(f"loading {model.name}: {model.model_id} (the weights are downloaded on first use)", flush=True)
    started = time.monotonic()
    model.load(device)
    load_seconds = time.monotonic() - started
    print(f"loaded {model.name} on {device} in {load_seconds:.0f} s", flush=True)
    if not shard or shard[0] == 0:
        measure_latency(model, text_kind, SMOKE_LATENCY_PAIRS if limit is not None else model.latency_pairs, device)

    out = BASELINES_DIR / f"{model.name}{part_suffix(shard)}.jsonl"
    batches = [todo[start:start + model.batch_size] for start in range(0, len(todo), model.batch_size)]
    started, scored, failed, unparsed, retried = time.monotonic(), 0, 0, 0, 0
    last_print = started
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
            results, batch_retried = score_batch(model, batch)
            retried += batch_retried
            append_jsonl(out, ({"image_id": item.image_id, **result} for (item, _, _), result in zip(batch, results)))
            scored += len(batch)
            failed += sum(result.get("score") is None for result in results)
            unparsed += sum(result.get("parsed") is False for result in results)
            if index < 3 or time.monotonic() - last_print >= PROGRESS_SECONDS or index + 1 == len(batches):
                last_print = time.monotonic()
                rate = scored / max(last_print - started, 1e-9)
                print(f"{model.name}: {scored}/{len(todo)} edits ({100 * scored // len(todo)}%), "
                      f"{1 / rate:.2f} s/edit, about {(len(todo) - scored) / rate / 3600:.1f} h left; "
                      f"retried alone {retried}, no score {failed}, unparsed {unparsed}", flush=True)
    seconds = time.monotonic() - started
    meta_path = BASELINES_DIR / f"{model.name}{part_suffix(shard)}.meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {"edits_scored": 0, "seconds": 0.0}
    for key, count in (("retried_alone", retried), ("without_score", failed), ("unparsed", unparsed)):
        meta[key] = meta.get(key, 0) + count
    meta.update(model=model.name, model_id=model.model_id, split=split, text=text_kind, device=device,
                batch_size=model.batch_size, **gpu_info(), edits_scored=meta["edits_scored"] + scored,
                seconds=round(meta["seconds"] + seconds, 1), load_seconds=round(load_seconds, 1),
                updated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"))
    meta["seconds_per_edit"] = round(meta["seconds"] / meta["edits_scored"], 4) if meta["edits_scored"] else None
    write_json(meta_path, meta)
    left = len(items) - len(scored_ids(model.name) & {item.image_id for item in items})
    return {**summary, "scored": scored, "without_score": failed, "unparsed": unparsed, "retried_alone": retried,
            "left": left, "complete": left == 0,
            "seconds_per_edit": round(seconds / scored, 4) if scored else None}
