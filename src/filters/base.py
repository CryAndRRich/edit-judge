"""The interface every detector implements, and the runner that scores images with it.

A detector returns, for each image, the probability that the image is positive for its task (sensitive content,
or a watermark). It runs on one of two image sets (`on`):
- "dataset": every image of the final dataset (data/images/);
- "benchmark": public sets with known labels (benchmark.py), to compare detectors before using them.
Scores go to work/filters/<task>/[benchmark/]<name>[.part-I-of-N].jsonl as {"image_id", "score"}, written after
every batch, so a run can be stopped and started again; `shard=(i, n)` takes every n-th image, so n processes (one
per GPU) share one model's work (parallel.py). <name>.meta.json records the device, the throughput and, when asked
(`latency`), the latency of one image: batch 1, after a warm-up, decoding excluded.
"""

import json
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

from PIL import Image, ImageOps

from ..common.dataset import iter_images
from ..common.device import check_heavy_run, pick_device
from ..common.io import append_jsonl, iter_jsonl, write_json
from ..paths import FILTERS_DIR

SETS = ("dataset", "benchmark")
WARMUP = 5


class Detector:
    task: str = ""          # "nsfw" or "watermark"
    name: str = ""          # short unique name; also the name of the score file
    model_id: str = ""      # where the weights come from (Hugging Face ID and revision, URL, package version)
    batch_size: int = 16
    install: tuple[str, ...] = ()   # extra pip arguments, one `pip install` each, beyond requirements-ml.txt;
                                    # detectors with the same `install` share one environment (envs.py)

    def load(self, device: str) -> None:
        """Load the model onto `device` ("cuda", "mps" or "cpu"). Import heavy libraries here, not at the top of
        the module, so that listing detectors needs none of them."""
        raise NotImplementedError

    def score(self, images: list[Image.Image]) -> list[float]:
        """Probability in [0, 1] that each image (RGB) is positive. Returns Python floats, so the device has
        finished when it returns (latency is measured around this call)."""
        raise NotImplementedError


def set_dir(task: str, on: str = "dataset"):
    if on not in SETS:
        raise ValueError(f"unknown image set {on!r}; expected one of {SETS}")
    return FILTERS_DIR / task / ("benchmark" if on == "benchmark" else "")


def score_path(task: str, name: str, on: str = "dataset", shard: tuple[int, int] | None = None):
    part = f".part-{shard[0]}-of-{shard[1]}" if shard else ""
    return set_dir(task, on) / f"{name}{part}.jsonl"


def score_files(task: str, name: str, on: str = "dataset") -> list:
    """The score file of a model and its shard files."""
    folder = set_dir(task, on)
    return sorted(folder.glob(f"{name}.jsonl")) + sorted(folder.glob(f"{name}.part-*.jsonl"))


def scored_models(task: str, on: str) -> list[str]:
    return sorted({p.name.split(".")[0] for p in set_dir(task, on).glob("*.jsonl")})


def image_refs(task: str, on: str, scope: str = "all", sets: list[str] | None = None) -> list:
    if on == "benchmark":
        from .benchmark import benchmark_refs
        return benchmark_refs(task, sets)
    return list(iter_images(scope))


def read_image(ref) -> Image.Image:
    with Image.open(ref.path) as im:
        return ImageOps.exif_transpose(im).convert("RGB")


def run(detector: Detector, scope: str = "all", limit: int | None = None, device: str | None = None,
        allow_cpu: bool = False, on: str = "dataset", shard: tuple[int, int] | None = None,
        latency: int = 0, sets: list[str] | None = None, workers: int = 4) -> dict:
    device = pick_device(device)
    out = score_path(detector.task, detector.name, on, shard)
    done = {r["image_id"] for f in score_files(detector.task, detector.name, on) for r in iter_jsonl(f)}
    refs = image_refs(detector.task, on, scope, sets)
    if shard:
        refs = refs[shard[0]::shard[1]]
    todo = [ref for ref in refs if ref.image_id not in done]
    if limit is not None:
        todo = todo[:limit]
    check_heavy_run(device, len(todo) + latency, allow_cpu)
    if not todo and not latency:
        return {"model": detector.name, "on": on, "scored": 0, "already_done": len(done)}
    started = time.monotonic()
    detector.load(device)
    load_seconds = time.monotonic() - started
    print(f"loaded {detector.name} on {device} in {load_seconds:.0f} s", flush=True)
    batches = [todo[i:i + detector.batch_size] for i in range(0, len(todo), detector.batch_size)]
    started = time.monotonic()
    with ThreadPoolExecutor(workers) as pool:            # decode the next batches while the model runs
        pending = [pool.submit(lambda b=b: [read_image(r) for r in b]) for b in batches[:2]]
        for i, batch in enumerate(batches):
            images = pending[i].result()
            if i + 2 < len(batches):
                pending.append(pool.submit(lambda b=batches[i + 2]: [read_image(r) for r in b]))
            pending[i] = None
            scores = detector.score(images)
            append_jsonl(out, ({"image_id": r.image_id, "score": round(float(s), 5)} for r, s in zip(batch, scores)))
            if (i + 1) % 50 == 0:
                print(f"{detector.name}: {(i + 1) * detector.batch_size}/{len(todo)}", flush=True)
    seconds = time.monotonic() - started
    lat = latency_probe(detector, refs[:latency]) if latency else None
    meta_path = out.with_suffix(".meta.json")
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {"images": 0, "seconds": 0.0}
    meta.update(model=detector.name, model_id=detector.model_id, task=detector.task, on=on, device=device,
                batch_size=detector.batch_size, **gpu_info(), images=meta["images"] + len(todo),
                seconds=round(meta["seconds"] + seconds, 1), load_seconds=round(load_seconds, 1),
                updated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"))
    meta["seconds_per_image"] = round(meta["seconds"] / meta["images"], 5) if meta["images"] else None
    if lat:
        meta["latency_ms"] = lat
    write_json(meta_path, meta)
    return {"model": detector.name, "on": on, "shard": "%d/%d" % shard if shard else None, "scored": len(todo),
            "already_done": len(done), "device": device,
            "seconds_per_image": round(seconds / len(todo), 5) if todo else None, "latency_ms": lat}


def latency_probe(detector: Detector, refs: list) -> dict | None:
    """Milliseconds to score one image (batch 1), decoding excluded, after WARMUP calls."""
    images = [read_image(r) for r in refs]
    if not images:
        return None
    for im in images[:WARMUP]:
        detector.score([im])
    times = []
    for im in images:
        t = time.perf_counter()
        detector.score([im])
        times.append((time.perf_counter() - t) * 1000)
    times.sort()
    return {"n": len(times), "mean": round(statistics.mean(times), 2), "p50": round(times[len(times) // 2], 2),
            "p95": round(times[min(len(times) - 1, int(len(times) * 0.95))], 2)}


def gpu_info() -> dict:
    try:
        import torch
        if torch.cuda.is_available():
            return {"gpu": torch.cuda.get_device_name(0)}
    except ImportError:
        pass
    return {}
