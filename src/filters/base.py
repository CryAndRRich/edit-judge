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
                                    # same `install`, same environment (src/common/envs.py)
    isolated: bool = False          # True: an environment that does not see the machine's packages at all, for
                                    # libraries that bring their own torch (the machine's torchaudio, torchvision
                                    # would not match it)

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
    return sorted({path.name.split(".")[0] for path in set_dir(task, on).glob("*.jsonl")})


def image_refs(task: str, on: str, scope: str = "all", sets: list[str] | None = None) -> list:
    if on == "benchmark":
        from .benchmark import benchmark_refs
        return benchmark_refs(task, sets)
    return list(iter_images(scope))


def read_image(ref) -> Image.Image:
    with Image.open(ref.path) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


def run(detector: Detector, scope: str = "all", limit: int | None = None, device: str | None = None,
        allow_cpu: bool = False, on: str = "dataset", shard: tuple[int, int] | None = None,
        latency: int = 0, sets: list[str] | None = None, workers: int = 4) -> dict:
    device = pick_device(device)
    out = score_path(detector.task, detector.name, on, shard)
    done = {row["image_id"] for path in score_files(detector.task, detector.name, on) for row in iter_jsonl(path)}
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
    batches = [todo[start:start + detector.batch_size] for start in range(0, len(todo), detector.batch_size)]
    started = time.monotonic()

    def decode(batch_refs):
        return [read_image(ref) for ref in batch_refs]

    with ThreadPoolExecutor(workers) as pool:            # decode the next batches while the model runs
        pending = [pool.submit(decode, batch_refs) for batch_refs in batches[:2]]
        for index, batch_refs in enumerate(batches):
            images = pending[index].result()
            if index + 2 < len(batches):
                pending.append(pool.submit(decode, batches[index + 2]))
            pending[index] = None
            scores = detector.score(images)
            append_jsonl(out, ({"image_id": ref.image_id, "score": round(float(score), 5)}
                               for ref, score in zip(batch_refs, scores)))
            if (index + 1) % max(1, 200 // detector.batch_size) == 0 or index + 1 == len(batches):
                n_done = min((index + 1) * detector.batch_size, len(todo))
                rate = n_done / max(time.monotonic() - started, 1e-9)
                print(f"{detector.name}: {n_done}/{len(todo)} images ({100 * n_done // len(todo)}%), "
                      f"{rate:.1f} img/s, about {(len(todo) - n_done) / rate / 60:.0f} min left", flush=True)
    seconds = time.monotonic() - started
    latency_ms = latency_probe(detector, refs[:latency]) if latency else None
    meta_path = out.with_suffix(".meta.json")
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {"images": 0, "seconds": 0.0}
    meta.update(model=detector.name, model_id=detector.model_id, task=detector.task, on=on, device=device,
                batch_size=detector.batch_size, **gpu_info(), images=meta["images"] + len(todo),
                seconds=round(meta["seconds"] + seconds, 1), load_seconds=round(load_seconds, 1),
                updated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"))
    meta["seconds_per_image"] = round(meta["seconds"] / meta["images"], 5) if meta["images"] else None
    if latency_ms:
        meta["latency_ms"] = latency_ms
    write_json(meta_path, meta)
    return {"model": detector.name, "on": on, "shard": "%d/%d" % shard if shard else None, "scored": len(todo),
            "already_done": len(done), "device": device,
            "seconds_per_image": round(seconds / len(todo), 5) if todo else None, "latency_ms": latency_ms}


def latency_probe(detector: Detector, refs: list) -> dict | None:
    """Milliseconds to score one image (batch 1), decoding excluded, after WARMUP calls."""
    images = [read_image(ref) for ref in refs]
    if not images:
        return None
    for image in images[:WARMUP]:
        detector.score([image])
    times = []
    for image in images:
        start = time.perf_counter()
        detector.score([image])
        times.append((time.perf_counter() - start) * 1000)
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
