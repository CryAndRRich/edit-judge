"""Latency: how long one request takes, i.e. comparing the two edits of one pair, end to end and one request at a
time (no batching over requests). Accuracy tables say how good a scorer is; this says what it costs to use.

The same pairs for every model: one pair per test sample, the first samples in the reward runner's fixed order
(data.sample_order). Each request is timed from the image files' bytes (already in memory, so the disk is not
measured) to the two scores: decoding, preprocessing, the encoders or model, and the head. The first WARMUP_PAIRS
requests are not counted (CUDA kernels and caches warm up). Results: median, 90th percentile and mean in ms, and
the peak GPU memory (weights included) while answering.

- Trained runs (`scorer_latency`, `python -m src.scorer latency`): every run with a runs/<name>/head.pt. The three
  images (original, edit a, edit b) go through each backbone of the run as one batch and the text once, then the
  features of train.py and every seed head (the ensemble). The scores are compared with the run's scores.jsonl
  (`max_difference`): it shows that the timed path computes what was evaluated (up to float16 rounding).
  Similarity baselines (CLIP-score, ...) are timed the same way with their one backbone. Runs sharing the same
  backbones are timed together, with only those backbones on the GPU. Writes work/scorer/latency.json.
- Reward models and VLM judges: the runner times each model right after loading it (rewards/base.py), the two
  edits of a request in one `score` call. Writes work/scorer/baselines/<model>.latency.json.
"""

import gc
import io
import json
import time
from datetime import datetime, timezone

import numpy as np

from ..common.dataset import iter_samples, text_of
from ..common.device import check_heavy_run, pick_device
from ..common.io import iter_jsonl, write_json
from ..paths import SAMPLES_DIR, SCORER_DIR
from .backbones import get_backbone
from .data import load_pairs, sample_order

LATENCY_PAIRS = 50
WARMUP_PAIRS = 3
LATENCY_FILE = SCORER_DIR / "latency.json"


def latency_pairs(count: int, text_kind: str = "instruction", split: str = "test") -> list[dict]:
    """`count` requests {"id", "keys": (a, b), "text", "bytes": {key: file bytes}}: the first pair of each of the
    first `count` samples of the split in the fixed order. The bytes are read here, outside the timing."""
    first = {}
    for pair in load_pairs("all", split):
        first.setdefault(pair["id"], pair)
    samples = {sample["id"]: sample for sample in iter_samples() if sample["id"] in first}
    requests = []
    for sample_id in sorted(samples, key=sample_order)[:count]:
        keys = (first[sample_id]["a"], first[sample_id]["b"])
        requests.append({"id": sample_id, "keys": keys, "text": text_of(samples[sample_id], text_kind),
                         "bytes": {key: (SAMPLES_DIR / sample_id / f"{key}.jpg").read_bytes()
                                   for key in ("original", *keys)}})
    return requests


def timing_summary(seconds: list[float]) -> dict:
    values = np.asarray(seconds) * 1000
    return {"pairs": len(seconds), "median_ms": round(float(np.median(values)), 1),
            "p90_ms": round(float(np.percentile(values, 90)), 1), "mean_ms": round(float(values.mean()), 1)}


def reset_gpu_peak() -> None:
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            torch.cuda.reset_peak_memory_stats(index)


def gpu_peak_gb() -> float | None:
    """Peak memory allocated by torch since `reset_gpu_peak`, over all GPUs."""
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    peak = sum(torch.cuda.max_memory_allocated(index) for index in range(torch.cuda.device_count()))
    return round(peak / 1e9, 2)


def gpu_name(device: str) -> str | None:
    if device != "cuda":
        return None
    import torch
    return torch.cuda.get_device_name(0)


def time_requests(requests: list[dict], answer) -> list[float]:
    """Seconds per request of `answer(request)`, without the warm-up requests."""
    seconds = []
    for index, request in enumerate(requests):
        started = time.perf_counter()
        answer(request)                        # returns CPU values, so the GPU work is finished here
        if index >= WARMUP_PAIRS:
            seconds.append(time.perf_counter() - started)
    return seconds


def encode_request(backbones: dict, request: dict, device: str) -> dict:
    """Backbone name -> (image vectors of original, a, b; text vector or None), from the file bytes."""
    import torch

    from .embed import square_image

    keys = ("original", *request["keys"])
    batch = torch.stack([torch.from_numpy(square_image(io.BytesIO(request["bytes"][key]))).permute(2, 0, 1)
                         for key in keys]).to(device)
    return {name: (backbone.encode_images(batch),
                   backbone.encode_text_batch([request["text"]]) if backbone.has_text else None)
            for name, backbone in backbones.items()}


class TrainedRun:
    """A run's seed heads and feature standardisation from head.pt, scoring a request's two edits."""

    def __init__(self, name: str, device: str):
        import torch

        from .heads import ScoreHead

        saved = torch.load(SCORER_DIR / "runs" / name / "head.pt", map_location="cpu", weights_only=False)
        self.name, self.cfg, self.device = name, saved["config"], device
        self.feature_mean = saved["feature_mean"].to(device)
        self.feature_std = saved["feature_std"].to(device)
        self.heads = []
        for head in saved["heads"]:
            model = ScoreHead(len(saved["feature_mean"]), self.cfg["head"], self.cfg["hidden"], self.cfg["dropout"],
                              self.cfg["input_dropout"], self.cfg["loss"] == "uncertainty")
            model.load_state_dict(head["state_dict"])
            self.heads.append((model.to(device).eval(), head["score_mean"], head["score_std"]))

    def score(self, encoded: dict) -> list[float]:
        import torch

        from .train import backbone_blocks

        blocks = []
        for name in self.cfg["backbones"]:
            images, text = encoded[name]
            blocks += backbone_blocks(images[1:], images[[0, 0]], None if text is None else text[[0, 0]],
                                      self.cfg["inputs"])
        features = torch.from_numpy(np.concatenate(blocks, axis=1).astype(np.float32)).to(self.device)
        features = (features - self.feature_mean) / self.feature_std
        with torch.no_grad():
            scores = torch.stack([(model(features)[0] - mean) / std for model, mean, std in self.heads]).mean(0)
        return scores.cpu().tolist()


def similarity_score(encoded: dict, backbone: str) -> list[float]:
    images, text = encoded[backbone]
    return (images[1:] @ text[0]).tolist()           # vectors are L2-normalised: the cosine


def scorer_latency(run_names: list[str] | None = None, count: int = LATENCY_PAIRS, device: str | None = None,
                   allow_cpu: bool = False, similarity: dict[str, str] | None = None,
                   text_kind: str = "instruction") -> dict:
    """Time every trained run (or `run_names`) and the similarity baselines (name -> backbone, default those of
    evaluate.py) on `count` requests; merges the results into work/scorer/latency.json."""
    import torch

    from .evaluate import SIMILARITY_BASELINES

    device = pick_device(device)
    check_heavy_run(device, count, allow_cpu)
    names = run_names or sorted(path.parent.name for path in (SCORER_DIR / "runs").glob("*/head.pt"))
    similarity = SIMILARITY_BASELINES if similarity is None else similarity
    jobs = {}                                  # backbones -> [(kind, name, run or backbone)]
    for name in names:
        run = TrainedRun(name, device)
        jobs.setdefault(tuple(run.cfg["backbones"]), []).append(("runs", name, run))
    for name, backbone in similarity.items():
        jobs.setdefault((backbone,), []).append(("baselines", name, backbone))
    requests = {}
    results = {"runs": {}, "baselines": {}}
    for backbone_names, group in jobs.items():
        backbones = {name: get_backbone(name) for name in backbone_names}
        for backbone in backbones.values():
            backbone.load(device)
        print(f"latency: {', '.join(backbone_names)} loaded", flush=True)
        reset_gpu_peak()
        for kind, name, target in group:
            text = target.cfg["text"] if kind == "runs" else text_kind
            if text not in requests:
                requests[text] = latency_pairs(count + WARMUP_PAIRS, text)
            given = {}

            def answer(request):
                encoded = encode_request(backbones, request, device)
                scores = target.score(encoded) if kind == "runs" else similarity_score(encoded, target)
                for key, score in zip(request["keys"], scores):
                    given[f"{request['id']}/{key}"] = score

            result = {"backbones": list(backbone_names), "text": text,
                      **timing_summary(time_requests(requests[text], answer)), "device": device,
                      "gpu": gpu_name(device)}
            if kind == "runs":
                stored = {row["image_id"]: row["score"]
                          for row in iter_jsonl(SCORER_DIR / "runs" / name / "scores.jsonl")
                          if row["image_id"] in given}
                result["max_difference"] = round(max((abs(given[key] - value) for key, value in stored.items()),
                                                     default=0.0), 4)
            results[kind][name] = result
            print(f"latency: {name}: median {result['median_ms']} ms, p90 {result['p90_ms']} ms", flush=True)
        peak = gpu_peak_gb()
        for kind, name, _ in group:
            results[kind][name]["peak_gpu_memory_gb"] = peak
        del backbones
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
    previous = load_latency(LATENCY_FILE)
    for kind in results:
        previous.setdefault(kind, {}).update(results[kind])
    previous.update(warmup_pairs=WARMUP_PAIRS, updated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"))
    write_json(LATENCY_FILE, previous)
    return results


def load_latency(path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
