"""Out-of-fold predictions, with fold-local preprocessing, resumable seeds and temperature calibration."""

import hashlib
import json
import math
import time

import numpy as np

from ..common.device import check_heavy_run, pick_device
from ..common.io import write_json, write_jsonl
from ..paths import FINAL_PAIRS
from ..rank.review.common import dataset, digest, fingerprint, frozen_json, round_dir
from .train import DEFAULTS, edit_features, load_config, train_head


def fold_of(sample_id: str, count: int = 5) -> int:
    return int(hashlib.sha256(f"review-fold:{sample_id}".encode()).hexdigest(), 16) % count


def sigmoid(value):
    value = np.asarray(value, dtype=np.float64)
    return np.exp(-np.logaddexp(0, -value))


def calibrate(logits: np.ndarray, targets: np.ndarray) -> tuple[float, dict]:
    """Calibrate the ensemble probability, not the individual seed logits, against soft labels."""
    def loss(log_temperature):
        probabilities = np.clip(sigmoid(logits / math.exp(log_temperature)).mean(axis=1), 1e-9, 1 - 1e-9)
        return float(-(targets * np.log(probabilities) + (1 - targets) * np.log1p(-probabilities)).mean())

    grid = np.linspace(-6, 6, 61)
    best = min(range(len(grid)), key=lambda index: loss(grid[index]))
    low, high = float(grid[max(0, best - 1)]), float(grid[min(len(grid) - 1, best + 1)])
    ratio = (math.sqrt(5) - 1) / 2
    left, right = high - ratio * (high - low), low + ratio * (high - low)
    for _ in range(100):
        if loss(left) < loss(right):
            high, right = right, left
            left = high - ratio * (high - low)
        else:
            low, left = left, right
            right = low + ratio * (high - low)
    optimum = min([0.0, -6.0, 6.0, (low + high) / 2], key=loss)
    return math.exp(optimum), {"log_loss_before": loss(0), "log_loss_after": loss(optimum),
                                "boundary": abs(optimum) > 5.99}


def run(number: int, config: str = "configs/scorer.toml", device: str | None = None,
        allow_cpu: bool = False, max_hours: float | None = None) -> dict:
    import torch

    cfg = load_config(config)
    run_cfg = next((item for item in cfg["runs"] if item["name"] == "mlp-siglip"), None)
    if run_cfg is None:
        raise ValueError("OOF requires the selected mlp-siglip run")
    settings = {**DEFAULTS, **cfg.get("train", {}), **run_cfg}
    if settings["loss"] != "soft" or len(settings["seeds"]) != 5 or len(set(settings["seeds"])) != 5:
        raise ValueError("OOF requires soft-label loss and five unique seeds")
    rows, samples = dataset()
    device = pick_device(device)
    check_heavy_run(device, len(samples), allow_cpu)
    pairs = [{"id": row["a"].split("/")[0], "a": row["a"], "b": row["b"], "p": row["p"]} for row in rows]
    edits = sorted((sid, key) for sid, sample in samples.items() for key in sample["keys"])
    # Unlike generic feature extraction, fail on incomplete image OR text caches.
    from .embed import load_embeddings
    for backbone in settings["backbones"]:
        images = load_embeddings(backbone, "images")
        texts = load_embeddings(backbone, f"texts-{settings['text']}")
        if any(f"{sid}/{key}" not in images for sid, sample in samples.items()
               for key in ["original", *sample["keys"]]) or set(samples) - set(texts):
            raise ValueError(f"incomplete {backbone} embeddings; embed the full active dataset first")
    features, ids = edit_features(edits, settings["backbones"], settings["inputs"], settings["text"])
    row_of = {key: index for index, key in enumerate(ids)}
    folder = round_dir(number)
    seed_dir = folder / "oof-seeds"
    matrix_hash = hashlib.sha256(features.tobytes()).hexdigest()
    manifest = {"pairs_sha256": fingerprint(FINAL_PAIRS), "features_sha256": matrix_hash,
                "ids_sha256": digest(ids), "config": settings, "folds": 5, "schema": 1}
    frozen_json(folder / "oof-inputs.json", manifest)
    started, all_logits = time.monotonic(), np.empty((len(pairs), len(settings["seeds"])), dtype=np.float64)
    diagnostics = []
    for fold in range(5):
        held = [i for i, pair in enumerate(pairs) if fold_of(pair["id"]) == fold]
        remaining = sorted({sid for sid in samples if fold_of(sid) != fold},
                           key=lambda sid: hashlib.sha256(f"inner-dev:{fold}:{sid}".encode()).hexdigest())
        dev_ids = set(remaining[:max(1, math.ceil(len(remaining) * 0.1))])
        train_ids = set(remaining) - dev_ids
        training = [i for i, pair in enumerate(pairs) if pair["id"] in train_ids]
        development = [i for i, pair in enumerate(pairs) if pair["id"] in dev_ids]
        if not held or not training or not development:
            raise ValueError("each of the five folds needs held-out, training and inner-dev samples")
        train_rows = sorted({row_of[pairs[i][key]] for i in training for key in ("a", "b")})
        base = torch.from_numpy(features)
        mean = base[train_rows].mean(0)
        std = base[train_rows].std(0, unbiased=False).clamp_min(1e-6)
        matrix = ((base - mean) / std).to(device)

        def tensors(indices):
            return {"a": torch.tensor([row_of[pairs[i]["a"]] for i in indices], device=device),
                    "b": torch.tensor([row_of[pairs[i]["b"]] for i in indices], device=device),
                    "soft": torch.tensor([pairs[i]["p"] for i in indices], device=device),
                    "hard": torch.ones(len(indices), device=device)}

        for column, seed in enumerate(settings["seeds"]):
            path = seed_dir / f"fold-{fold}-seed-{seed}.json"
            if path.exists():
                saved = json.loads(path.read_text(encoding="utf-8"))
                if saved["indices"] != held or len(saved["logits"]) != len(held):
                    raise ValueError(f"inconsistent OOF cache: {path}")
            else:
                if max_hours is not None and time.monotonic() - started > max_hours * 3600:
                    return {"complete": False, "round": number, "resume": "rerun the same command"}
                head, history = train_head(settings, matrix, tensors(training), tensors(development), seed, device)
                with torch.no_grad():
                    scores = head["model"](matrix)[0].float().cpu().numpy()
                # Scale is fitted only to training edits, never to held-out samples.
                scale = max(float(scores[train_rows].std()), 1e-6)
                logits = [(float(scores[row_of[pairs[i]["a"]]]) - float(scores[row_of[pairs[i]["b"]]])) / scale
                          for i in held]
                saved = {"indices": held, "logits": logits, "fold": fold, "seed": seed,
                         "best_epoch": head["best_epoch"], "train_samples": len(train_ids),
                         "dev_samples": len(dev_ids), "history": history}
                write_json(path, saved)
                from ..common.checkpoint import checkpoint
                checkpoint(f'oof-fold-{fold}-seed-{seed}')
                print(f"OOF fold {fold + 1}/5, seed {seed}: epoch {head['best_epoch']}", flush=True)
            all_logits[held, column] = saved["logits"]
            diagnostics.append({key: saved[key] for key in ("fold", "seed", "best_epoch", "train_samples", "dev_samples")})
    if not np.isfinite(all_logits).all():
        raise ValueError("non-finite OOF logits")
    temperature, calibration = calibrate(all_logits, np.array([pair["p"] for pair in pairs]))
    probabilities = sigmoid(all_logits / temperature)
    from .evaluate import baseline_scores
    baselines = baseline_scores()
    predictions = []
    for index, pair in enumerate(pairs):
        votes = {name: float(scores[pair["a"]] > scores[pair["b"]]) if scores[pair["a"]] != scores[pair["b"]] else 0.5
                 for name, scores in baselines.items() if pair["a"] in scores and pair["b"] in scores}
        predictions.append({**pair, "fold": fold_of(pair["id"]), "P_model": float(probabilities[index].mean()),
                            "spread": float(probabilities[index].std()), "baselines": votes})
    write_jsonl(folder / "oof_pairs.jsonl", predictions)
    summary = {"complete": True, "round": number, "pairs": len(predictions), "samples": len(samples),
               "temperature": temperature, "calibration": calibration, "training": diagnostics,
               "baseline_pair_coverage": {name: sum(name in row["baselines"] for row in predictions) for name in baselines},
               "inputs_sha256": digest(manifest)}
    write_json(folder / "oof-summary.json", summary)
    return summary
