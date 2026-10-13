"""Combine the scores of several detectors of one task into one score per image.

Methods (the combined score is in [0, 1]; an image is flagged when it reaches `threshold`):
- "mean": mean of the scores;
- "weighted": weighted mean, with `weights` per model (missing weights count as 1);
- "max": the highest score (flag if any model is confident);
- "majority": share of models whose score reaches their own threshold (`thresholds`, default 0.5);
  with threshold 0.5 this is a majority vote.
Only images scored by every model are combined.
"""

from ..common.io import iter_jsonl
from ..paths import FILTERS_DIR

METHODS = ("mean", "weighted", "max", "majority")


def load_scores(task: str, models: list[str]) -> dict[str, dict[str, float]]:
    """model -> image_id -> score."""
    return {m: {r["image_id"]: r["score"] for r in iter_jsonl(FILTERS_DIR / task / f"{m}.jsonl")} for m in models}


def combine(scores: dict[str, dict[str, float]], method: str, thresholds: dict | None = None,
            weights: dict | None = None) -> dict[str, float]:
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; expected one of {METHODS}")
    if not scores:
        return {}
    thresholds, weights = thresholds or {}, weights or {}
    models = list(scores)
    common = set.intersection(*(set(s) for s in scores.values()))
    out = {}
    for image_id in common:
        values = [scores[m][image_id] for m in models]
        if method == "mean":
            out[image_id] = sum(values) / len(values)
        elif method == "weighted":
            w = [weights.get(m, 1.0) for m in models]
            out[image_id] = sum(v * x for v, x in zip(values, w)) / sum(w)
        elif method == "max":
            out[image_id] = max(values)
        else:
            out[image_id] = sum(v >= thresholds.get(m, 0.5) for m, v in zip(models, values)) / len(models)
    return out
