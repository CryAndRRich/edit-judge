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
from .base import score_files

METHODS = ("mean", "weighted", "max", "majority")


def load_scores(task: str, models: list[str], on: str = "dataset") -> dict[str, dict[str, float]]:
    """model -> image_id -> score."""
    return {model: {row["image_id"]: row["score"] for path in score_files(task, model, on) for row in iter_jsonl(path)}
            for model in models}


def combine(scores: dict[str, dict[str, float]], method: str, thresholds: dict | None = None,
            weights: dict | None = None) -> dict[str, float]:
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; expected one of {METHODS}")
    if not scores:
        return {}
    thresholds, weights = thresholds or {}, weights or {}
    models = list(scores)
    common = set.intersection(*(set(model_scores) for model_scores in scores.values()))
    combined = {}
    for image_id in common:
        values = [scores[model][image_id] for model in models]
        if method == "mean":
            combined[image_id] = sum(values) / len(values)
        elif method == "weighted":
            model_weights = [weights.get(model, 1.0) for model in models]
            weighted_sum = sum(value * weight for value, weight in zip(values, model_weights))
            combined[image_id] = weighted_sum / sum(model_weights)
        elif method == "max":
            combined[image_id] = max(values)
        else:
            votes = sum(value >= thresholds.get(model, 0.5) for model, value in zip(models, values))
            combined[image_id] = votes / len(models)
    return combined
