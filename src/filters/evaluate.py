"""Compare detectors and voting methods with labels, on quality and speed.

Labels: on the dataset, hand labels in work/labels/<task>.jsonl ({"image_id", "label"}, 1 = positive); on a public
benchmark, its labels (benchmark.py). For every model and every voting method this reports, at the configured
threshold, accuracy, precision, recall, F1, the false positive rate and the confusion counts; ROC-AUC and average
precision over all thresholds; the best F1 over thresholds (optimistic: chosen on the same labels); the share of
images flagged per class (benchmark image_ids start with their class); and the time per image (models only).
The answer to "vote or single best model?" is the F1 gap against the time cost.
"""

import json
from collections import defaultdict

from ..common.io import iter_jsonl, write_json
from ..paths import LABELS_DIR
from .base import set_dir
from .voting import METHODS, combine, load_scores


def prf(scores: dict[str, float], labels: dict[str, int], threshold: float) -> dict:
    ids = [i for i in labels if i in scores]
    tp = sum(1 for i in ids if scores[i] >= threshold and labels[i])
    fp = sum(1 for i in ids if scores[i] >= threshold and not labels[i])
    fn = sum(1 for i in ids if scores[i] < threshold and labels[i])
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {"n": len(ids), "precision": round(p, 4), "recall": round(r, 4),
            "f1": round(2 * p * r / (p + r), 4) if p + r else 0.0}


def confusion(scores: dict[str, float], labels: dict[str, int], threshold: float) -> dict:
    ids = [i for i in labels if i in scores]
    tp = sum(1 for i in ids if scores[i] >= threshold and labels[i])
    fp = sum(1 for i in ids if scores[i] >= threshold and not labels[i])
    fn = sum(1 for i in ids if scores[i] < threshold and labels[i])
    tn = len(ids) - tp - fp - fn
    return {"accuracy": round((tp + tn) / len(ids), 4) if ids else 0.0,
            "fpr": round(fp / (fp + tn), 4) if fp + tn else 0.0, "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def roc_auc(scores: dict[str, float], labels: dict[str, int]) -> float | None:
    """Probability that a random positive scores higher than a random negative (ties count 1/2)."""
    pos = sorted(scores[i] for i in labels if i in scores and labels[i])
    neg = sorted(scores[i] for i in labels if i in scores and not labels[i])
    if not pos or not neg:
        return None
    wins, j, k = 0.0, 0, 0
    for s in pos:                                  # neg[:j] < s, neg[j:k] == s
        while j < len(neg) and neg[j] < s:
            j += 1
        k = max(k, j)
        while k < len(neg) and neg[k] == s:
            k += 1
        wins += j + (k - j) / 2
    return round(wins / (len(pos) * len(neg)), 4)


def average_precision(scores: dict[str, float], labels: dict[str, int]) -> float | None:
    ranked = sorted((i for i in labels if i in scores), key=lambda i: -scores[i])
    n_pos = sum(labels[i] for i in ranked)
    if not n_pos:
        return None
    hits, total = 0, 0.0
    for k, i in enumerate(ranked, 1):
        if labels[i]:
            hits += 1
            total += hits / k
    return round(total / n_pos, 4)


def best_f1(scores: dict[str, float], labels: dict[str, int]) -> dict:
    candidates = sorted({round(scores[i], 4) for i in labels if i in scores})
    best = {"f1": 0.0, "threshold": None}
    for t in candidates:
        m = prf(scores, labels, t)
        if m["f1"] > best["f1"]:
            best = {"f1": m["f1"], "threshold": t}
    return best


def per_class(scores: dict[str, float], labels: dict[str, int], threshold: float) -> dict | None:
    """Share of images flagged per class, when image_ids are "<class>/<name>" (benchmark sets)."""
    groups = defaultdict(list)
    for i in labels:
        if i in scores and "/" in i:
            groups[i.split("/")[0]].append(scores[i] >= threshold)
    return {c: round(sum(v) / len(v), 4) for c, v in sorted(groups.items())} or None


def report_for(scores: dict[str, float], labels: dict[str, int], threshold: float, classes: bool) -> dict:
    out = {**prf(scores, labels, threshold), **confusion(scores, labels, threshold), "threshold": threshold,
           "roc_auc": roc_auc(scores, labels), "average_precision": average_precision(scores, labels),
           "best": best_f1(scores, labels)}
    if classes:
        out["flagged_per_class"] = per_class(scores, labels, threshold)
    return out


def evaluate(task: str, cfg: dict, on: str = "dataset") -> dict:
    """On the dataset: the models of the config. On a benchmark: every model with scores there."""
    folder = set_dir(task, on)
    if on == "benchmark":
        from .benchmark import read_labels
        labels = read_labels(task)
        models = sorted(p.stem for p in folder.glob("*.jsonl"))
    else:
        labels = {r["image_id"]: int(r["label"]) for r in iter_jsonl(LABELS_DIR / f"{task}.jsonl")}
        models = cfg.get("models", [])
    if not labels:
        raise SystemExit(f"no labels for {task} on the {on}")
    thresholds, weights = cfg.get("thresholds", {}), cfg.get("weights", {})
    scores = load_scores(task, models, on)
    report = {"task": task, "on": on, "labels": len(labels), "positives": sum(labels.values()),
              "models": {}, "voting": {}}
    for m in models:
        meta = folder / f"{m}.meta.json"
        speed = json.loads(meta.read_text()).get("seconds_per_image") if meta.exists() else None
        report["models"][m] = {**report_for(scores[m], labels, thresholds.get(m, 0.5), on == "benchmark"),
                               "seconds_per_image": speed}
    if len(models) > 1:
        for method in METHODS:
            combined = combine(scores, method, thresholds, weights)
            report["voting"][method] = report_for(combined, labels, cfg.get("threshold", 0.5), on == "benchmark")
        report["voting_seconds_per_image"] = round(sum(r["seconds_per_image"] or 0 for r in report["models"].values()), 4)
    write_json(folder / "evaluation.json", report)
    return report
