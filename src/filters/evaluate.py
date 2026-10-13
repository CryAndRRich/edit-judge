"""Compare detectors and voting methods with hand labels, on quality and speed.

Hand labels: data/labels/<task>.jsonl with {"image_id", "label"} (1 = positive). For every model and every voting
method this reports precision, recall and F1 at the configured threshold, the best F1 over all thresholds, and the
time per image (models only). The answer to "vote or single best model?" is the F1 gap against the time cost.
"""

import json

from ..common.io import iter_jsonl, write_json
from ..paths import FILTERS_DIR, LABELS_DIR
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


def best_f1(scores: dict[str, float], labels: dict[str, int]) -> dict:
    candidates = sorted({round(scores[i], 4) for i in labels if i in scores})
    best = {"f1": 0.0, "threshold": None}
    for t in candidates:
        m = prf(scores, labels, t)
        if m["f1"] > best["f1"]:
            best = {"f1": m["f1"], "threshold": t}
    return best


def evaluate(task: str, cfg: dict) -> dict:
    labels = {r["image_id"]: int(r["label"]) for r in iter_jsonl(LABELS_DIR / f"{task}.jsonl")}
    if not labels:
        raise SystemExit(f"no hand labels in {LABELS_DIR / f'{task}.jsonl'}")
    models = cfg.get("models", [])
    thresholds, weights = cfg.get("thresholds", {}), cfg.get("weights", {})
    scores = load_scores(task, models)
    report = {"task": task, "labels": len(labels), "positives": sum(labels.values()), "models": {}, "voting": {}}
    for m in models:
        meta = FILTERS_DIR / task / f"{m}.meta.json"
        speed = json.loads(meta.read_text()).get("seconds_per_image") if meta.exists() else None
        report["models"][m] = {**prf(scores[m], labels, thresholds.get(m, 0.5)),
                               "best": best_f1(scores[m], labels), "seconds_per_image": speed}
    if len(models) > 1:
        for method in METHODS:
            combined = combine(scores, method, thresholds, weights)
            report["voting"][method] = {**prf(combined, labels, cfg.get("threshold", 0.5)),
                                        "best": best_f1(combined, labels)}
    write_json(FILTERS_DIR / task / "evaluation.json", report)
    return report
