"""Compare detectors and voting methods with labels, on quality and speed.

Labels: those of the public benchmark sets (benchmark.py); the dataset has no labels (hand labels were dropped
on 2026-11-06), so its decisions come from the benchmark's choice (configs/filters.toml). For every model and every voting method this reports, at the configured
threshold, accuracy, precision, recall, F1, the false positive rate and the confusion counts; ROC-AUC and average
precision over all thresholds; the best F1 over thresholds (optimistic: chosen on the same labels); results per
benchmark set and the share of images flagged per class; and the time per image (models only).

Votes are tried over every subset of two or more models and every method (a weak model can spoil a vote of all
of them). The answer to "vote or single best model?" is `choice`: the best single model and the best vote at the
configured thresholds, the F1 gain of the vote and its time cost (the sum of its models' times).
"""

import itertools
import json
from collections import defaultdict

from ..common.io import write_json
from .base import scored_models, set_dir
from .voting import METHODS, combine, load_scores


def counts(scores: dict[str, float], labels: dict[str, int], threshold: float) -> tuple[int, int, int, int]:
    """True positives, false positives, false negatives and true negatives of the labelled, scored images."""
    image_ids = [image_id for image_id in labels if image_id in scores]
    flagged = {image_id for image_id in image_ids if scores[image_id] >= threshold}
    tp = sum(1 for image_id in flagged if labels[image_id])
    fp = len(flagged) - tp
    fn = sum(1 for image_id in image_ids if labels[image_id] and image_id not in flagged)
    return tp, fp, fn, len(image_ids) - tp - fp - fn


def precision_recall_f1(scores: dict[str, float], labels: dict[str, int], threshold: float) -> dict:
    tp, fp, fn, tn = counts(scores, labels, threshold)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"n": tp + fp + fn + tn, "precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4)}


def confusion(scores: dict[str, float], labels: dict[str, int], threshold: float) -> dict:
    tp, fp, fn, tn = counts(scores, labels, threshold)
    total = tp + fp + fn + tn
    return {"accuracy": round((tp + tn) / total, 4) if total else 0.0,
            "fpr": round(fp / (fp + tn), 4) if fp + tn else 0.0, "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def roc_auc(scores: dict[str, float], labels: dict[str, int]) -> float | None:
    """Probability that a random positive scores higher than a random negative (ties count 1/2)."""
    positives = sorted(scores[i] for i in labels if i in scores and labels[i])
    negatives = sorted(scores[i] for i in labels if i in scores and not labels[i])
    if not positives or not negatives:
        return None
    # negatives[:below] < score and negatives[below:below_or_equal] == score
    wins, below, below_or_equal = 0.0, 0, 0
    for score in positives:
        while below < len(negatives) and negatives[below] < score:
            below += 1
        below_or_equal = max(below_or_equal, below)
        while below_or_equal < len(negatives) and negatives[below_or_equal] == score:
            below_or_equal += 1
        wins += below + (below_or_equal - below) / 2
    return round(wins / (len(positives) * len(negatives)), 4)


def average_precision(scores: dict[str, float], labels: dict[str, int]) -> float | None:
    ranked = sorted((i for i in labels if i in scores), key=lambda image_id: -scores[image_id])
    n_positive = sum(labels[i] for i in ranked)
    if not n_positive:
        return None
    hits, total = 0, 0.0
    for rank, image_id in enumerate(ranked, 1):
        if labels[image_id]:
            hits += 1
            total += hits / rank
    return round(total / n_positive, 4)


def best_f1(scores: dict[str, float], labels: dict[str, int]) -> dict:
    best = {"f1": 0.0, "threshold": None}
    for threshold in sorted({round(scores[i], 4) for i in labels if i in scores}):
        f1 = precision_recall_f1(scores, labels, threshold)["f1"]
        if f1 > best["f1"]:
            best = {"f1": f1, "threshold": threshold}
    return best


def per_class(scores: dict[str, float], labels: dict[str, int], threshold: float) -> dict | None:
    """Share of images flagged per "<set>/<class>" (benchmark image_ids are "<set>/<class>/<name>")."""
    groups = defaultdict(list)
    for image_id in labels:
        if image_id in scores and image_id.count("/") >= 2:
            groups["/".join(image_id.split("/")[:2])].append(scores[image_id] >= threshold)
    return {group: round(sum(flags) / len(flags), 4) for group, flags in sorted(groups.items())} or None


def by_set(scores: dict[str, float], labels: dict[str, int], threshold: float) -> dict:
    groups = defaultdict(dict)
    for image_id, label in labels.items():
        groups[image_id.split("/")[0]][image_id] = label
    return {name: {**precision_recall_f1(scores, group, threshold),
                   "accuracy": confusion(scores, group, threshold)["accuracy"]}
            for name, group in sorted(groups.items())}


def report_for(scores: dict[str, float], labels: dict[str, int], threshold: float, benchmark: bool) -> dict:
    out = {**precision_recall_f1(scores, labels, threshold), **confusion(scores, labels, threshold),
           "threshold": threshold, "roc_auc": roc_auc(scores, labels),
           "average_precision": average_precision(scores, labels), "best": best_f1(scores, labels)}
    if benchmark:
        out["by_set"] = by_set(scores, labels, threshold)
        out["flagged_per_class"] = per_class(scores, labels, threshold)
    return out


def subset_votes(scores: dict[str, dict[str, float]], labels: dict[str, int], cfg: dict,
                 seconds: dict[str, float | None], top: int = 15) -> list[dict]:
    """The best `top` votes over every subset of 2+ models and every method, by F1 then time."""
    thresholds, weights, threshold = cfg.get("thresholds", {}), cfg.get("weights", {}), cfg.get("threshold", 0.5)
    rows = []
    for size in range(2, len(scores) + 1):
        for subset in itertools.combinations(sorted(scores), size):
            for method in METHODS:
                if method == "weighted" and not weights:     # the same as "mean"
                    continue
                combined = combine({model: scores[model] for model in subset}, method, thresholds, weights)
                rows.append({"models": list(subset), "method": method,
                             **precision_recall_f1(combined, labels, threshold),
                             "accuracy": confusion(combined, labels, threshold)["accuracy"],
                             "roc_auc": roc_auc(combined, labels),
                             "seconds_per_image": round(sum(seconds.get(model) or 0 for model in subset), 4)})
    rows.sort(key=lambda row: (-row["f1"], row["seconds_per_image"]))
    return rows[:top]


def evaluate(task: str, cfg: dict, on: str = "dataset", models: list[str] | None = None) -> dict:
    """Every model with scores on the benchmark (or `models`). Only the benchmark has labels."""
    if on != "benchmark":
        raise SystemExit("only the benchmark has labels: use --on benchmark")
    from .benchmark import read_labels
    folder = set_dir(task, on)
    labels = read_labels(task)
    models = models or scored_models(task, on)
    if not labels:
        raise SystemExit(f"no labels for {task} on the {on}")
    thresholds, weights = cfg.get("thresholds", {}), cfg.get("weights", {})
    scores = load_scores(task, models, on)
    report = {"task": task, "on": on, "labels": len(labels), "positives": sum(labels.values()),
              "models": {}, "voting": {}}
    for model in models:
        meta_path = folder / f"{model}.meta.json"
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        report["models"][model] = {
            **report_for(scores[model], labels, thresholds.get(model, 0.5), on == "benchmark"),
            "seconds_per_image": meta.get("seconds_per_image"), "latency_ms": meta.get("latency_ms"),
            "batch_size": meta.get("batch_size"), "gpu": meta.get("gpu")}
    if len(models) > 1:
        for method in METHODS:
            combined = combine(scores, method, thresholds, weights)
            report["voting"][method] = report_for(combined, labels, cfg.get("threshold", 0.5), on == "benchmark")
        seconds = {model: result["seconds_per_image"] for model, result in report["models"].items()}
        report["voting_seconds_per_image"] = round(sum(value or 0 for value in seconds.values()), 4)
        report["subset_votes"] = subset_votes(scores, labels, cfg, seconds)
        single = max(report["models"], key=lambda model: (report["models"][model]["f1"], -(seconds[model] or 0)))
        vote = report["subset_votes"][0]
        report["choice"] = {
            "single": {"model": single, "f1": report["models"][single]["f1"], "seconds_per_image": seconds[single]},
            "vote": vote, "f1_gain": round(vote["f1"] - report["models"][single]["f1"], 4),
            "time_ratio": round(vote["seconds_per_image"] / seconds[single], 2) if seconds[single] else None}
    write_json(folder / "evaluation.json", report)
    return report


def three_decimals(value: float | None) -> str:
    return "-" if value is None else f"{value:.3f}"


def one_decimal(value: float | None) -> str:
    return "-" if value is None else f"{value:.1f}"


def write_markdown(report: dict) -> str:
    """The report's tables (models, models by set, best votes), also written as evaluation.md."""
    models = sorted(report["models"].items(), key=lambda item: -item[1]["f1"])
    lines = [f"# {report['task']} detectors on the {report['on']} ({report['labels']} images, "
             f"{report['positives']} positive)", "",
             "Threshold 0.5 unless set in configs/filters.toml. Latency: one image (batch 1) after a warm-up, "
             "decoding excluded; throughput: batched, in ms per image.", "",
             "| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Latency p50 (ms) | Latency p95 (ms) | "
             "Throughput (ms/img) |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for model, result in models:
        latency = result.get("latency_ms") or {}
        seconds = result.get("seconds_per_image")
        lines.append(f"| {model} | {three_decimals(result['accuracy'])} | {three_decimals(result['precision'])} | "
                     f"{three_decimals(result['recall'])} | {three_decimals(result['f1'])} | "
                     f"{three_decimals(result['roc_auc'])} | {one_decimal(latency.get('p50'))} | "
                     f"{one_decimal(latency.get('p95'))} | {one_decimal(seconds * 1000 if seconds else None)} |")
    set_names = sorted({name for _, result in models for name in (result.get("by_set") or {})})
    if len(set_names) > 1:
        lines += ["", "F1 (accuracy) by set:", "", "| Model | " + " | ".join(set_names) + " |",
                  "|---|" + "---:|" * len(set_names)]
        for model, result in models:
            cells = [f"{three_decimals(result['by_set'][name]['f1'])} "
                     f"({three_decimals(result['by_set'][name]['accuracy'])})" if name in result["by_set"] else "-"
                     for name in set_names]
            lines.append(f"| {model} | " + " | ".join(cells) + " |")
    if report.get("subset_votes"):
        lines += ["", "Best votes over subsets of models:", "",
                  "| Models | Method | Accuracy | Precision | Recall | F1 | Throughput (ms/img, sum) |",
                  "|---|---|---:|---:|---:|---:|---:|"]
        for vote in report["subset_votes"][:10]:
            lines.append(f"| {' + '.join(vote['models'])} | {vote['method']} | {three_decimals(vote['accuracy'])} | "
                         f"{three_decimals(vote['precision'])} | {three_decimals(vote['recall'])} | "
                         f"{three_decimals(vote['f1'])} | {one_decimal(vote['seconds_per_image'] * 1000)} |")
        choice = report["choice"]
        lines += ["", f"Choice: best single model `{choice['single']['model']}` "
                      f"(F1 {three_decimals(choice['single']['f1'])}); "
                      f"best vote {' + '.join(choice['vote']['models'])} "
                      f"({choice['vote']['method']}, F1 {three_decimals(choice['vote']['f1'])}): F1 gain "
                      f"{choice['f1_gain']:+.3f} for {choice['time_ratio']} times the time."]
    text = "\n".join(lines) + "\n"
    (set_dir(report["task"], report["on"]) / "evaluation.md").write_text(text)
    return text
