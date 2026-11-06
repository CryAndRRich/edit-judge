"""Compare detectors and voting methods with labels, on quality and speed.

Labels: on the dataset, hand labels in work/labels/<task>.jsonl ({"image_id", "label"}, 1 = positive); on a public
benchmark, its labels (benchmark.py). For every model and every voting method this reports, at the configured
threshold, accuracy, precision, recall, F1, the false positive rate and the confusion counts; ROC-AUC and average
precision over all thresholds; the best F1 over thresholds (optimistic: chosen on the same labels); the share of
images flagged per class (benchmark image_ids start with their class); and the time per image (models only).

Votes are tried over every subset of two or more models and every method (a weak model can spoil a vote of all
of them). The answer to "vote or single best model?" is `choice`: the best single model and the best vote at the
configured thresholds, the F1 gain of the vote and its time cost (the sum of its models' times).
"""

import itertools
import json
from collections import defaultdict

from ..common.io import iter_jsonl, write_json
from ..paths import LABELS_DIR
from .base import scored_models, set_dir
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
    """Share of images flagged per "<set>/<class>" (benchmark image_ids are "<set>/<class>/<name>")."""
    groups = defaultdict(list)
    for i in labels:
        if i in scores and i.count("/") >= 2:
            groups["/".join(i.split("/")[:2])].append(scores[i] >= threshold)
    return {c: round(sum(v) / len(v), 4) for c, v in sorted(groups.items())} or None


def by_set(scores: dict[str, float], labels: dict[str, int], threshold: float) -> dict:
    groups = defaultdict(dict)
    for i, y in labels.items():
        groups[i.split("/")[0]][i] = y
    return {s: {**prf(scores, ls, threshold), "accuracy": confusion(scores, ls, threshold)["accuracy"]}
            for s, ls in sorted(groups.items())}


def report_for(scores: dict[str, float], labels: dict[str, int], threshold: float, classes: bool) -> dict:
    out = {**prf(scores, labels, threshold), **confusion(scores, labels, threshold), "threshold": threshold,
           "roc_auc": roc_auc(scores, labels), "average_precision": average_precision(scores, labels),
           "best": best_f1(scores, labels)}
    if classes:
        out["by_set"] = by_set(scores, labels, threshold)
        out["flagged_per_class"] = per_class(scores, labels, threshold)
    return out


def subset_votes(scores: dict[str, dict[str, float]], labels: dict[str, int], cfg: dict,
                 seconds: dict[str, float | None], top: int = 15) -> list[dict]:
    """The best `top` votes over every subset of 2+ models and every method, by F1 then time."""
    thresholds, weights, threshold = cfg.get("thresholds", {}), cfg.get("weights", {}), cfg.get("threshold", 0.5)
    rows = []
    for k in range(2, len(scores) + 1):
        for combo in itertools.combinations(sorted(scores), k):
            for method in METHODS:
                if method == "weighted" and not weights:     # the same as "mean"
                    continue
                combined = combine({m: scores[m] for m in combo}, method, thresholds, weights)
                rows.append({"models": list(combo), "method": method, **prf(combined, labels, threshold),
                             "accuracy": confusion(combined, labels, threshold)["accuracy"],
                             "roc_auc": roc_auc(combined, labels),
                             "seconds_per_image": round(sum(seconds.get(m) or 0 for m in combo), 4)})
    rows.sort(key=lambda r: (-r["f1"], r["seconds_per_image"]))
    return rows[:top]


def evaluate(task: str, cfg: dict, on: str = "dataset", models: list[str] | None = None) -> dict:
    """On the dataset: the models of the config. On a benchmark: every model with scores there. `models`
    overrides both."""
    folder = set_dir(task, on)
    if on == "benchmark":
        from .benchmark import read_labels
        labels = read_labels(task)
        models = models or scored_models(task, on)
    else:
        labels = {r["image_id"]: int(r["label"]) for r in iter_jsonl(LABELS_DIR / f"{task}.jsonl")}
        models = models or cfg.get("models", [])
    if not labels:
        raise SystemExit(f"no labels for {task} on the {on}")
    thresholds, weights = cfg.get("thresholds", {}), cfg.get("weights", {})
    scores = load_scores(task, models, on)
    report = {"task": task, "on": on, "labels": len(labels), "positives": sum(labels.values()),
              "models": {}, "voting": {}}
    for m in models:
        meta_path = folder / f"{m}.meta.json"
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        report["models"][m] = {**report_for(scores[m], labels, thresholds.get(m, 0.5), on == "benchmark"),
                               "seconds_per_image": meta.get("seconds_per_image"),
                               "latency_ms": meta.get("latency_ms"), "batch_size": meta.get("batch_size"),
                               "gpu": meta.get("gpu")}
    if len(models) > 1:
        for method in METHODS:
            combined = combine(scores, method, thresholds, weights)
            report["voting"][method] = report_for(combined, labels, cfg.get("threshold", 0.5), on == "benchmark")
        report["voting_seconds_per_image"] = round(sum(r["seconds_per_image"] or 0 for r in report["models"].values()), 4)
        seconds = {m: r["seconds_per_image"] for m, r in report["models"].items()}
        report["subset_votes"] = subset_votes(scores, labels, cfg, seconds)
        single = max(report["models"], key=lambda m: (report["models"][m]["f1"], -(seconds[m] or 0)))
        vote = report["subset_votes"][0]
        report["choice"] = {
            "single": {"model": single, "f1": report["models"][single]["f1"], "seconds_per_image": seconds[single]},
            "vote": vote, "f1_gain": round(vote["f1"] - report["models"][single]["f1"], 4),
            "time_ratio": round(vote["seconds_per_image"] / seconds[single], 2) if seconds[single] else None}
    write_json(folder / "evaluation.json", report)
    return report


def write_markdown(report: dict) -> str:
    """The report's tables (models, models by set, best votes), also written as evaluation.md."""
    f = lambda x: "-" if x is None else f"{x:.3f}"
    ms = lambda x: "-" if x is None else f"{x:.1f}"
    models = sorted(report["models"].items(), key=lambda kv: -kv[1]["f1"])
    lines = [f"# {report['task']} detectors on the {report['on']} ({report['labels']} images, "
             f"{report['positives']} positive)", "",
             "Threshold 0.5 unless set in configs/filters.toml. Latency: one image (batch 1) after a warm-up, "
             "decoding excluded; throughput: batched, in ms per image.", "",
             "| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Latency p50 (ms) | Latency p95 (ms) | Throughput (ms/img) |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for m, r in models:
        lat = r.get("latency_ms") or {}
        sec = r.get("seconds_per_image")
        lines.append(f"| {m} | {f(r['accuracy'])} | {f(r['precision'])} | {f(r['recall'])} | {f(r['f1'])} | "
                     f"{f(r['roc_auc'])} | {ms(lat.get('p50'))} | {ms(lat.get('p95'))} | "
                     f"{ms(sec * 1000 if sec else None)} |")
    sets = sorted({s for _, r in models for s in (r.get("by_set") or {})})
    if len(sets) > 1:
        lines += ["", "F1 (accuracy) by set:", "", "| Model | " + " | ".join(sets) + " |",
                  "|---|" + "---:|" * len(sets)]
        for m, r in models:
            cells = [f"{f(r['by_set'][s]['f1'])} ({f(r['by_set'][s]['accuracy'])})" if s in r["by_set"] else "-"
                     for s in sets]
            lines.append(f"| {m} | " + " | ".join(cells) + " |")
    if report.get("subset_votes"):
        lines += ["", "Best votes over subsets of models:", "",
                  "| Models | Method | Accuracy | Precision | Recall | F1 | Throughput (ms/img, sum) |",
                  "|---|---|---:|---:|---:|---:|---:|"]
        for v in report["subset_votes"][:10]:
            lines.append(f"| {' + '.join(v['models'])} | {v['method']} | {f(v['accuracy'])} | {f(v['precision'])} | "
                         f"{f(v['recall'])} | {f(v['f1'])} | {ms(v['seconds_per_image'] * 1000)} |")
        c = report["choice"]
        lines += ["", f"Choice: best single model `{c['single']['model']}` (F1 {f(c['single']['f1'])}); best vote "
                      f"{' + '.join(c['vote']['models'])} ({c['vote']['method']}, F1 {f(c['vote']['f1'])}): F1 gain "
                      f"{c['f1_gain']:+.3f} for {c['time_ratio']} times the time."]
    text = "\n".join(lines) + "\n"
    (set_dir(report["task"], report["on"]) / "evaluation.md").write_text(text)
    return text
