"""Evaluation of scores against the final team rankings, the reference points a scorer must beat, and the human
ceiling.

Metrics for any score per edit (higher = better), on the pairs of one split:
- accuracy: share of pairs where the edit with the higher score is the final winner (equal scores count 1/2),
  with a 95% bootstrap interval over samples (pairs of one sample are not independent);
- weighted accuracy: pairs weighted by |2 p_a - 1|, so pairs the team was unsure about count less;
- confident accuracy: only pairs with p_a >= 0.8 or <= 0.2;
- by source (RealEdit, GIER, ...), and top-1: the highest-scored edit is the final best edit (vs chance).

Baselines (`baselines`), all without training: the edit key order (01 first: the Reddit upvote order and the order
the ranking app showed, so it measures label bias, not quality), image size, and per backbone the cosine
similarities edit-request (CLIP-score), edit-original (preservation) and (edit - original)-request (direction).
Human ceiling: how often one annotator agrees with the other two when those two agree.
Writes data/scorer/baselines.json.
"""

from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from ..common.dataset import images_of
from ..common.io import write_json
from ..paths import SCORER_DIR
from .backbones import BACKBONES
from .data import load_pairs, load_rankings
from .embed import load_embeddings, scope_samples

CONFIDENT = 0.8
EVAL_SPLITS = ("dev", "test")


def _hit(sa: float, sb: float, a_wins: bool) -> float:
    return 0.5 if sa == sb else float((sa > sb) == a_wins)


def bootstrap_ci(per_sample: dict[str, list[float]], n_boot: int = 1000, seed: int = 0) -> list[float]:
    """95% interval of the mean hit rate, resampling samples (with all their pairs)."""
    sums = np.array([sum(h) for h in per_sample.values()])
    counts = np.array([len(h) for h in per_sample.values()])
    if len(sums) < 2:
        return [None, None]
    idx = np.random.default_rng(seed).integers(0, len(sums), size=(n_boot, len(sums)))
    acc = sums[idx].sum(axis=1) / counts[idx].sum(axis=1)
    return [round(float(q), 4) for q in np.percentile(acc, [2.5, 97.5])]


def top1(pairs: list[dict], scores: dict[str, float], rankings: dict[str, dict]) -> dict:
    """Share of samples whose highest-scored edit is the final best edit (ties share the credit)."""
    hits, chance = [], []
    for sid in sorted({p["id"] for p in pairs}):
        order = rankings.get(sid, {}).get("ranking")
        if not order or any(f"{sid}/{k}" not in scores for k in order):
            continue
        s = {k: scores[f"{sid}/{k}"] for k in order}
        best = [k for k in order if s[k] == max(s.values())]
        hits.append((order[0] in best) / len(best))
        chance.append(1 / len(order))
    n = len(hits)
    return {"n": n, "top1": round(sum(hits) / n, 4) if n else None,
            "top1_chance": round(sum(chance) / n, 4) if n else None}


def pair_metrics(pairs: list[dict], scores: dict[str, float], rankings: dict[str, dict]) -> dict:
    per_sample, by_source = defaultdict(list), defaultdict(list)
    weighted, confident = [0.0, 0.0], []
    for p in pairs:
        ka, kb = f"{p['id']}/{p['a']}", f"{p['id']}/{p['b']}"
        if ka not in scores or kb not in scores:
            continue
        h = _hit(scores[ka], scores[kb], p["winner"] == p["a"])
        per_sample[p["id"]].append(h)
        by_source[rankings.get(p["id"], {}).get("source") or "unknown"].append(h)
        w = abs(2 * p["p_a"] - 1)
        weighted[0] += w * h
        weighted[1] += w
        if p["p_a"] >= CONFIDENT or p["p_a"] <= 1 - CONFIDENT:
            confident.append(h)
    hits = [h for hs in per_sample.values() for h in hs]
    mean = lambda xs: round(sum(xs) / len(xs), 4) if xs else None
    return {"pairs": len(hits), "accuracy": mean(hits), "ci95": bootstrap_ci(per_sample),
            "weighted_accuracy": round(weighted[0] / weighted[1], 4) if weighted[1] else None,
            "confident_pairs": len(confident), "confident_accuracy": mean(confident),
            "by_source": {s: {"pairs": len(h), "accuracy": mean(h)} for s, h in sorted(by_source.items())},
            **top1(pairs, scores, rankings)}


def eval_data(version: str = "v1") -> tuple[dict[str, list[dict]], dict[str, dict]]:
    """Pairs of every split, and the final rankings."""
    pairs = defaultdict(list)
    for p in load_pairs(version):
        pairs[p["split"]].append(p)
    return dict(pairs), load_rankings(version)


def evaluate_scores(scores: dict[str, float], pairs: dict[str, list[dict]], rankings: dict[str, dict],
                    splits=EVAL_SPLITS) -> dict:
    return {split: pair_metrics(pairs.get(split, []), scores, rankings) for split in splits}


def human_agreement(pairs: list[dict], rankings: dict[str, dict]) -> dict:
    """Agreement of single annotators with the final winner, and with the other two annotators of the sample when
    those agree (the second is a fair ceiling: the annotator's own vote is not part of the reference)."""
    final, loo = [], []
    for p in pairs:
        votes = {v: r.index(p["a"]) < r.index(p["b"]) for v, r in rankings.get(p["id"], {}).get("rankings", {}).items()}
        for v, prefers_a in votes.items():
            final.append(prefers_a == (p["winner"] == p["a"]))
            others = {u: x for u, x in votes.items() if u != v}
            if len(others) >= 2 and len(set(others.values())) == 1:
                loo.append(prefers_a == next(iter(others.values())))
    mean = lambda xs: round(sum(xs) / len(xs), 4) if xs else None
    return {"annotator_vs_final": mean(final), "annotator_vs_other_two": mean(loo), "votes": len(final),
            "votes_with_agreeing_others": len(loo)}


def _cos(x: np.ndarray, y: np.ndarray) -> float:
    n = float(np.linalg.norm(x) * np.linalg.norm(y))
    return float(x @ y) / n if n > 0 else 0.0


def baseline_scores(version: str = "v1") -> dict[str, dict[str, float]]:
    """Baseline name -> image_id -> score, for every edit of the samples in scope."""
    samples = scope_samples(version)
    edits = [(s, ref) for s in samples for ref in images_of(s) if ref.key != "original"]
    out: dict[str, dict[str, float]] = defaultdict(dict)
    for s, ref in edits:
        out["edit-key-order"][ref.image_id] = -int(ref.key) if ref.key.isdigit() else 0
        with Image.open(ref.path) as im:
            out["image-pixels"][ref.image_id] = im.width * im.height
        out["file-bytes"][ref.image_id] = Path(ref.path).stat().st_size
    for name in BACKBONES:
        images, texts = load_embeddings(name, "images"), load_embeddings(name, "texts")
        for s, ref in edits:
            e, o = images.get(ref.image_id), images.get(f"{s['id']}/original")
            if e is None or o is None:
                continue
            out[f"{name}:edit-original"][ref.image_id] = _cos(e, o)
            t = texts.get(s["id"])
            if t is not None:
                out[f"{name}:edit-request"][ref.image_id] = _cos(e, t)
                out[f"{name}:direction"][ref.image_id] = _cos(e - o, t)
    return dict(out)


def baselines(version: str = "v1") -> dict:
    pairs, rankings = eval_data(version)
    report = {"version": version,
              "human": {split: human_agreement(pairs.get(split, []), rankings) for split in EVAL_SPLITS},
              "baselines": {name: evaluate_scores(scores, pairs, rankings)
                            for name, scores in baseline_scores(version).items()}}
    write_json(SCORER_DIR / "baselines.json", report)
    return report
