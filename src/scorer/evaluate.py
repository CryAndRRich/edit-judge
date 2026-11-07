"""Evaluation of scores against the final rankings, the baselines a scorer is compared with, and the human ceiling.

Metrics for any score per edit (higher = better), on the pairs of one split:
- accuracy: share of pairs where the edit with the higher score is the preferred one (equal scores count 1/2),
  with a 95% bootstrap interval over samples (pairs of one sample are not independent);
- weighted accuracy: pairs weighted by |2 p_a - 1|, so pairs the labels are unsure about count less;
- confident accuracy: only pairs with p_a >= 0.8;
- by source (RealEdit, GIER, ...; only where work/ has the provenance), and top-1: the highest-scored edit is the
  final best edit (vs chance).
`paired_difference` compares two scorers on the same pairs: the accuracy difference with a 95% bootstrap interval
over samples. It is the test for "model A is better than model B"; two separate intervals that overlap are not.

Baselines, all without training on our labels:
- similarity: the cosine between the edit and the text in the space of a dual encoder (embed.py): CLIP-score
  (CLIP ViT-L/14), PickScore, SigLIP and HPSv2.1;
- reward models and VLM judges scored per edit by src/scorer/rewards (ImageReward, EditReward, Qwen2.5-VL judge,
  EditScore): work/scorer/baselines/<model>[.part-*].jsonl.
Human ceiling: how often one annotator agrees with the other two when those two agree (needs work/ranking).
Writes work/scorer/baselines.json.
"""

from collections import defaultdict

import numpy as np

from ..common.io import iter_jsonl, write_json, write_jsonl
from ..paths import SCORER_DIR
from .data import load_pairs, load_rankings
from .embed import load_embeddings

CONFIDENT = 0.8
EVAL_SPLITS = ("dev", "test")
SIMILARITY_BASELINES = {"clip-score": "clip-l14", "pickscore": "pickscore", "siglip": "siglip", "hpsv2.1": "hpsv2"}
BASELINES_DIR = SCORER_DIR / "baselines"


def hit(score_a: float, score_b: float, a_wins: bool) -> float:
    return 0.5 if score_a == score_b else float((score_a > score_b) == a_wins)


def mean_or_none(values) -> float | None:
    return round(sum(values) / len(values), 4) if values else None


def bootstrap_ci(per_sample: dict[str, list[float]], n_boot: int = 1000, seed: int = 0) -> list:
    """95% interval of the mean hit rate, resampling samples (with all their pairs)."""
    sums = np.array([sum(hits) for hits in per_sample.values()])
    counts = np.array([len(hits) for hits in per_sample.values()])
    if len(sums) < 2:
        return [None, None]
    draws = np.random.default_rng(seed).integers(0, len(sums), size=(n_boot, len(sums)))
    accuracy = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    return [round(float(value), 4) for value in np.percentile(accuracy, [2.5, 97.5])]


def top1(pairs: list[dict], scores: dict[str, float], rankings: dict[str, dict]) -> dict:
    """Share of samples whose highest-scored edit is the final best edit (ties share the credit)."""
    hits, chance = [], []
    for sample_id in sorted({pair["id"] for pair in pairs}):
        order = rankings.get(sample_id, {}).get("ranking")
        if not order or any(f"{sample_id}/{key}" not in scores for key in order):
            continue
        sample_scores = {key: scores[f"{sample_id}/{key}"] for key in order}
        best = [key for key in order if sample_scores[key] == max(sample_scores.values())]
        hits.append((order[0] in best) / len(best))
        chance.append(1 / len(order))
    return {"top1_samples": len(hits), "top1": mean_or_none(hits), "top1_chance": mean_or_none(chance)}


def scored_pairs(pairs: list[dict], scores: dict[str, float]) -> list[dict]:
    return [pair for pair in pairs
            if f"{pair['id']}/{pair['a']}" in scores and f"{pair['id']}/{pair['b']}" in scores]


def pair_hit(pair: dict, scores: dict[str, float]) -> float:
    return hit(scores[f"{pair['id']}/{pair['a']}"], scores[f"{pair['id']}/{pair['b']}"], pair["winner"] == pair["a"])


def pair_metrics(pairs: list[dict], scores: dict[str, float], rankings: dict[str, dict]) -> dict:
    per_sample, by_source, confident = defaultdict(list), defaultdict(list), []
    weighted_hits = weight_total = 0.0
    for pair in scored_pairs(pairs, scores):
        pair_result = pair_hit(pair, scores)
        per_sample[pair["id"]].append(pair_result)
        by_source[rankings.get(pair["id"], {}).get("source") or "unknown"].append(pair_result)
        weight = abs(2 * pair["p_a"] - 1)
        weighted_hits += weight * pair_result
        weight_total += weight
        if max(pair["p_a"], 1 - pair["p_a"]) >= CONFIDENT:
            confident.append(pair_result)
    hits = [pair_result for sample_hits in per_sample.values() for pair_result in sample_hits]
    return {"pairs": len(hits), "accuracy": mean_or_none(hits), "ci95": bootstrap_ci(per_sample),
            "weighted_accuracy": round(weighted_hits / weight_total, 4) if weight_total else None,
            "confident_pairs": len(confident), "confident_accuracy": mean_or_none(confident),
            "by_source": {source: {"pairs": len(source_hits), "accuracy": mean_or_none(source_hits)}
                          for source, source_hits in sorted(by_source.items())},
            **top1(pairs, scores, rankings)}


def paired_difference(pairs: list[dict], scores_a: dict[str, float], scores_b: dict[str, float],
                      n_boot: int = 1000, seed: int = 0) -> dict:
    """Accuracy of A minus accuracy of B on the pairs both scored, with a 95% bootstrap interval over samples."""
    per_sample = defaultdict(lambda: [0.0, 0.0, 0])
    for pair in scored_pairs(scored_pairs(pairs, scores_a), scores_b):
        totals = per_sample[pair["id"]]
        totals[0] += pair_hit(pair, scores_a)
        totals[1] += pair_hit(pair, scores_b)
        totals[2] += 1
    if not per_sample:
        return {"pairs": 0, "difference": None, "ci95": [None, None]}
    table = np.array(list(per_sample.values()))
    difference = (table[:, 0].sum() - table[:, 1].sum()) / table[:, 2].sum()
    draws = np.random.default_rng(seed).integers(0, len(table), size=(n_boot, len(table)))
    resampled = (table[draws, 0].sum(axis=1) - table[draws, 1].sum(axis=1)) / table[draws, 2].sum(axis=1)
    return {"pairs": int(table[:, 2].sum()), "difference": round(float(difference), 4),
            "ci95": [round(float(value), 4) for value in np.percentile(resampled, [2.5, 97.5])]}


def eval_data(version: str = "all") -> tuple[dict[str, list[dict]], dict[str, dict]]:
    """Pairs of every split, and the final rankings."""
    pairs = defaultdict(list)
    for pair in load_pairs(version):
        pairs[pair["split"]].append(pair)
    return dict(pairs), load_rankings(version)


def evaluate_scores(scores: dict[str, float], pairs: dict[str, list[dict]], rankings: dict[str, dict],
                    splits=EVAL_SPLITS) -> dict:
    return {split: pair_metrics(pairs.get(split, []), scores, rankings) for split in splits}


def human_agreement(pairs: list[dict], rankings: dict[str, dict]) -> dict:
    """Agreement of single annotators with the final winner, and with the other two annotators of the sample when
    those agree (the second is a fair ceiling: the annotator's own vote is not part of the reference)."""
    with_final, leave_one_out = [], []
    for pair in pairs:
        votes = {slot: order.index(pair["a"]) < order.index(pair["b"])
                 for slot, order in rankings.get(pair["id"], {}).get("rankings", {}).items()}
        for slot, prefers_a in votes.items():
            with_final.append(prefers_a == (pair["winner"] == pair["a"]))
            others = {other: vote for other, vote in votes.items() if other != slot}
            if len(others) >= 2 and len(set(others.values())) == 1:
                leave_one_out.append(prefers_a == next(iter(others.values())))
    return {"annotator_vs_final": mean_or_none(with_final), "annotator_vs_other_two": mean_or_none(leave_one_out),
            "votes": len(with_final), "votes_with_agreeing_others": len(leave_one_out)}


def cosine(first: np.ndarray, second: np.ndarray) -> float:
    norm = float(np.linalg.norm(first) * np.linalg.norm(second))
    return float(first @ second) / norm if norm > 0 else 0.0


def similarity_scores(text_kind: str = "instruction") -> dict[str, dict[str, float]]:
    """Baseline name -> image_id -> cosine(edit, text), for every embedded edit."""
    out = {}
    for name, backbone in SIMILARITY_BASELINES.items():
        images, texts = load_embeddings(backbone, "images"), load_embeddings(backbone, f"texts-{text_kind}")
        if images and texts:
            out[name] = {image_id: cosine(vector, texts[image_id.split("/")[0]])
                         for image_id, vector in images.items()
                         if not image_id.endswith("/original") and image_id.split("/")[0] in texts}
    return out


def baseline_scores() -> dict[str, dict[str, float]]:
    """Baseline -> image_id -> score, from the score files in work/scorer/baselines/ (and their shard files)."""
    out = defaultdict(dict)
    for path in sorted(BASELINES_DIR.glob("*.jsonl")):
        name = path.name.split(".part-")[0].removesuffix(".jsonl")
        out[name].update((row["image_id"], row["score"]) for row in iter_jsonl(path) if row.get("score") is not None)
    return dict(out)


def baselines(version: str = "all", text_kind: str = "instruction") -> dict:
    """Write the similarity baselines' scores next to the reward models' (so the report needs no embeddings),
    then evaluate every baseline and the human ceiling."""
    for name, scores in similarity_scores(text_kind).items():
        write_jsonl(BASELINES_DIR / f"{name}.jsonl", ({"image_id": image_id, "score": round(score, 6)}
                                                      for image_id, score in sorted(scores.items())))
    pairs, rankings = eval_data(version)
    report = {"version": version, "text": text_kind,
              "human": {split: human_agreement(pairs.get(split, []), rankings) for split in EVAL_SPLITS},
              "baselines": {name: evaluate_scores(scores, pairs, rankings)
                            for name, scores in baseline_scores().items()}}
    write_json(SCORER_DIR / "baselines.json", report)
    return report
