"""Select a stratified review round. Reserve random controls BEFORE targeted selection."""

import json
import math
import random
from collections import Counter, defaultdict

from ...common.io import iter_jsonl, write_json, write_jsonl
from ...paths import FINAL_PAIRS, RANKING_DIR
from ...scorer.data import split_of
from .common import REVIEW_DIR, dataset, digest, fingerprint, frozen_json, originals, round_dir

SHARES = {"conflict": 0.5, "label_doubt": 0.2, "model_doubt": 0.1, "random_control": 0.2}


def allocate(total: int, weights: dict) -> dict:
    if not weights or sum(weights.values()) == 0:
        return {key: 0 for key in weights}
    exact = {key: total * weight / sum(weights.values()) for key, weight in weights.items()}
    counts = {key: math.floor(value) for key, value in exact.items()}
    for key in sorted(exact, key=lambda key: (-(exact[key] - counts[key]), key))[:total - sum(counts.values())]:
        counts[key] += 1
    return counts


def pair_signals(row: dict, doubtful: bool) -> dict:
    probability, label = float(row["P_model"]), float(row["p"])
    if not all(math.isfinite(x) for x in (probability, label, float(row["spread"]))) or not 0 <= probability <= 1:
        raise ValueError("invalid OOF probabilities")
    # Canonical pairs always put the preferred edit in a; a neutral label has no opposite direction.
    opposite = 1 - probability
    votes = sum(value < 0.5 for value in row.get("baselines", {}).values())
    return {"conflict": opposite * (1 + 0.5 * votes) if opposite > 0.5 and label > 0.5 else 0.0,
            "raw_conflict": opposite if label > 0.5 else 0.0,
            "label_doubt": 1 - 2 * abs(label - 0.5) + (0.5 if doubtful else 0),
            "model_doubt": 1 - 2 * abs(probability - 0.5) + float(row["spread"])}


def select(number: int, size: int = 250, seed: int = 42, shortlist: bool = False) -> dict:
    if size < 5:
        raise ValueError("select at least 5 samples to retain a random control")
    folder = round_dir(number)
    inputs_path = folder / "oof-inputs.json"
    if not inputs_path.exists() or not (folder / "oof_pairs.jsonl").exists():
        raise ValueError("complete OOF scoring before selecting a round")
    inputs = json.loads(inputs_path.read_text(encoding="utf-8"))
    if fingerprint(FINAL_PAIRS) != inputs["pairs_sha256"]:
        raise ValueError("active labels changed since OOF scoring")
    _, samples = dataset()
    original = originals()
    earlier = {row["id"] for path in REVIEW_DIR.glob("round-*/selected.jsonl")
               if int(path.parent.name.split("-")[-1]) < number for row in iter_jsonl(path)}
    # Coordinator metadata is optional; never invent the hard/noisy replacement history from final labels.
    noisy, hard = set(), set()
    for path in RANKING_DIR.glob("v*/final/rankings.jsonl"):
        noisy.update(row["id"] for row in iter_jsonl(path) if row.get("replaced"))
    for path in RANKING_DIR.glob("v*/final/pairs.jsonl"):
        hard.update((row["id"], *sorted((row["a"], row["b"]))) for row in iter_jsonl(path) if row.get("hard"))
    signals, covered = defaultdict(lambda: dict.fromkeys([*SHARES, "raw_conflict"], 0.0)), set()
    predictions = list(iter_jsonl(folder / "oof_pairs.jsonl"))
    expected = {(row["a"], row["b"]) for sample in samples.values() for row in sample["pairs"]}
    for row in predictions:
        sid = row["id"]
        if sid not in samples or sid not in original:
            raise ValueError(f"missing original metadata or unexpected OOF sample: {sid}")
        if (row["a"], row["b"]) in covered:
            raise ValueError("duplicate OOF pair")
        covered.add((row["a"], row["b"]))
        a, b = row["a"].split("/")[1], row["b"].split("/")[1]
        ranks = original[sid]["rankings"]
        votes = {order.index(a) < order.index(b) for order in ranks.values()}
        values = pair_signals(row, len(votes) > 1 or sid in noisy or (sid, *sorted((a, b))) in hard)
        multiplier = 2 if samples[sid]["ranking"][0] in (a, b) else 1
        for key, value in values.items():
            signals[sid][key] = max(signals[sid][key], value * (1 if key == "raw_conflict" else multiplier))
    if covered != expected:
        raise ValueError("OOF predictions must cover every active pair exactly once")
    eligible = {sid for sid in samples if not original[sid]["skips"]
                and (sid not in earlier or signals[sid]["raw_conflict"] > 0.8)}
    if shortlist:
        candidates = sorted(eligible, key=lambda sid: (-signals[sid]["conflict"], -signals[sid]["label_doubt"], sid))[:size]
        write_jsonl(folder / "baseline-candidates.jsonl", ({"id": sid, "signals": signals[sid]} for sid in candidates))
        wanted = set(candidates)
        path = folder / "baseline-pairs.jsonl"
        write_jsonl(path, (row for sid, sample in samples.items() if sid in wanted for row in sample["pairs"]))
        return {"samples": len(wanted), "edits": sum(len(samples[sid]["keys"]) for sid in wanted),
                "pairs_file": str(path.resolve()), "purpose": "EditReward shortlist; not a human review selection"}
    if size > len(eligible):
        raise ValueError(f"requested {size} samples, only {len(eligible)} are eligible")
    quotas = allocate(size, SHARES)
    rng, chosen, output = random.Random(seed), set(), []
    # Counter preserves insertion order; sort the set so RNG use is stable across Python hash seeds.
    split_weights = Counter(split_of(sid) for sid in sorted(eligible))
    for bucket in ("random_control", "conflict", "label_doubt", "model_doubt"):
        pool = eligible - chosen
        if bucket == "random_control":
            pool -= earlier
        if len(pool) < quotas[bucket]:
            raise ValueError(f"not enough eligible samples for {bucket}")
        counts = allocate(quotas[bucket], split_weights)
        picks = []
        for split, count in counts.items():
            candidates = sorted(sid for sid in pool if split_of(sid) == split)
            if bucket == "random_control":
                rng.shuffle(candidates)
            else:
                candidates.sort(key=lambda sid: (-signals[sid][bucket], sid))
            picks += candidates[:count]
        if len(picks) < quotas[bucket]:
            candidates = sorted(pool - set(picks))
            if bucket == "random_control":
                rng.shuffle(candidates)
            else:
                candidates.sort(key=lambda sid: (-signals[sid][bucket], sid))
            picks += candidates[:quotas[bucket] - len(picks)]
        for sid in picks:
            chosen.add(sid)
            output.append({"id": sid, "bucket": bucket, "split": split_of(sid), "signals": signals[sid],
                           "original_annotators": original[sid]["annotators"]})
    output.sort(key=lambda row: row["id"])
    frozen_json(folder / "selection-inputs.json", {"size": size, "seed": seed,
                "pairs_sha256": fingerprint(FINAL_PAIRS), "oof_sha256": fingerprint(folder / "oof_pairs.jsonl"),
                "originals_sha256": digest(original), "selection_sha256": digest(output)})
    write_jsonl(folder / "selected.jsonl", output)
    summary = {"samples": len(output), "buckets": dict(Counter(row["bucket"] for row in output)),
               "splits": dict(Counter(row["split"] for row in output)),
               "hard_noisy_flags_available": bool(list(RANKING_DIR.glob("v*/final/rankings.jsonl"))),
               "control_population": "eligible samples not selected in an earlier round; stratified by split"}
    write_json(folder / "selection.json", summary)
    (folder / "selection.md").write_text("# Review selection\n\n```json\n" + json.dumps(summary, indent=2)
                                          + "\n```\n", encoding="utf-8")
    return summary
