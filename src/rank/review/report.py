"""Round diagnostics and paired scorer comparisons on BOTH label versions."""

import json
import itertools
from collections import defaultdict

from ...common.io import iter_jsonl, write_json
from ...paths import SCORER_DIR
from ...scorer.data import split_of
from .common import REVIEW_DIR, dataset, fingerprint, originals, round_dir, selected


def metric_pairs(rows):
    output = defaultdict(list)
    for row in rows:
        sid, a = row["a"].split("/")
        output[split_of(sid)].append({"id": sid, "a": a, "b": row["b"].split("/")[1],
                                     "winner": a, "p_a": row["p"]})
    return output


def report(number: int, before_run: str | None = None, after_run: str | None = None) -> dict:
    folder = round_dir(number)
    applied = json.loads((folder / "applied.json").read_text(encoding="utf-8"))
    after_path = folder / f"labels-r{number}" / "pairs.jsonl"
    if fingerprint(after_path) != applied["pairs_sha256"]:
        raise ValueError("applied labels have changed")
    before_rows, before_samples = dataset(folder / "before-pairs.jsonl")
    after_rows = list(iter_jsonl(after_path))
    after_samples = dataset(after_path)[1] if after_rows else {}
    old = {(row["a"].split("/")[0], *sorted((row["a"], row["b"]))): row for row in before_rows}
    new = {(row["a"].split("/")[0], *sorted((row["a"], row["b"]))): row for row in after_rows}
    counts = defaultdict(lambda: defaultdict(int))
    retained = {row['id'] for row in iter_jsonl(folder / 'decisions.jsonl') if row.get('retained_original')}
    for pick in selected(number):
        bucket, sid = pick["bucket"], pick["id"]
        if sid in retained:
            counts[bucket]['retained_samples'] += 1
            continue
        counts[bucket]["samples"] += 1
        counts[bucket]["top1_changed"] += sid not in after_samples or before_samples[sid]["ranking"][0] != after_samples[sid]["ranking"][0]
        for key, prior in old.items():
            if key[0] != sid:
                continue
            current = new.get(key)
            counts[bucket]["reviewed_pairs"] += 1
            if current is None:
                counts[bucket]["removed_pairs"] += 1
            elif prior["p"] == 0.5 or current["p"] == 0.5:
                counts[bucket]["neutral_pairs"] += 1
            else:
                flipped = prior["a"] != current["a"]
                counts[bucket]["directional_pairs"] += 1
                counts[bucket]["flips"] += flipped
    overall = defaultdict(int)
    for values in counts.values():
        for key, value in values.items():
            overall[key] += value
    counts["overall"] = overall
    buckets = {}
    for bucket, values in counts.items():
        buckets[bucket] = {**values, "flip_rate": values.get("flips", 0) / values["directional_pairs"] if values.get("directional_pairs") else None,
                           "top1_change_rate": values.get("top1_changed", 0) / values["samples"] if values.get("samples") else None}
    merged = json.loads((folder / "merge-summary.json").read_text(encoding="utf-8"))
    if merged["decisions_sha256"] != applied["decisions_sha256"]:
        raise ValueError("decisions changed after applying labels")
    original_reviews = originals()
    original_units = []
    for pick in selected(number):
        sid = pick["id"]
        ranks = list(original_reviews[sid]["rankings"].values())
        if len(ranks) >= 2:
            for a, b in itertools.combinations(before_samples[sid]["keys"], 2):
                original_units.append([int(rank.index(a) < rank.index(b)) for rank in ranks])
    original_pairs = [a == b for unit in original_units for a, b in itertools.combinations(unit, 2)]
    from ..merge import alpha_nominal
    original_agreement = {"pairwise_agreement": sum(original_pairs) / len(original_pairs) if original_pairs else None,
                          "krippendorff_alpha": alpha_nominal(original_units),
                          "scope": "original raw reviews on this round's selected samples"}
    comparisons = {}
    if bool(before_run) != bool(after_run):
        raise ValueError("provide both --before-run and --after-run")
    if before_run:
        from ...scorer.evaluate import evaluate_scores, paired_difference, scored_pairs
        scores = {}
        for label, name, expected in (("before", before_run, folder / "before-pairs.jsonl"),
                                       ("after", after_run, after_path)):
            path = SCORER_DIR / "runs" / name
            metrics = json.loads((path / "metrics.json").read_text(encoding="utf-8"))
            if metrics.get("labels_sha256") != fingerprint(expected):
                raise ValueError(f"{name}: label fingerprint is missing or does not match; train on the matching label version")
            scores[label] = {row["image_id"]: row["score"] for row in iter_jsonl(path / "scores.jsonl")}
        for version, path in (("original", REVIEW_DIR / "labels-r0" / "pairs.jsonl"), ("reviewed", after_path)):
            rows, samples = dataset(path)
            pairs = metric_pairs(rows)
            common = {split: scored_pairs(scored_pairs(part, scores["before"]), scores["after"])
                      for split, part in pairs.items()}
            rankings = {sid: {"ranking": sample["ranking"]} for sid, sample in samples.items()
                        if all(f"{sid}/{key}" in table for table in scores.values() for key in sample["keys"])}
            comparisons[version] = {label: evaluate_scores(table, common, rankings) for label, table in scores.items()}
            comparisons[version]["coverage"] = {split: {"available": sum(row["p_a"] != 0.5 for row in part),
                                                        "common": sum(row["p_a"] != 0.5 for row in common[split])}
                                                  for split, part in pairs.items()}
            comparisons[version]["after_minus_before"] = {
                split: paired_difference(pairs.get(split, []), scores["after"], scores["before"]) for split in ("dev", "test")}
    conflict, control = (buckets.get(bucket, {}).get("flip_rate") for bucket in ("conflict", "random_control"))
    control_removed = buckets.get("random_control", {}).get("removed_pairs", 0)
    complete = bool(comparisons) and all(
        comparisons["reviewed"]["coverage"].get(split, {}).get("common", 0) > 0
        and comparisons["reviewed"]["coverage"][split]["common"] == comparisons["reviewed"]["coverage"][split]["available"]
        for split in ("dev", "test"))
    enough = complete and conflict is not None and control is not None and not control_removed and not buckets.get("random_control", {}).get("neutral_pairs", 0)
    stop = enough and (number >= 5 or (conflict < 0.15 and control < 0.05))
    previous_path = round_dir(number - 1) / "report.json" if number > 1 else None
    if enough and previous_path and previous_path.exists() and control < 0.05:
        previous = json.loads(previous_path.read_text(encoding="utf-8"))
        prior_control = previous.get("buckets", {}).get("random_control", {}).get("flip_rate")
        if prior_control is not None and prior_control < 0.05:
            stop = True
    result = {"round": number, "buckets": buckets, "reviewer_agreement": merged,
              "review_mode": merged.get('review_mode', 'independent_reviews'),
              "retained_samples": len(retained),
              "original_reviewer_agreement": original_agreement,
              "scorers": comparisons, "complete": complete, "stop": bool(stop),
              "hard_limit_reached": number >= 5,
              "notes": ["Neutral pairs are excluded from direction accuracy and flip denominators; removals are separate.",
                        "Random controls estimate the remaining unreviewed pool, not already-reviewed samples.",
                        "Small control groups have sampling uncertainty; inspect counts before stopping.",
                        "The reviewed test set is adaptive; retain original-label comparisons and do not select models on test."]}
    result['notes'].append('User approval of a group proposal supplies no independent reviewer agreement.'
                          if merged.get('review_mode') == 'user_approved_group_proposal'
                          else 'Independent reviewer agreement is reported above.')
    write_json(folder / "report.json", result)
    lines = ["# Review round report", "", "| bucket | samples | directional pairs | flip rate | top-1 change |",
             "|---|---:|---:|---:|---:|"]
    for bucket, values in buckets.items():
        lines.append(f"| {bucket} | {values.get('samples', 0)} | {values.get('directional_pairs', 0)} | {values['flip_rate']} | {values['top1_change_rate']} |")
    lines += ["", f"Reviewer agreement: {merged['pairwise_agreement']}; alpha: {merged['krippendorff_alpha']}.",
              f"Original reviewer agreement on these samples: {original_agreement['pairwise_agreement']}; alpha: {original_agreement['krippendorff_alpha']}.",
              f"Retrained comparison complete: {result['complete']}. Stop recommendation: {result['stop']}.",
              "", *result["notes"], "", "## Scorer comparisons", "", "```json", json.dumps(comparisons, indent=2), "```"]
    (folder / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result
