"""Merge the annotators' result files of a ranking version into preference labels, and measure agreement.

    python -m src.rank merge [--version v1]

Reads data/ranking/<version>/results/<slot>/*.json (one folder per annotator slot R1-R9, as in assignments.csv),
checks every file against the manifest and the dataset, and writes data/ranking/<version>/labels/:

- rankings.jsonl: per sample, the rankings and skips of each annotator;
- pairs.jsonl: per sample and pair of edits, the votes and the majority label;
- summary.json: counts and agreement (also printed).

A pair gets a label when a strict majority of the rankings agrees: 3-0 or 2-1 with three rankings, 2-0 with two
(1-1 has no label). `confidence` is the share of rankings that agree with the label.
"""

import argparse
import csv
import itertools
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

from ..paths import RANKING_DIR, SAMPLES_JSONL


def load(version: str):
    manifest = json.loads((RANKING_DIR / version / "manifest.json").read_text())
    chunk_of = {sid: c for c, ids in manifest["chunks"].items() for sid in ids}
    team = defaultdict(set)
    with open(RANKING_DIR / version / "assignments.csv") as fh:
        for row in csv.DictReader(fh):
            team[row["chunk"]].add(row["annotator"])
    keys = {}
    with open(SAMPLES_JSONL, encoding="utf-8") as fh:
        for line in fh:
            s = json.loads(line)
            if s["id"] in chunk_of:
                keys[s["id"]] = sorted(Path(e["image"]).stem for e in s["edits"])
    return manifest, chunk_of, team, keys


def read_results(version: str, chunk_of: dict, team: dict, keys: dict):
    """records[sid][slot] = record; names[slot] = annotator names found. Exits on any invalid file."""
    records, names, seconds, errors = defaultdict(dict), defaultdict(Counter), defaultdict(list), []
    for path in sorted((RANKING_DIR / version / "results").glob("R*/*.json")):
        slot, d = path.parent.name, json.loads(path.read_text(encoding="utf-8"))
        if slot not in team.get(d["chunk"], ()):
            errors.append(f"{path}: {slot} is not assigned to {d['chunk']}")
            continue
        names[slot][d["annotator"]] += 1
        for sid, r in d["records"].items():
            if chunk_of.get(sid) != d["chunk"]:
                errors.append(f"{path}: sample {sid} is not in {d['chunk']}")
            elif sid in records and slot in records[sid]:
                errors.append(f"{path}: sample {sid} appears twice for {slot}")
            elif not r["skipped"] and sorted(r["ranking"] or []) != keys[sid]:
                errors.append(f"{path}: sample {sid} has an incomplete or invalid ranking")
            elif r["skipped"] and not (r.get("skip_reason") or "").strip():
                errors.append(f"{path}: sample {sid} is skipped without a reason")
            else:
                records[sid][slot] = r
                if r.get("seconds") is not None:
                    seconds[slot].append(r["seconds"])
    if errors:
        sys.exit("invalid results:\n" + "\n".join(errors[:30]))
    return records, names, seconds


def prefers(ranking: list[str], a: str, b: str) -> str:
    return a if ranking.index(a) < ranking.index(b) else b


def alpha_nominal(units: list[list[str]]) -> float:
    """Krippendorff's alpha for nominal values; each unit is the list of values given by its coders."""
    o = Counter()
    for vals in units:
        m = len(vals)
        if m < 2:
            continue
        for i, j in itertools.permutations(range(m), 2):
            o[(vals[i], vals[j])] += 1 / (m - 1)
    n_c = Counter()
    for (c, _), v in o.items():
        n_c[c] += v
    n = sum(n_c.values())
    d_o = sum(v for (c, k), v in o.items() if c != k) / n
    d_e = sum(n_c[c] * n_c[k] for c in n_c for k in n_c if c != k) / (n * (n - 1))
    return 1 - d_o / d_e


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m src.rank merge", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--version", default="v1")
    args = parser.parse_args(argv)
    manifest, chunk_of, team, keys = load(args.version)
    records, names, seconds = read_results(args.version, chunk_of, team, keys)
    out = RANKING_DIR / args.version / "labels"
    out.mkdir(exist_ok=True)

    slots = sorted({a for t in team.values() for a in t}, key=lambda a: int(a[1:]))
    pair_votes = defaultdict(lambda: [0, 0])                  # (slot, slot) -> [agree, total] on shared pairs
    loo = defaultdict(lambda: [0, 0])                         # slot -> [agrees with the other two, they agreed]
    top1 = defaultdict(lambda: [0, 0])                        # slot -> [rank 1 is edit "01", rankings]
    chance_top1 = defaultdict(float)
    units, n_pairs, n_labelled, conf = [], 0, 0, Counter()
    status = Counter()
    all_top1_same, three_ranked = 0, 0

    with open(out / "rankings.jsonl", "w") as fr, open(out / "pairs.jsonl", "w") as fp:
        for sid in chunk_of:
            recs = records.get(sid, {})
            ranked = {a: r["ranking"] for a, r in recs.items() if not r["skipped"]}
            skipped = {a: r["skip_reason"] for a, r in recs.items() if r["skipped"]}
            missing = sorted(team[chunk_of[sid]] - set(recs))
            usable = len(ranked) >= 2 and len(skipped) < 2
            status["usable" if usable else "skipped by 2 or more" if len(skipped) >= 2 else "fewer than 2 rankings"] += 1
            fr.write(json.dumps({"id": sid, "chunk": chunk_of[sid], "n_edits": len(keys[sid]), "rankings": ranked,
                                 "skips": skipped, "missing": missing, "usable": usable}, ensure_ascii=False) + "\n")
            for a, rk in ranked.items():
                top1[a][0] += rk[0] == keys[sid][0]
                top1[a][1] += 1
                chance_top1[a] += 1 / len(rk)
            if len(ranked) == 3:
                three_ranked += 1
                all_top1_same += len({rk[0] for rk in ranked.values()}) == 1
            if not usable:
                continue
            for x, y in itertools.combinations(keys[sid], 2):
                votes = {a: prefers(rk, x, y) for a, rk in ranked.items()}
                units.append([v == x for v in votes.values()])   # binary: "x preferred over y"
                for a, b in itertools.combinations(sorted(votes), 2):
                    pair_votes[(a, b)][0] += votes[a] == votes[b]
                    pair_votes[(a, b)][1] += 1
                if len(votes) == 3:
                    for a in votes:
                        others = [v for b, v in votes.items() if b != a]
                        if others[0] == others[1]:
                            loo[a][0] += votes[a] == others[0]
                            loo[a][1] += 1
                c = Counter(votes.values())
                n_x, n_y = c[x], c[y]
                n_pairs += 1
                label = x if n_x > n_y else y if n_y > n_x else None
                if label:
                    n_labelled += 1
                    conf[f"{max(n_x, n_y)}/{n_x + n_y}"] += 1
                fp.write(json.dumps({"id": sid, "a": x, "b": y, "votes_a": n_x, "votes_b": n_y, "label": label,
                                     "confidence": round(max(n_x, n_y) / (n_x + n_y), 3) if label else None}) + "\n")

    agree_all = sum(v[0] for v in pair_votes.values()) / sum(v[1] for v in pair_votes.values())
    per_slot = {}
    for a in slots:
        shared = [v for k, v in pair_votes.items() if a in k]
        rec = [r for s in records.values() for b, r in s.items() if b == a]
        per_slot[a] = {
            "names": dict(names[a]),
            "samples": len(rec),
            "skipped": sum(r["skipped"] for r in rec),
            "agreement_with_others": round(sum(v[0] for v in shared) / sum(v[1] for v in shared), 3),
            "agreement_with_majority_of_other_two": round(loo[a][0] / loo[a][1], 3) if loo[a][1] else None,
            "median_seconds": round(statistics.median(seconds[a]), 1) if seconds[a] else None,
            "rank1_is_first_edit": round(top1[a][0] / top1[a][1], 3),
            "rank1_is_first_edit_by_chance": round(chance_top1[a] / top1[a][1], 3),
        }
    summary = {
        "version": args.version,
        "samples": len(chunk_of),
        "sample_status": dict(status),
        "pairs_of_usable_samples": n_pairs,
        "pairs_labelled": n_labelled,
        "label_confidence": dict(conf),
        "pairwise_agreement": round(agree_all, 3),
        "krippendorff_alpha_pairs": round(alpha_nominal(units), 3),
        "same_best_edit_all_three": round(all_top1_same / three_ranked, 3) if three_ranked else None,
        "annotators": per_slot,
        "annotator_pair_agreement": {f"{a}-{b}": round(v[0] / v[1], 3) for (a, b), v in sorted(pair_votes.items())},
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))

    print(f"{summary['samples']} samples: {dict(status)}")
    print(f"{n_pairs} pairs in usable samples, {n_labelled} labelled; confidence {dict(conf)}")
    print(f"pairwise agreement {summary['pairwise_agreement']}, Krippendorff alpha {summary['krippendorff_alpha_pairs']}, "
          f"same best edit by all three {summary['same_best_edit_all_three']}")
    print(f"{'slot':5}{'samples':>8}{'skip':>6}{'agree':>7}{'vs maj':>8}{'sec':>7}{'top=01':>8}{'chance':>8}  names")
    for a, s in per_slot.items():
        print(f"{a:5}{s['samples']:8}{s['skipped']:6}{s['agreement_with_others']:7}"
              f"{s['agreement_with_majority_of_other_two'] or 0:8}{s['median_seconds'] or 0:7}"
              f"{s['rank1_is_first_edit']:8}{s['rank1_is_first_edit_by_chance']:8}  {', '.join(s['names'])}")
    print(f"wrote {out}/rankings.jsonl, pairs.jsonl, summary.json")


if __name__ == "__main__":
    main()
