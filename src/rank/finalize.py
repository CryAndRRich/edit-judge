"""Decide the final ranking of the edits of every sample from the team rankings and, for RealEdit samples,
the Reddit upvote order.

    python -m src.rank finalize [--version v1] [--community-weight 0.5]

Method (see README.md, "Final ranking"):

1. Filter and voters. A sample that at least one annotator skipped is dropped (no ranking). Every annotator
   slot is one voter; the time spent on a sample plays no role.
2. Reliability. Each human voter gets an accuracy p (the chance that it picks the better edit of a pair)
   under the one-coin model: voters are right independently with their own p, so two voters agree with
   probability p_a p_b + (1 - p_a)(1 - p_b). The p values are fitted to the observed agreement of every pair
   of voters (estimate_weights). p is kept in [0.5, 0.95]; a voter at chance level gets weight 0.
   Weight w = log(p / (1 - p)), the optimal weight for a vote of independent voters.
3. Reference vote. For RealEdit samples the upvote order (e1 > e2 > ...) is one more voter, with a fixed
   weight of `--community-weight` times the median human weight. It is not estimated from the data: the app
   shows edit 01 (the most upvoted) first, so agreement with it is inflated by position bias.
4. Pairs. For edits x and y, L = sum of w over voters preferring x minus sum over voters preferring y, and
   P(x better than y) = sigmoid(L).
5. Final ranking. The order with the highest total log P over all its pairs (exact search over orders, by
   dynamic programming over subsets). Ties go to the upvote order, then to the edit key.
6. Status. "skipped" when at least one annotator skipped the sample (no ranking); "ranked" when at least one
   human voter with weight > 0 ranked it; "reference only" otherwise. `confidence` is the mean of P over the
   adjacent pairs of the final ranking.

Writes data/ranking/<version>/final/: rankings.jsonl, pairs.jsonl (soft labels for the scorer), summary.json.
"""

import argparse
import itertools
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

from ..paths import PROVENANCE, RANKING_DIR
from .merge import load, read_results

UPVOTE_REF = re.compile(r"-e(\d+)_")
P_MIN, P_MAX = 0.5, 0.95


def sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x)) if x >= 0 else math.exp(x) / (1 + math.exp(x))


def log_odds(p: float) -> float:
    return math.log(p / (1 - p))


def upvote_orders(ids: set[str]) -> tuple[dict[str, list[str]], dict[str, str]]:
    """Edit keys in upvote order for RealEdit samples, and the source of every sample."""
    orders, sources = {}, {}
    with open(PROVENANCE, encoding="utf-8") as fh:
        for line in fh:
            p = json.loads(line)
            if p["id"] not in ids:
                continue
            sources[p["id"]] = p["source"]
            if p["source"] == "realedit":
                rank = {Path(e["image"]).stem: int(UPVOTE_REF.search(e["ref"]).group(1)) for e in p["edits"]}
                orders[p["id"]] = sorted(rank, key=rank.get)
    return orders, sources


def votes_of(sample_records: dict) -> tuple[dict[str, list[str]], dict[str, str]]:
    """Rankings by voter, and skip reasons by annotator slot."""
    ranked, skipped = {}, {}
    for slot, r in sample_records.items():
        if r["skipped"]:
            skipped[slot] = r["skip_reason"]
        else:
            ranked[slot] = r["ranking"]
    return ranked, skipped


def pair_signs(ranking: list[str], keys: list[str]) -> dict[tuple[str, str], int]:
    pos = {k: i for i, k in enumerate(ranking)}
    return {(x, y): 1 if pos[x] < pos[y] else -1 for x, y in itertools.combinations(keys, 2)}


def estimate_weights(units: list[dict[str, int]], iterations: int = 200) -> tuple[dict, dict]:
    """One-coin model: voter v picks the better edit with probability p_v, independently of the others.
    Two voters then agree with probability p_a p_b + (1 - p_a)(1 - p_b), so 2 * agreement - 1 = d_a * d_b
    with d = 2p - 1. d is fitted to the observed agreement of every voter pair (least squares weighted by
    the number of shared pairs, alternating updates). `units` holds, per pair of edits, voter -> +1/-1."""
    n, m = Counter(), Counter()
    for u in units:
        for a, b in itertools.combinations(sorted(u), 2):
            n[(a, b)] += 1
            m[(a, b)] += u[a] == u[b]
    voters = sorted({v for u in units for v in u})
    links = defaultdict(list)                             # voter -> [(other voter, shared pairs, 2 * agreement - 1)]
    for (a, b), k in n.items():
        r = 2 * m[(a, b)] / k - 1
        links[a].append((b, k, r))
        links[b].append((a, k, r))
    d = dict.fromkeys(voters, 0.3)
    for _ in range(iterations):
        new = {}
        for v in voters:
            num = sum(k * r * d[o] for o, k, r in links[v])
            den = sum(k * d[o] ** 2 for o, k, r in links[v])
            new[v] = min(max(num / den, 0.0), 2 * P_MAX - 1) if den else 0.0
        done = max(abs(new[v] - d[v]) for v in voters) < 1e-9
        d = new
        if done:
            break
    p = {v: (1 + d[v]) / 2 for v in voters}
    w = {v: log_odds(p[v]) if p[v] > P_MIN else 0.0 for v in voters}
    return w, p


def best_order(keys: list[str], logp: dict[tuple[str, str], float], tie: list[str]) -> list[str]:
    """The order of `keys` maximising the sum of logp[(x, y)] over x placed before y (exact, subset DP)."""
    n = len(keys)
    full = (1 << n) - 1
    gain = [[logp[(keys[i], keys[j])] if i != j else 0.0 for j in range(n)] for i in range(n)]
    rank_of = {k: i for i, k in enumerate(tie)}

    @lru_cache(maxsize=None)
    def solve(mask: int) -> tuple[float, tuple[int, ...]]:
        if mask == full:
            return 0.0, ()
        best = None
        for i in sorted((i for i in range(n) if not mask >> i & 1), key=lambda i: rank_of[keys[i]]):
            g = sum(gain[i][j] for j in range(n) if j != i and not mask >> j & 1)
            rest, order = solve(mask | 1 << i)
            if best is None or g + rest > best[0] + 1e-12:
                best = (g + rest, (i, *order))
        return best

    return [keys[i] for i in solve(0)[1]]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m src.rank finalize", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--version", default="v1")
    parser.add_argument("--community-weight", type=float, default=0.5,
                        help="weight of the upvote order, as a multiple of the median human weight")
    args = parser.parse_args(argv)

    manifest, chunk_of, team, keys = load(args.version)
    records, names, _ = read_results(args.version, chunk_of, team, keys)
    upvotes, sources = upvote_orders(set(chunk_of))

    samples = {}
    for sid in chunk_of:
        ranked, skipped = votes_of(records.get(sid, {}))
        samples[sid] = (ranked, skipped)
    units = [{v: pair_signs(rk, keys[sid])[pair] for v, rk in ranked.items()}
             for sid, (ranked, skipped) in samples.items() if not skipped
             for pair in itertools.combinations(keys[sid], 2)]
    w, p = estimate_weights(units)
    human_median = statistics.median(x for x in w.values() if x > 0)
    w_ref = args.community_weight * human_median

    out = RANKING_DIR / args.version / "final"
    out.mkdir(exist_ok=True)
    status, conf_all, ref_agree, team_vs_ref = Counter(), [], [0, 0], [0, 0]
    with open(out / "rankings.jsonl", "w") as fr, open(out / "pairs.jsonl", "w") as fp:
        for sid in chunk_of:
            ranked, skipped = samples[sid]
            k = keys[sid]
            ref = upvotes.get(sid)
            base = {"id": sid, "source": sources.get(sid), "chunk": chunk_of[sid], "n_edits": len(k),
                    "rankings": {v: rk for v, rk in ranked.items()}, "skips": skipped}
            if skipped:
                status["skipped"] += 1
                fr.write(json.dumps({**base, "status": "skipped", "ranking": None}, ensure_ascii=False) + "\n")
                continue
            signs = {v: pair_signs(rk, k) for v, rk in ranked.items()}
            ref_signs = pair_signs(ref, k) if ref else {}
            human_weight = sum(w[v] for v in ranked)
            prob, logp = {}, {}
            for x, y in itertools.combinations(k, 2):
                team_l = sum(w[v] * signs[v][(x, y)] for v in ranked)
                l = team_l + w_ref * ref_signs.get((x, y), 0)
                prob[(x, y)] = sigmoid(l)
                logp[(x, y)] = math.log(max(prob[(x, y)], 1e-12))
                logp[(y, x)] = math.log(max(1 - prob[(x, y)], 1e-12))
                if ref and team_l != 0:                   # does the upvote order agree with the team alone?
                    team_vs_ref[0] += (team_l > 0) == (ref_signs[(x, y)] > 0)
                    team_vs_ref[1] += 1
            order = best_order(k, logp, ref or k)
            pos = {e: i for i, e in enumerate(order)}
            adj = [prob[(a, b)] if (a, b) in prob else 1 - prob[(b, a)] for a, b in zip(order, order[1:])]
            st = "ranked" if human_weight > 0 else "reference only" if ref else "no information"
            status[st] += 1
            conf = round(sum(adj) / len(adj), 3)
            conf_all.append(conf)
            if ref:
                ref_agree[0] += order == ref
                ref_agree[1] += 1
            fr.write(json.dumps({**base, "status": st, "ranking": order, "confidence": conf,
                                 "upvote_order": ref}, ensure_ascii=False) + "\n")
            for (x, y), px in prob.items():
                fp.write(json.dumps({"id": sid, "a": x, "b": y, "p_a": round(px, 4),
                                     "winner": x if pos[x] < pos[y] else y, "status": st}) + "\n")

    summary = {
        "version": args.version,
        "samples": len(chunk_of),
        "status": dict(status),
        "voters": {v: {"accuracy": round(p[v], 3), "weight": round(w[v], 3)}
                   for v in sorted(w, key=lambda v: int(v[1:]))},
        "upvote_order_weight": round(w_ref, 3),
        "confidence": {"mean": round(statistics.mean(conf_all), 3),
                       "quartiles": [round(q, 3) for q in statistics.quantiles(conf_all, n=4)]},
        "final_equals_upvote_order": round(ref_agree[0] / ref_agree[1], 3) if ref_agree[1] else None,
        "upvote_order_agrees_with_team_pairs": round(team_vs_ref[0] / team_vs_ref[1], 3) if team_vs_ref[1] else None,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(f"{summary['samples']} samples: {dict(status)}")
    print("voter accuracy (weight): " + ", ".join(f"{v} {d['accuracy']} ({d['weight']})"
                                                  for v, d in summary["voters"].items()))
    print(f"upvote order weight {summary['upvote_order_weight']}; confidence {summary['confidence']}")
    print(f"upvote order agrees with the team on {summary['upvote_order_agrees_with_team_pairs']} of pairs; "
          f"final ranking equals the upvote order in {summary['final_equals_upvote_order']} of RealEdit samples")
    print(f"wrote {out}/rankings.jsonl, pairs.jsonl, summary.json")


if __name__ == "__main__":
    main()
