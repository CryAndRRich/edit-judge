"""Preference pairs for the scorer, and the split of samples into train, dev and test (PROJECT.md Section 6).

The pairs are those of the final dataset (data/pairs.jsonl, every ranking version). Extra information that is
not in the final dataset (the source of a sample, each annotator's ranking, the ranking version) is read from
work/ when it exists (on the coordinator's computer); without it, results by source and the human ceiling are
left out, and only `version="all"` is possible.
"""

import hashlib
import json
from collections import defaultdict

from ..common.dataset import not_trainable
from ..common.io import iter_jsonl
from ..paths import FINAL_PAIRS, PROVENANCE, RANKING_DIR

SPLITS = (("train", 0.70), ("dev", 0.15), ("test", 0.15))


def split_of(sample_id: str) -> str:
    """Stable split of a sample, from a hash of its ID (so it never changes between runs or machines).
    Near-duplicate originals are already merged by the build, so a split by sample keeps them apart."""
    u = int(hashlib.sha1(sample_id.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    edge = 0.0
    for name, share in SPLITS:
        edge += share
        if u < edge:
            return name
    return SPLITS[-1][0]


def version_ids(version: str | None) -> set[str] | None:
    """Sample IDs of one ranking version (from work/ranking/<version>/manifest.json), or None for "all"."""
    if version in (None, "all"):
        return None
    manifest = RANKING_DIR / version / "manifest.json"
    if not manifest.exists():
        raise SystemExit(f"version {version!r} needs {manifest} (not in the final dataset); use version 'all'")
    return {sid for ids in json.loads(manifest.read_text())["chunks"].values() for sid in ids}


def load_pairs(version: str | None = "all", split: str | None = None) -> list[dict]:
    """Pairs {"id", "a", "b" (edit keys), "p_a", "winner", "split"}: a is the preferred edit and p_a = P(a better
    than b) >= 0.5. Samples whose source forbids training are left out."""
    banned, keep = not_trainable(), version_ids(version)
    out = []
    for p in iter_jsonl(FINAL_PAIRS):
        sid, a = p["a"].split("/")
        if sid in banned or (keep is not None and sid not in keep):
            continue
        s = split_of(sid)
        if split is None or s == split:
            out.append({"id": sid, "a": a, "b": p["b"].split("/")[1], "p_a": p["p"], "winner": a, "split": s})
    return out


def load_rankings(version: str | None = "all") -> dict[str, dict]:
    """Sample ID -> {"ranking": edit keys from best to worst, from the pairs (an edit's rank is its number of wins),
    "source" and "rankings" (by annotator slot) when work/ has them}."""
    wins, keys = defaultdict(lambda: defaultdict(int)), defaultdict(set)
    for p in load_pairs(version):
        wins[p["id"]][p["a"]] += 1
        keys[p["id"]].update((p["a"], p["b"]))
    out = {sid: {"ranking": sorted(k, key=lambda e: (-wins[sid][e], e))} for sid, k in keys.items()}
    if PROVENANCE.exists():
        for r in iter_jsonl(PROVENANCE):
            if r["id"] in out:
                out[r["id"]]["source"] = r["source"]
    for final in RANKING_DIR.glob("v*/final/rankings.jsonl"):
        for r in iter_jsonl(final):
            if r["id"] in out:
                out[r["id"]]["rankings"] = {v: rk for v, rk in r["rankings"].items() if v not in r.get("replaced", [])}
                if r.get("source"):
                    out[r["id"]].setdefault("source", r["source"])
    return out
