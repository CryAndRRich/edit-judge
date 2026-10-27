"""Preference pairs for the scorer, and the split of samples into train, dev and test (PROJECT.md Section 6)."""

import hashlib

from ..common.dataset import not_trainable
from ..common.io import iter_jsonl
from ..paths import RANKING_DIR

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


def load_rankings(version: str = "v1") -> dict[str, dict]:
    """Sample ID -> final ranking record {"source", "rankings" (by annotator slot), "ranking", ...} of ranked
    samples."""
    return {r["id"]: r for r in iter_jsonl(RANKING_DIR / version / "final" / "rankings.jsonl")
            if r["status"] == "ranked"}


def load_pairs(version: str = "v1", split: str | None = None) -> list[dict]:
    """Pairs {"id", "a", "b", "p_a", "winner", "split"} of ranked samples; samples whose source forbids training
    are left out."""
    banned = not_trainable()
    out = []
    for p in iter_jsonl(RANKING_DIR / version / "final" / "pairs.jsonl"):
        if p["status"] != "ranked" or p["id"] in banned:
            continue
        s = split_of(p["id"])
        if split is None or s == split:
            out.append({k: p[k] for k in ("id", "a", "b", "p_a", "winner")} | {"split": s})
    return out
