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
    position = int(hashlib.sha1(sample_id.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    edge = 0.0
    for name, share in SPLITS:
        edge += share
        if position < edge:
            return name
    return SPLITS[-1][0]


def sample_order(sample_id: str) -> str:
    """A fixed, well-mixed order of samples (by a hash of the ID), used by the reward runner and the latency
    measurement."""
    return hashlib.sha1(f"order:{sample_id}".encode()).hexdigest()


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
    pairs = []
    for pair in iter_jsonl(FINAL_PAIRS):
        sample_id, better = pair["a"].split("/")
        if sample_id in banned or (keep is not None and sample_id not in keep):
            continue
        sample_split = split_of(sample_id)
        if split is None or sample_split == split:
            pairs.append({"id": sample_id, "a": better, "b": pair["b"].split("/")[1], "p_a": pair["p"],
                          "winner": better, "split": sample_split})
    return pairs


def load_rankings(version: str | None = "all") -> dict[str, dict]:
    """Sample ID -> {"ranking": edit keys from best to worst, from the pairs (an edit's rank is its number of wins),
    "source" and "rankings" (by annotator slot) when work/ has them}."""
    wins, keys = defaultdict(lambda: defaultdict(int)), defaultdict(set)
    for pair in load_pairs(version):
        wins[pair["id"]][pair["a"]] += 1
        keys[pair["id"]].update((pair["a"], pair["b"]))
    rankings = {sample_id: {"ranking": sorted(edits, key=lambda edit: (-wins[sample_id][edit], edit))}
                for sample_id, edits in keys.items()}
    if PROVENANCE.exists():
        for record in iter_jsonl(PROVENANCE):
            if record["id"] in rankings:
                rankings[record["id"]]["source"] = record["source"]
    for final in RANKING_DIR.glob("v*/final/rankings.jsonl"):
        for record in iter_jsonl(final):
            if record["id"] in rankings:
                replaced = record.get("replaced", [])
                rankings[record["id"]]["rankings"] = {slot: order for slot, order in record["rankings"].items()
                                                      if slot not in replaced}
                if record.get("source"):
                    rankings[record["id"]].setdefault("source", record["source"])
    return rankings
