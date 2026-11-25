"""Validate independent reviews, accept agreement and require explicit third-person adjudication."""

import itertools
import json
from collections import defaultdict
from pathlib import Path

from ...common.io import iter_jsonl, write_json, write_jsonl
from .common import chunk_metadata, digest, identity, round_dir, selected
from .prepare import documents


def validate_decision(row: dict, keys: set[str], reviewers: list[str]) -> None:
    if row.get("reviewers") != reviewers or not isinstance(row.get("reason"), str) or not row["reason"].strip():
        raise ValueError(f"{row.get('id')}: reviewer identities and a reason are required")
    if row.get("removed"):
        if row.get("ranking") is not None:
            raise ValueError("removed decisions cannot contain a ranking")
        return
    rank = row.get("ranking")
    if not isinstance(rank, list) or len(rank) != len(keys) or set(rank) != keys:
        raise ValueError(f"{row.get('id')}: decision must rank every current edit exactly once")
    if row.get("confidence") not in (0.8, 0.9):
        raise ValueError("review confidence must be 0.9 for agreement or 0.8 for adjudication")
    seen = set()
    for pair in row.get("uncertain_pairs", []):
        if not isinstance(pair, list) or len(pair) != 2 or pair[0] == pair[1] or not set(pair) <= keys:
            raise ValueError("uncertain pairs must contain two different current edit keys")
        canonical = tuple(sorted(pair))
        if canonical in seen:
            raise ValueError("duplicate uncertain pair")
        seen.add(canonical)


def merge(number: int, results: Path) -> dict:
    folder = round_dir(number)
    if (folder / "applied.json").exists():
        raise ValueError("this round is already applied; its decisions are frozen")
    manifest, mapping = chunk_metadata(number)
    received, calibration, seen_files = defaultdict(dict), defaultdict(dict), set()
    paths = sorted(results.rglob("*")) if results.is_dir() else [results]
    for path in paths:
        if not path.is_file():
            continue
        for source, document in documents(path):
            if not isinstance(document, dict) or "records" not in document:
                continue
            if not isinstance(document["records"], dict):
                raise ValueError(f"{source}: records must be an object")
            chunk = document.get("chunk")
            if chunk not in manifest["chunks"] or document.get("version") != f"review-{number:02d}":
                raise ValueError(f"unexpected chunk or round in {source}")
            entry = manifest["chunks"][chunk]
            reviewer = identity(document["annotator"])
            if reviewer != entry["reviewer"]:
                raise ValueError(f"{source}: unassigned reviewer")
            seen_files.add((chunk, reviewer))
            target = calibration if entry["kind"] == "calibration" else received
            for sid, record in document["records"].items():
                if sid not in mapping[chunk]:
                    raise ValueError(f"{source}: sample not assigned to this chunk: {sid}")
                labels = mapping[chunk][sid]
                if not isinstance(record.get("skipped"), bool):
                    raise ValueError(f"{source}: skipped must be boolean")
                if record["skipped"]:
                    if not str(record.get("skip_reason") or "").strip():
                        raise ValueError(f"{source}: skip needs a reason")
                    decoded = {"skipped": True, "ranking": None, "skip_reason": record["skip_reason"]}
                else:
                    rank = record.get("ranking")
                    if not isinstance(rank, list) or len(rank) != len(labels) or set(rank) != set(labels):
                        raise ValueError(f"{source}: invalid or incomplete ranking")
                    decoded = {"skipped": False, "ranking": [labels[key] for key in rank]}
                if reviewer in target[sid] and target[sid][reviewer] != decoded:
                    raise ValueError(f"conflicting duplicate review: {sid}/{reviewer}")
                target[sid][reviewer] = decoded
    adjudications = {}
    for row in iter_jsonl(folder / "adjudications.jsonl"):
        if row["id"] in adjudications or row["id"] not in manifest["assignments"]:
            raise ValueError("duplicate or unselected adjudication")
        adjudications[row["id"]] = row
    decisions, pending, units, reviews = [], [], [], []
    for pick in selected(number):
        sid, pair = pick["id"], manifest["assignments"][pick["id"]]
        records = received.get(sid, {})
        reviews.append({"id": sid, "reviews": records})
        keys = set(next(labels[sid].values() for labels in mapping.values() if sid in labels))
        if len(records) != 2:
            pending.append({"id": sid, "status": "missing review", "missing": sorted(set(pair) - set(records))})
            continue
        ranks = [records[name]["ranking"] for name in pair if not records[name]["skipped"]]
        if len(ranks) == 2:
            for a, b in itertools.combinations(sorted(keys), 2):
                units.append([int(rank.index(a) < rank.index(b)) for rank in ranks])
        base = {"id": sid, "reviewers": pair, "uncertain_pairs": [], "removed": False}
        if not ranks:
            decision = {**base, "removed": True, "ranking": None, "reason": "both independent reviewers skipped"}
        elif len(ranks) == 2 and ranks[0] == ranks[1]:
            decision = {**base, "ranking": ranks[0], "confidence": 0.9,
                        "reason": "independent reviewers agreed"}
        elif sid in adjudications:
            decision = {**base, **adjudications[sid], "confidence": 0.8}
            third = identity(decision.get("third_reviewer", ""))
            if third in pair or third not in manifest["reviewers"]:
                raise ValueError(f"{sid}: adjudication needs a third member from the roster")
            decision["third_reviewer"] = third
        else:
            pending.append({"id": sid, "status": "needs discussion", "reviewers": pair,
                            "ranking": None, "third_reviewer": None, "reason": "", "uncertain_pairs": []})
            continue
        validate_decision(decision, keys, pair)
        decisions.append(decision)
    write_jsonl(folder / "reviews.jsonl", reviews)
    write_jsonl(folder / "calibration-reviews.jsonl", ({"id": sid, "reviews": records} for sid, records in sorted(calibration.items())))
    write_jsonl(folder / "decisions.jsonl", decisions)
    write_jsonl(folder / "pending.jsonl", pending)
    from ..merge import alpha_nominal
    agreement = sum(a == b for a, b in units) / len(units) if units else None
    calibration_units = []
    for sid, records in calibration.items():
        ranks = [row["ranking"] for row in records.values() if not row["skipped"]]
        if len(ranks) >= 2:
            for a, b in itertools.combinations(sorted(ranks[0]), 2):
                calibration_units.append([int(rank.index(a) < rank.index(b)) for rank in ranks])
    summary = {"complete": not pending, "decisions": len(decisions), "pending": len(pending),
               "pairwise_agreement": agreement,
               "krippendorff_alpha": alpha_nominal(units) if units else None,
               "calibration_alpha": alpha_nominal(calibration_units) if calibration_units else None,
               "calibration_samples_received": len(calibration), "decisions_sha256": digest(decisions)}
    # Constant responses have undefined alpha; alpha_nominal returns None for them.
    write_json(folder / "merge-summary.json", summary)
    (folder / "disagreements.md").write_text("# Pending reviews\n\n" + "\n".join(
        f"- {row['id']}: {row['status']}" for row in pending) + "\n\nCopy discussion entries from pending.jsonl "
        "to adjudications.jsonl. Supply a complete order, a third reviewer and a reason; rerun review-merge.\n", encoding="utf-8")
    return summary
