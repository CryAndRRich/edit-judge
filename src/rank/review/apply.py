"""Write new pair labels without touching the read-only Kaggle images or the original labels."""

import json
from collections import defaultdict
from pathlib import Path

from ...common.io import iter_jsonl, write_json, write_jsonl
from ...paths import FINAL_PAIRS
from .common import REVIEW_DIR, chunk_metadata, dataset, digest, fingerprint, frozen_json, round_dir, selected
from .merge import validate_decision


def overlay(rows: list[dict], decisions: dict[str, dict]) -> tuple[list[dict], list[dict]]:
    output, changes = [], []
    for row in rows:
        sid, a = row["a"].split("/")
        b = row["b"].split("/")[1]
        decision = decisions.get(sid)
        if not decision or decision.get("retained_original"):
            output.append(row.copy())
            continue
        new = None
        if not decision.get("removed"):
            positions = {key: index for index, key in enumerate(decision["ranking"])}
            winner, loser = (a, b) if positions[a] < positions[b] else (b, a)
            uncertain = {frozenset(pair) for pair in decision.get("uncertain_pairs", [])}
            new = {**row, "a": f"{sid}/{winner}", "b": f"{sid}/{loser}",
                   "p": 0.5 if frozenset((a, b)) in uncertain else decision["confidence"]}
            output.append(new)
        if new != row:
            changes.append({"id": sid, "old": row, "new": new, "reviewers": decision["reviewers"],
                            "third_reviewer": decision.get("third_reviewer"), "reason": decision["reason"]})
    return output, changes


def apply(number: int) -> dict:
    folder = round_dir(number)
    inputs = json.loads((folder / "selection-inputs.json").read_text(encoding="utf-8"))
    if fingerprint(FINAL_PAIRS) != inputs["pairs_sha256"]:
        raise ValueError("active pairs must match the labels used to select this round")
    rows, samples = dataset()
    decisions_list = list(iter_jsonl(folder / "decisions.jsonl"))
    decisions = {row["id"]: row for row in decisions_list}
    summary = json.loads((folder / "merge-summary.json").read_text(encoding="utf-8"))
    approved = summary.get('review_mode') == 'user_approved_group_proposal'
    if approved:
        from .approve import validate_approval
        approval = validate_approval(number, decisions_list)
    else:
        manifest, _ = chunk_metadata(number)
    if (not summary["complete"] or set(decisions) != {row["id"] for row in selected(number)}
            or len(decisions) != len(decisions_list) or digest(decisions_list) != summary["decisions_sha256"]):
        raise ValueError("complete review-merge for every selected sample before applying labels")
    for sid, decision in decisions.items():
        validate_decision(decision, set(samples[sid]["keys"]),
                          [approval['approver']] if approved else manifest["assignments"][sid])
    original = REVIEW_DIR / "labels-r0" / "pairs.jsonl"
    if number == 1:
        if original.exists() and fingerprint(original) != fingerprint(FINAL_PAIRS):
            raise ValueError("original labels-r0 snapshot differs from this dataset")
        if not original.exists():
            original.parent.mkdir(parents=True, exist_ok=True)
            original.write_bytes(FINAL_PAIRS.read_bytes())
    else:
        previous = round_dir(number - 1) / "applied.json"
        if not previous.exists() or not original.exists():
            raise ValueError("apply the previous round and restore the original snapshot first")
        if json.loads(previous.read_text(encoding="utf-8"))["pairs_sha256"] != fingerprint(FINAL_PAIRS):
            raise ValueError("later rounds must start from the preceding round's labels")
    output, changes = overlay(rows, decisions)
    destination = folder / f"labels-r{number}" / "pairs.jsonl"
    frozen_json(folder / "apply-inputs.json", {"before_sha256": fingerprint(FINAL_PAIRS),
                "decisions_sha256": digest(decisions_list), "after_sha256": digest(output)})
    if destination.exists() and list(iter_jsonl(destination)) != output:
        raise ValueError("existing label version differs; refusing to overwrite it")
    if not destination.exists():
        write_jsonl(destination, output)
    before = folder / "before-pairs.jsonl"
    if not before.exists():
        before.write_bytes(FINAL_PAIRS.read_bytes())
    write_jsonl(folder / "changes.jsonl", changes)
    result = {"round": number, "pairs_file": str(destination.resolve()), "pairs": len(output),
              "pairs_sha256": fingerprint(destination), "changed_pairs": len(changes),
              "samples_removed": sum(bool(row.get("removed")) for row in decisions_list),
              "decisions_sha256": digest(decisions_list),
              "review_mode": summary.get('review_mode', 'independent_reviews'),
              "retained_samples": sum(bool(row.get('retained_original')) for row in decisions_list)}
    write_json(folder / "applied.json", result)
    return result


def apply_final_directory(final: Path, reviews_root: Path) -> None:
    """Coordinator's finalize --reviews integration; final outputs are reproducible from frozen originals.

    Keep filtered-out edits at the end of the full ranking; export will filter them as before. A review
    cannot resurrect an edit absent from the final exported dataset.
    """
    decisions = {}
    for path in sorted(reviews_root.glob("round-*/decisions.jsonl")):
        status = path.parent / "merge-summary.json"
        rows = list(iter_jsonl(path))
        if not status.exists():
            raise ValueError(f"missing review-merge summary: {path.parent}")
        summary = json.loads(status.read_text(encoding="utf-8"))
        if not summary["complete"] or summary["decisions_sha256"] != digest(rows):
            raise ValueError(f"incomplete or changed decisions: {path}")
        decisions.update((row["id"], row) for row in rows)
    rankings, pairs = list(iter_jsonl(final / "rankings.jsonl")), list(iter_jsonl(final / "pairs.jsonl"))
    relevant = {row["id"] for row in rankings} & set(decisions)
    for filename in ("rankings.jsonl", "pairs.jsonl"):
        backup = final / f"original-{filename}"
        if not backup.exists():
            backup.write_bytes((final / filename).read_bytes())
    for row in rankings:
        if row["id"] not in relevant:
            continue
        decision = decisions[row["id"]]
        if decision.get('retained_original'):
            continue
        if decision.get("removed"):
            row.update(status="review removed", ranking=None, review=decision)
        else:
            if row["status"] != "ranked" or not set(decision["ranking"]) <= set(row["ranking"]):
                raise ValueError("a review cannot restore a skipped sample or introduce unknown edits")
            row["ranking"] = decision["ranking"] + [key for key in row["ranking"] if key not in decision["ranking"]]
            row["review"] = decision
    updated = []
    for row in pairs:
        decision = decisions.get(row["id"])
        if row["id"] not in relevant or decision is None or decision.get('retained_original'):
            updated.append(row)
        elif not decision.get("removed"):
            if {row["a"], row["b"]} <= set(decision["ranking"]):
                a_wins = decision["ranking"].index(row["a"]) < decision["ranking"].index(row["b"])
                neutral = frozenset((row["a"], row["b"])) in {frozenset(p) for p in decision.get("uncertain_pairs", [])}
                row = {**row, "winner": row["a"] if a_wins else row["b"],
                       "p_a": 0.5 if neutral else decision["confidence"] if a_wins else 1 - decision["confidence"]}
            updated.append(row)
    write_jsonl(final / "rankings.jsonl", rankings)
    write_jsonl(final / "pairs.jsonl", updated)
    from collections import Counter
    summary_path = final / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    summary["review_overlay"] = {"samples": len(relevant), "status": dict(Counter(row["status"] for row in rankings)),
                                 "decisions_sha256": digest({sid: decisions[sid] for sid in sorted(relevant)})}
    write_json(summary_path, summary)
