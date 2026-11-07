"""Write the final dataset: data/pairs.jsonl and data/images/, holding exactly the ranked, unfiltered images.

    python -m src.rank export [--dry-run] [--no-filters]

Reads the final rankings of every version (work/ranking/<version>/final/, from `finalize`), the requests
(work/samples.jsonl) and the image filters' decisions (work/filters/<task>/decisions.jsonl, from
`python -m src.filters vote`, for every task that has one). data/ then holds:

    images/<sample id>/original.jpg, 01.jpg, 02.jpg, ...      the file of image_id "<sample id>/<key>"
    pairs.jsonl                                              one line per pair of edits of a sample

    {"a": "00039bc7754c/01", "b": "00039bc7754c/02", "p": 0.6437, "request": "Can someone fix the exposure of this?",
     "instruction": "Brighten the photo.", "categories": ["Enhance Media"]}

- a, b: image_ids of two edits of the same sample; the original is "<sample id>/original".
- p: probability that a is better than b (the soft label of `finalize`), always at least 0.5, so a is the preferred
  edit (the hard label). Lines are sorted by sample ID, then by the final ranking.
- request: the edit request as written.
- instruction, categories: the simplified instruction and the edit categories of the request (RealEdit's
  instruction where it published one, otherwise the VLM's; categories from the VLM), from
  work/instructions/<client>/final.jsonl (`python -m src.instructions complete`); left out when not there yet.

Filters (PROJECT.md Step 1b and 1d): a flagged edit loses every pair it is in; a sample is dropped when its
original is flagged, or when fewer than 2 edits are left.

Images move, never get deleted: images of samples without a final ranking go to work/unranked/<id>/
(original.jpg, edits/<key>.jpg); flagged images, and every image of a dropped sample, go to
work/filtered/<id>/<key>.jpg. A later export moves them back when the rankings or the decisions change.
"""

import argparse
import json
import os
import shutil
from collections import defaultdict
from pathlib import Path

from ..common.io import iter_jsonl
from ..paths import (FILTERS_DIR, FINAL_PAIRS, INSTRUCTIONS_DIR, RANKING_DIR, SAMPLES_DIR, SAMPLES_JSONL,
                     UNRANKED_DIR, WORK_DIR)

FILTERED_DIR = WORK_DIR / "filtered"
TASKS = ("nsfw", "watermark")
INSTRUCTIONS = INSTRUCTIONS_DIR / "qwen2.5-vl-7b" / "final.jsonl"     # src/instructions/complete.py


def request_analysis() -> dict[str, dict]:
    """Sample ID -> {"instruction", "categories"}, from the request analysis when it exists."""
    if not INSTRUCTIONS.exists():
        return {}
    return {row["id"]: {"instruction": row["instruction"], "categories": row["categories"]}
            for row in iter_jsonl(INSTRUCTIONS)}


def flagged_images() -> dict[str, set[str]]:
    """Task -> flagged image_ids, from every task that has decisions."""
    out = {}
    for task in TASKS:
        path = FILTERS_DIR / task / "decisions.jsonl"
        if path.exists():
            out[task] = {row["image_id"] for row in iter_jsonl(path) if row["flagged"]}
    return out


def ranked_samples() -> tuple[list[tuple[dict, list[str], dict]], dict]:
    """(sample, final order, pair probabilities) of every ranked sample, and the number per version."""
    requests = {sample["id"]: sample for sample in iter_jsonl(SAMPLES_JSONL)}
    out, per_version = [], {}
    for final in sorted(RANKING_DIR.glob("v*/final/rankings.jsonl")):
        prob = defaultdict(dict)
        for pair in iter_jsonl(final.parent / "pairs.jsonl"):
            prob[pair["id"]][(pair["a"], pair["b"])] = pair["p_a"]
            prob[pair["id"]][(pair["b"], pair["a"])] = round(1 - pair["p_a"], 4)
        ranked = [row for row in iter_jsonl(final) if row["status"] == "ranked"]
        for row in ranked:
            sample, order = requests[row["id"]], row["ranking"]
            if sample["original"] != "original.jpg" or sorted(edit["image"] for edit in sample["edits"]) != sorted(
                    f"edits/{key}.jpg" for key in order):
                raise SystemExit(f"{row['id']}: image names do not match the ranking")
            out.append((sample, order, prob[row["id"]]))
        per_version[final.parent.parent.name] = len(ranked)
    out.sort(key=lambda first: first[0]["id"])
    return out, per_version


def final_lines(flagged: set[str]) -> tuple[list[dict], dict[str, set[str]], dict]:
    """Pair lines of the kept samples, the keys each ranked sample keeps in data/, and filter counts."""
    samples, per_version = ranked_samples()
    analysis = request_analysis()
    lines, keep, counts = [], {}, defaultdict(int)
    for sample, order, prob in samples:
        sample_id = sample["id"]
        edits = [key for key in order if f"{sample_id}/{key}" not in flagged]
        counts["edits_flagged"] += len(order) - len(edits)
        if f"{sample_id}/original" in flagged:
            counts["samples_dropped_original_flagged"] += 1
            keep[sample_id] = set()
            continue
        if len(edits) < 2:
            counts["samples_dropped_fewer_than_2_edits"] += 1
            keep[sample_id] = set()
            continue
        keep[sample_id] = {"original", *edits}
        for position, first in enumerate(edits):
            for second in edits[position + 1:]:
                better, worse = (first, second) if prob[(first, second)] >= 0.5 else (second, first)
                lines.append({"a": f"{sample_id}/{better}", "b": f"{sample_id}/{worse}", "p": prob[(better, worse)],
                              "request": sample["request"], **analysis.get(sample_id, {})})
    counts["samples"] = sum(bool(key) for key in keep.values())
    counts["with_instruction"] = sum(1 for sample_id, wanted in keep.items() if wanted and sample_id in analysis)
    return lines, keep, {"ranked_by_version": per_version, **counts}


def locate(sample_id: str, key: str) -> Path | None:
    for path in (SAMPLES_DIR / sample_id / f"{key}.jpg", FILTERED_DIR / sample_id / f"{key}.jpg",
                 UNRANKED_DIR / sample_id / ("original.jpg" if key == "original" else f"edits/{key}.jpg")):
        if path.exists():
            return path
    return None


def remove_empty(folder: Path) -> None:
    subfolders = sorted((path for path in folder.rglob("*") if path.is_dir()), key=lambda path: -len(path.parts))
    for directory in subfolders + [folder]:
        if directory.exists() and not any(directory.iterdir()):
            directory.rmdir()


def plan_moves(keep: dict[str, set[str]]) -> list[tuple[Path, Path]]:
    """(source, target) of every image that must move."""
    keys = {sample["id"]: ["original", *(Path(edit["image"]).stem for edit in sample["edits"])]
            for sample in iter_jsonl(SAMPLES_JSONL)}
    moves = []
    for sample_id, wanted in keep.items():                     # ranked samples: data/ or work/filtered/
        for key in keys[sample_id]:
            src = locate(sample_id, key)
            if src is None:
                raise SystemExit(f"image {sample_id}/{key} not found")
            dst = SAMPLES_DIR / sample_id / f"{key}.jpg" if key in wanted else FILTERED_DIR / sample_id / f"{key}.jpg"
            if src != dst:
                moves.append((src, dst))
    present = [path.name for path in SAMPLES_DIR.iterdir() if path.is_dir()] if SAMPLES_DIR.exists() else []
    for sample_id in present:                                  # no longer ranked: work/unranked/
        if sample_id not in keep:
            for image_file in sorted((SAMPLES_DIR / sample_id).glob("*.jpg")):
                target = "original.jpg" if image_file.stem == "original" else f"edits/{image_file.name}"
                moves.append((image_file, UNRANKED_DIR / sample_id / target))
    return moves


def export(dry_run: bool = False, use_filters: bool = True) -> dict:
    by_task = flagged_images() if use_filters else {}
    flagged = set().union(*by_task.values()) if by_task else set()
    lines, keep, counts = final_lines(flagged)
    moves = plan_moves(keep)
    summary = {**counts, "pairs": len(lines), "filters": {task_name: len(ids) for task_name, ids in by_task.items()},
               "images_to_move": len(moves)}
    if dry_run:
        return summary
    touched = set()
    for src, dst in moves:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(src, dst)
        touched.add(src.parents[1] if src.parent.name == "edits" else src.parent)
    for folder in touched:
        if folder.exists():
            remove_empty(folder)
    for root in (FILTERED_DIR, UNRANKED_DIR):
        if root.exists() and not any(root.iterdir()):
            root.rmdir()
    FINAL_PAIRS.parent.mkdir(parents=True, exist_ok=True)
    tmp = FINAL_PAIRS.with_name(f".{FINAL_PAIRS.name}.tmp")
    with open(tmp, "w", encoding="utf-8") as handle:
        handle.writelines(json.dumps(line, ensure_ascii=False) + "\n" for line in lines)
    os.replace(tmp, FINAL_PAIRS)
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m src.rank export", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="print what would change, write nothing")
    parser.add_argument("--no-filters", action="store_true", help="ignore the image filters' decisions")
    args = parser.parse_args(argv)
    print(json.dumps(export(args.dry_run, not args.no_filters), indent=1))


if __name__ == "__main__":
    main()
