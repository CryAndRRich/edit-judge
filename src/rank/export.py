"""Write the final dataset: data/pairs.jsonl and data/images/, holding exactly the ranked, unfiltered images.

    python -m src.rank export [--dry-run] [--no-filters]

Reads the final rankings of every version (work/ranking/<version>/final/, from `finalize`), the requests
(work/samples.jsonl) and the image filters' decisions (work/filters/<task>/decisions.jsonl, from
`python -m src.filters vote`, for every task that has one). data/ then holds:

    images/<sample id>/original.jpg, 01.jpg, 02.jpg, ...      the file of image_id "<sample id>/<key>"
    pairs.jsonl                                              one line per pair of edits of a sample

    {"a": "00039bc7754c/01", "b": "00039bc7754c/02", "p": 0.6437, "request": "Can someone fix the exposure of this?"}

- a, b: image_ids of two edits of the same sample; the original is "<sample id>/original".
- p: probability that a is better than b (the soft label of `finalize`), always at least 0.5, so a is the preferred
  edit (the hard label). Lines are sorted by sample ID, then by the final ranking.
- request: the edit request as written.

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
from ..paths import FILTERS_DIR, FINAL_PAIRS, RANKING_DIR, SAMPLES_DIR, SAMPLES_JSONL, UNRANKED_DIR, WORK_DIR

FILTERED_DIR = WORK_DIR / "filtered"
TASKS = ("nsfw", "watermark")


def flagged_images() -> dict[str, set[str]]:
    """Task -> flagged image_ids, from every task that has decisions."""
    out = {}
    for task in TASKS:
        path = FILTERS_DIR / task / "decisions.jsonl"
        if path.exists():
            out[task] = {r["image_id"] for r in iter_jsonl(path) if r["flagged"]}
    return out


def ranked_samples() -> tuple[list[tuple[dict, list[str], dict]], dict]:
    """(sample, final order, pair probabilities) of every ranked sample, and the number per version."""
    requests = {s["id"]: s for s in iter_jsonl(SAMPLES_JSONL)}
    out, per_version = [], {}
    for final in sorted(RANKING_DIR.glob("v*/final/rankings.jsonl")):
        prob = defaultdict(dict)
        for p in iter_jsonl(final.parent / "pairs.jsonl"):
            prob[p["id"]][(p["a"], p["b"])] = p["p_a"]
            prob[p["id"]][(p["b"], p["a"])] = round(1 - p["p_a"], 4)
        ranked = [r for r in iter_jsonl(final) if r["status"] == "ranked"]
        for r in ranked:
            s, order = requests[r["id"]], r["ranking"]
            if s["original"] != "original.jpg" or sorted(e["image"] for e in s["edits"]) != sorted(
                    f"edits/{k}.jpg" for k in order):
                raise SystemExit(f"{r['id']}: image names do not match the ranking")
            out.append((s, order, prob[r["id"]]))
        per_version[final.parent.parent.name] = len(ranked)
    out.sort(key=lambda x: x[0]["id"])
    return out, per_version


def final_lines(flagged: set[str]) -> tuple[list[dict], dict[str, set[str]], dict]:
    """Pair lines of the kept samples, the keys each ranked sample keeps in data/, and filter counts."""
    samples, per_version = ranked_samples()
    lines, keep, counts = [], {}, defaultdict(int)
    for s, order, prob in samples:
        sid = s["id"]
        edits = [k for k in order if f"{sid}/{k}" not in flagged]
        counts["edits_flagged"] += len(order) - len(edits)
        if f"{sid}/original" in flagged:
            counts["samples_dropped_original_flagged"] += 1
            keep[sid] = set()
            continue
        if len(edits) < 2:
            counts["samples_dropped_fewer_than_2_edits"] += 1
            keep[sid] = set()
            continue
        keep[sid] = {"original", *edits}
        for i, x in enumerate(edits):
            for y in edits[i + 1:]:
                a, b = (x, y) if prob[(x, y)] >= 0.5 else (y, x)
                lines.append({"a": f"{sid}/{a}", "b": f"{sid}/{b}", "p": prob[(a, b)], "request": s["request"]})
    counts["samples"] = sum(bool(k) for k in keep.values())
    return lines, keep, {"ranked_by_version": per_version, **counts}


def locate(sid: str, key: str) -> Path | None:
    for p in (SAMPLES_DIR / sid / f"{key}.jpg", FILTERED_DIR / sid / f"{key}.jpg",
              UNRANKED_DIR / sid / ("original.jpg" if key == "original" else f"edits/{key}.jpg")):
        if p.exists():
            return p
    return None


def remove_empty(folder: Path) -> None:
    for d in sorted((p for p in folder.rglob("*") if p.is_dir()), key=lambda p: -len(p.parts)) + [folder]:
        if d.exists() and not any(d.iterdir()):
            d.rmdir()


def plan_moves(keep: dict[str, set[str]]) -> list[tuple[Path, Path]]:
    """(source, target) of every image that must move."""
    keys = {s["id"]: ["original", *(Path(e["image"]).stem for e in s["edits"])] for s in iter_jsonl(SAMPLES_JSONL)}
    moves = []
    for sid, wanted in keep.items():                     # ranked samples: data/ or work/filtered/
        for key in keys[sid]:
            src = locate(sid, key)
            if src is None:
                raise SystemExit(f"image {sid}/{key} not found")
            dst = SAMPLES_DIR / sid / f"{key}.jpg" if key in wanted else FILTERED_DIR / sid / f"{key}.jpg"
            if src != dst:
                moves.append((src, dst))
    present = [p.name for p in SAMPLES_DIR.iterdir() if p.is_dir()] if SAMPLES_DIR.exists() else []
    for sid in present:                                  # no longer ranked: work/unranked/
        if sid not in keep:
            for f in sorted((SAMPLES_DIR / sid).glob("*.jpg")):
                moves.append((f, UNRANKED_DIR / sid / ("original.jpg" if f.stem == "original" else f"edits/{f.name}")))
    return moves


def export(dry_run: bool = False, use_filters: bool = True) -> dict:
    by_task = flagged_images() if use_filters else {}
    flagged = set().union(*by_task.values()) if by_task else set()
    lines, keep, counts = final_lines(flagged)
    moves = plan_moves(keep)
    summary = {**counts, "pairs": len(lines), "filters": {t: len(v) for t, v in by_task.items()},
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
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.writelines(json.dumps(l, ensure_ascii=False) + "\n" for l in lines)
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
