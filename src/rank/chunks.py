"""Split a frozen dataset version into ranking chunks, assign each chunk to several annotators,
and package chunks and the app as zip files to hand out.

    python -m src.rank plan [--version v1] [--chunks 39] [--annotators 9] [--per-sample 3]
    python -m src.rank zip  [--version v1] [--only v1_c01 ...]
    python -m src.rank app
    python -m src.rank results --name Hai

`plan` freezes the version: it writes data/ranking/<version>/manifest.json (chunk -> sample IDs) and
assignments.csv (chunk, annotator slot, pass). A later version only takes samples that no earlier
manifest contains, so new data gets its own chunks. Zip files go to data/ranking/dist/.
"""

import argparse
import csv
import itertools
import json
import random
import sys
import time
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from ..paths import RANKING_APP as APP, RANKING_DIR, RANKING_DIST as DIST, SAMPLES_DIR, SAMPLES_JSONL


def load_samples() -> list[dict]:
    with open(SAMPLES_JSONL, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]


def earlier_ids(version: str) -> set[str]:
    ids = set()
    for manifest in RANKING_DIR.glob("v*/manifest.json"):
        if manifest.parent.name != version:
            for chunk in json.loads(manifest.read_text())["chunks"].values():
                ids.update(chunk)
    return ids


def split(samples: list[dict], n_chunks: int, rng: random.Random) -> list[list[str]]:
    """Equal-size chunks with a similar mix of edit counts (so similar workload): sort by number
    of edits, then deal in snake order. Samples are shuffled within each chunk."""
    rng.shuffle(samples)
    samples.sort(key=lambda s: len(s["edits"]), reverse=True)
    chunks = [[] for _ in range(n_chunks)]
    for i, s in enumerate(samples):
        lap, pos = divmod(i, n_chunks)
        chunks[pos if lap % 2 == 0 else n_chunks - 1 - pos].append(s["id"])
    for chunk in chunks:
        rng.shuffle(chunk)
    return chunks


def assign(n_chunks: int, n_annotators: int, per_sample: int) -> list[list[int]]:
    """Give each chunk `per_sample` different annotators: loads stay within one chunk of each
    other, and every pair of annotators shares a similar number of chunks (for agreement)."""
    load, pairs, out = Counter(), Counter(), []
    for _ in range(n_chunks):
        best = min(itertools.combinations(range(n_annotators), per_sample),
                   key=lambda t: (max(load[a] for a in t), sum(load[a] for a in t),
                                  sum(pairs[p] for p in itertools.combinations(t, 2)), t))
        for a in best:
            load[a] += 1
        for p in itertools.combinations(best, 2):
            pairs[p] += 1
        out.append(list(best))
    return out


def passes(teams: list[list[int]], n_annotators: int) -> list[dict[int, int]]:
    """Order each chunk's annotators as pass 1, 2, 3, ... so that every annotator has a similar
    number of first-pass chunks. Doing pass-1 chunks first covers all data once early."""
    count, out = Counter(), []
    for team in teams:
        order, left = {}, list(team)
        for p in range(1, len(team) + 1):
            a = min(left, key=lambda x: (count[(x, p)], x))
            order[a] = p
            count[(a, p)] += 1
            left.remove(a)
        out.append(order)
    return out


def plan(args) -> None:
    out = RANKING_DIR / args.version
    if (out / "manifest.json").exists() and not args.force:
        sys.exit(f"{args.version} is frozen ({out / 'manifest.json'} exists); use --force to replace it")
    taken = earlier_ids(args.version)
    samples = [s for s in load_samples() if s["id"] not in taken]
    if not samples:
        sys.exit("no samples left that earlier versions do not already contain")
    rng = random.Random(args.seed)
    chunks = split(samples, args.chunks, rng)
    names = [f"{args.version}_c{i:02d}" for i in range(1, args.chunks + 1)]
    teams = assign(args.chunks, args.annotators, args.per_sample)
    order = passes(teams, args.annotators)

    out.mkdir(parents=True, exist_ok=True)
    manifest = {"version": args.version, "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "seed": args.seed, "annotators": args.annotators, "per_sample": args.per_sample,
                "samples": len(samples), "chunks": dict(zip(names, chunks))}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    with open(out / "assignments.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["annotator", "pass", "chunk", "samples"])
        rows = [(f"R{a + 1}", p, name, len(chunk))
                for name, chunk, o in zip(names, chunks, order) for a, p in o.items()]
        w.writerows(sorted(rows, key=lambda r: (int(r[0][1:]), r[1], r[2])))

    load = Counter(a for t in teams for a in t)
    pair = Counter(p for t in teams for p in itertools.combinations(t, 2))
    first = Counter(a for o in order for a, p in o.items() if p == 1)
    sizes = [len(c) for c in chunks]
    print(f"{len(samples)} samples -> {args.chunks} chunks of {min(sizes)}-{max(sizes)}; "
          f"each chunk ranked by {args.per_sample} of {args.annotators} annotators")
    print(f"chunks per annotator: {min(load.values())}-{max(load.values())}; "
          f"samples per annotator: about {len(samples) * args.per_sample // args.annotators}")
    print(f"shared chunks per annotator pair: {min(pair.values())}-{max(pair.values())}; "
          f"first-pass chunks per annotator: {min(first.values())}-{max(first.values())}")
    print(f"wrote {out / 'manifest.json'} and {out / 'assignments.csv'}")


def zip_chunks(args) -> None:
    manifest = json.loads((RANKING_DIR / args.version / "manifest.json").read_text())
    by_id = {s["id"]: s for s in load_samples()}
    DIST.mkdir(exist_ok=True)
    for name, ids in manifest["chunks"].items():
        if args.only and name not in args.only:
            continue
        records = []
        target = DIST / f"{name}.zip"
        with zipfile.ZipFile(target, "w", zipfile.ZIP_STORED) as zf:  # images are already compressed
            for sid in ids:
                s, folder = by_id[sid], SAMPLES_DIR / sid
                rec = {"id": sid, "request": s["request"], "original": f"samples/{sid}/{s['original']}", "edits": []}
                zf.write(folder / s["original"], f"{name}/{rec['original']}")
                for e in s["edits"]:
                    key = Path(e["image"]).stem
                    rec["edits"].append({"key": key, "image": f"samples/{sid}/{e['image']}", "note": e.get("note")})
                    zf.write(folder / e["image"], f"{name}/samples/{sid}/{e['image']}")
                records.append(rec)
            zf.writestr(f"{name}/chunk.json", json.dumps(
                {"version": manifest["version"], "chunk": name, "samples": records}, ensure_ascii=False, indent=1))
        print(f"{target.name}: {len(records)} samples, {target.stat().st_size / 1e6:.0f} MB")


APP_ZIP = "ranking_app_v2.zip"

# The repository has a single README; annotators get this text as README.txt inside the app zip.
ANNOTATOR_README = """Edit ranking app
================

You rank community photo edits from best to worst. Full instructions are shown in the app.

Start
-----
1. Install Python 3.10 or newer if you do not have it (https://www.python.org/downloads/;
   on Windows, tick "Add python.exe to PATH" during the installation).
2. Put the chunk zip files you were given (e.g. v1_c07.zip) into chunks/. Do not unzip them.
3. Double-click the start file:
   - Windows: run_windows.bat
   - macOS: run_mac.command (if macOS blocks it, right-click it, choose Open, then Open again)
   - Linux: run "bash run_mac.command" in a terminal
   The first start installs the app into .venv/ (a few minutes, needs internet). The app opens in
   your browser; keep the black window open while you rank.
4. Enter your name (always the same), choose a chunk, and press Start.

Updating from an older version
------------------------------
Unzip the new version over the old ranking_app folder. chunks/ and results/ stay as they are.

Disk space
----------
The app makes smaller copies of large images in chunks/<chunk>/_display/. When a chunk is finished
and its result is sent, you can delete both its folder and its zip file in chunks/.

Sending results
---------------
Your work is saved after every sample in results/<chunk>__<your name>.json. When a chunk is
finished, send that file to the coordinator, and keep it until the coordinator confirms.

The photos come from community posts and include real people: do not share them outside the team.
"""


def zip_app(_args) -> None:
    """The app folder to hand out: unzip, put chunk zips in chunks/, double-click a run_* file."""
    DIST.mkdir(exist_ok=True)
    target = DIST / APP_ZIP
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as zf:
        info = zipfile.ZipInfo("ranking_app/README.txt", date_time=time.localtime()[:6])
        info.compress_type = zipfile.ZIP_DEFLATED
        info.external_attr = 0o644 << 16
        zf.writestr(info, ANNOTATOR_README.replace("\n", "\r\n"))   # readable in Notepad too
        for name in ("app.py", "requirements.txt", "run_mac.command", "run_windows.bat"):
            data = (APP / name).read_bytes()
            if name.endswith(".bat"):
                data = data.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")   # Windows line endings
            info = zipfile.ZipInfo(f"ranking_app/{name}", date_time=time.localtime()[:6])
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = (0o755 if name.startswith("run_") else 0o644) << 16   # run files executable
            zf.writestr(info, data)
        for folder in ("chunks", "results"):
            info = zipfile.ZipInfo(f"ranking_app/{folder}/", date_time=time.localtime()[:6])
            info.external_attr = (0o40755 << 16) | 0x10
            zf.writestr(info, "")
    print(f"wrote {target}")


def zip_results(args) -> None:
    """All result files of one annotator, to send to the coordinator."""
    files = sorted((APP / "results").glob(f"*__{args.annotator_name}.json"))
    if not files:
        sys.exit(f"no results/*__{args.annotator_name}.json")
    DIST.mkdir(exist_ok=True)
    target = DIST / f"results_{args.annotator_name}.zip"
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in files:
            zf.write(f, f"results_{args.annotator_name}/{f.name}")
    print(f"wrote {target} ({len(files)} files)")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m src.rank", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["plan", "zip", "app", "results"])
    parser.add_argument("--version", default="v1")
    parser.add_argument("--chunks", type=int, default=39)
    parser.add_argument("--annotators", type=int, default=9)
    parser.add_argument("--per-sample", type=int, default=3)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--force", action="store_true", help="plan: replace an existing (frozen) version")
    parser.add_argument("--only", nargs="*", help="zip: chunk names to package (default: all)")
    parser.add_argument("--name", dest="annotator_name", help="results: annotator name in the result files")
    args = parser.parse_args(argv)
    {"plan": plan, "zip": zip_chunks, "app": zip_app, "results": zip_results}[args.command](args)


if __name__ == "__main__":
    main()
