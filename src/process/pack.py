"""Pack the data needed on Kaggle into one zip file, to upload as a private Kaggle dataset.

    python -m src.process pack [--out data/exports/editjudge-data.zip]

The zip has no top folder, so on Kaggle the dataset root is the data folder itself (EJ_DATA_DIR =
/kaggle/input/<dataset>). It holds what later stages read:
- samples.jsonl and samples/ (the normalized images and sample.json files);
- index/provenance.jsonl (sources, the "trainable" flag, RealEdit upvote order);
- ranking/<version>/manifest.json and assignments.csv, and for ranked versions the final rankings and the merged
  labels (rankings.jsonl, pairs.jsonl).
Left out: raw/ (and its username salt), the build ledger and image cache, the annotators' result files and the
labels summary (they carry team members' names), zip files in ranking/dist, logs and later-stage outputs.
Images are stored without compression (JPEG is already compressed); text files are compressed.
"""

import os
import zipfile
from pathlib import Path

from ..paths import DATA_DIR, PROVENANCE, RANKING_DIR, SAMPLES_DIR, SAMPLES_JSONL

ABOUT = """EditJudge data (private: share only within the team; it contains photos of real people)

samples.jsonl               one line per sample: id, request (as written), original, edits [{image, note}]
samples/<id>/               original.jpg, edits/01.jpg ..., sample.json
index/provenance.jsonl      sample id -> source, keys, trainable flag (exclude trainable=false from training)
ranking/<version>/          manifest.json, assignments.csv; v1: final/ (final rankings, pair labels), labels/

Images: JPEG, sRGB, no metadata, longest side at most 1024 px.
Code: https://github.com/CryAndRRich/edit-judge (set EJ_DATA_DIR to this folder).
"""


def files_to_pack() -> list[Path]:
    files = [SAMPLES_JSONL, PROVENANCE]
    files += sorted(p for p in SAMPLES_DIR.rglob("*") if p.is_file() and not p.name.startswith("."))
    for version in sorted(p for p in RANKING_DIR.glob("v*") if p.is_dir()):
        files += [p for p in (version / "manifest.json", version / "assignments.csv") if p.exists()]
        files += sorted((version / "final").glob("*.jsonl")) + [p for p in [version / "final" / "summary.json"] if p.exists()]
        files += [p for p in (version / "labels" / "rankings.jsonl", version / "labels" / "pairs.jsonl") if p.exists()]
    return files


def pack(out: Path | None = None) -> dict:
    out = out or DATA_DIR / "exports" / "editjudge-data.zip"
    out.parent.mkdir(parents=True, exist_ok=True)
    files = files_to_pack()
    tmp = out.with_name(f".{out.name}.tmp")
    with zipfile.ZipFile(tmp, "w") as zf:
        zf.writestr("ABOUT.txt", ABOUT, compress_type=zipfile.ZIP_DEFLATED)
        for f in files:
            method = zipfile.ZIP_STORED if f.suffix.lower() in (".jpg", ".jpeg", ".png") else zipfile.ZIP_DEFLATED
            zf.write(f, f.relative_to(DATA_DIR).as_posix(), compress_type=method)
    os.replace(tmp, out)
    images = sum(1 for f in files if f.suffix == ".jpg")
    return {"zip": str(out), "files": len(files) + 1, "images": images, "gb": round(out.stat().st_size / 1e9, 2)}
