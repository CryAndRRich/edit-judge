"""Pack the final dataset into one zip file, to upload as a private Kaggle dataset.

    python -m src.process pack [--out work/exports/editjudge-data.zip]

The zip holds data/ as it is, with no top folder: images/<sample id>/<key>.jpg and pairs.jsonl (see
src/rank/export.py). On Kaggle set EJ_DATA_DIR to /kaggle/input/<dataset>. Images are stored without compression
(JPEG is already compressed); pairs.jsonl is compressed.
"""

import os
import zipfile
from pathlib import Path

from ..paths import DATA_DIR, FINAL_PAIRS, SAMPLES_DIR, WORK_DIR


def files_to_pack() -> list[Path]:
    return [FINAL_PAIRS] + sorted(p for p in SAMPLES_DIR.rglob("*.jpg") if p.is_file())


def pack(out: Path | None = None) -> dict:
    out = out or WORK_DIR / "exports" / "editjudge-data.zip"
    out.parent.mkdir(parents=True, exist_ok=True)
    files = files_to_pack()
    tmp = out.with_name(f".{out.name}.tmp")
    with zipfile.ZipFile(tmp, "w") as zf:
        for f in files:
            method = zipfile.ZIP_STORED if f.suffix == ".jpg" else zipfile.ZIP_DEFLATED
            zf.write(f, f.relative_to(DATA_DIR).as_posix(), compress_type=method)
    os.replace(tmp, out)
    images = sum(1 for f in files if f.suffix == ".jpg")
    return {"zip": str(out), "files": len(files), "images": images, "gb": round(out.stat().st_size / 1e9, 2)}
