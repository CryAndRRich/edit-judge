"""A tiny synthetic dataset in a temporary folder, so tests never touch the real data/.
Import this module before any src module: it sets EJ_DATA_DIR, EJ_WORK_DIR and EJ_OUTPUT_DIR (data and work share
one temporary folder)."""

import atexit
import json
import os
import shutil
import tempfile
from pathlib import Path

from PIL import Image

TMP = Path(tempfile.mkdtemp(prefix="editjudge-test-"))
atexit.register(shutil.rmtree, TMP, ignore_errors=True)
os.environ["EJ_DATA_DIR"] = str(TMP)
os.environ["EJ_WORK_DIR"] = str(TMP)
os.environ["EJ_OUTPUT_DIR"] = str(TMP / "out")


def make_dataset(n_samples: int = 2, n_edits: int = 2) -> list[dict]:
    """Final dataset (data/images/, data/pairs.jsonl: edits ranked in key order, p = 0.7) and the built sample list
    (work/samples.jsonl). Data and work share TMP."""
    rows, pairs = [], []
    for s in range(n_samples):
        sid = f"s{s:03d}"
        folder = TMP / "images" / sid
        folder.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (64, 48), (s * 40, 0, 0)).save(folder / "original.jpg")
        keys = [f"{e:02d}" for e in range(1, n_edits + 1)]
        for e, k in enumerate(keys, 1):
            Image.new("RGB", (64, 48), (s * 40, e * 60, 0)).save(folder / f"{k}.jpg")
        pairs += [{"a": f"{sid}/{a}", "b": f"{sid}/{b}", "p": 0.7, "request": "test"}
                  for i, a in enumerate(keys) for b in keys[i + 1:]]
        rows.append({"id": sid, "request": "test", "original": "original.jpg",
                     "edits": [{"image": f"edits/{k}.jpg", "note": None} for k in keys], "dir": f"samples/{sid}"})
    with open(TMP / "pairs.jsonl", "w") as fh:
        fh.writelines(json.dumps(p) + "\n" for p in pairs)
    with open(TMP / "samples.jsonl", "w") as fh:
        fh.writelines(json.dumps(r) + "\n" for r in rows)
    return rows
