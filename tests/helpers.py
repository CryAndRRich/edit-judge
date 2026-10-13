"""A tiny synthetic dataset in a temporary folder, so tests never touch the real data/.
Import this module before any src module: it sets EJ_DATA_DIR and EJ_OUTPUT_DIR."""

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
os.environ["EJ_OUTPUT_DIR"] = str(TMP / "out")


def make_dataset(n_samples: int = 2, n_edits: int = 2) -> list[dict]:
    rows = []
    for s in range(n_samples):
        sid = f"s{s:03d}"
        folder = TMP / "samples" / sid
        (folder / "edits").mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (64, 48), (s * 40, 0, 0)).save(folder / "original.jpg")
        edits = []
        for e in range(1, n_edits + 1):
            Image.new("RGB", (64, 48), (s * 40, e * 60, 0)).save(folder / "edits" / f"{e:02d}.jpg")
            edits.append({"image": f"edits/{e:02d}.jpg", "note": None})
        meta = {"id": sid, "request": "test", "original": "original.jpg", "edits": edits}
        (folder / "sample.json").write_text(json.dumps(meta))
        rows.append({**meta, "dir": f"samples/{sid}"})
    with open(TMP / "samples.jsonl", "w") as fh:
        fh.writelines(json.dumps(r) + "\n" for r in rows)
    return rows
