import hashlib
import json
import math
import re
from collections import defaultdict
from pathlib import Path

from ...common.io import iter_jsonl, write_json
from ...paths import FINAL_PAIRS, OUTPUT_DIR

REVIEW_DIR = OUTPUT_DIR / "review"


def round_dir(number: int) -> Path:
    if number < 1 or number > 5:
        raise ValueError("review round must be between 1 and 5")
    return REVIEW_DIR / f"round-{number:02d}"


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode()).hexdigest()


def fingerprint(path: Path) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def frozen_json(path: Path, value) -> None:
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != value:
            raise ValueError(f"{path} already exists with different inputs; use a new round")
    else:
        write_json(path, value)


def identity(name: str) -> str:
    value = " ".join(str(name).split()).casefold()
    if not value or any(c in value for c in "/\\\0"):
        raise ValueError(f"invalid reviewer name: {name!r}")
    return value


def safe_key(value: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", value):
        raise ValueError(f"unsafe sample/edit key: {value!r}")
    return value


def dataset(path: Path | None = None) -> tuple[list[dict], dict[str, dict]]:
    path = FINAL_PAIRS if path is None else path
    if not path.is_file():
        raise ValueError(f"missing pairs file: {path}; attach the final Kaggle dataset")
    rows, samples, seen = list(iter_jsonl(path)), {}, set()
    for row in rows:
        sid, a = row["a"].split("/")
        other, b = row["b"].split("/")
        for key in (sid, a, other, b):
            safe_key(key)
        if sid != other or a == b or not math.isfinite(row["p"]) or not 0.5 <= row["p"] <= 1:
            raise ValueError(f"invalid final pair: {row['a']}, {row['b']}")
        pair_key = (sid, *sorted((a, b)))
        if pair_key in seen:
            raise ValueError(f"duplicate pair: {pair_key}")
        seen.add(pair_key)
        sample = samples.setdefault(sid, {"id": sid, "request": row["request"], "keys": set(),
                                          "wins": defaultdict(float), "pairs": []})
        sample["keys"].update((a, b))
        sample["wins"][a] += 0.5 if row["p"] == 0.5 else 1
        sample["wins"][b] += 0.5 if row["p"] == 0.5 else 0
        sample["pairs"].append(row)
    if not rows:
        raise ValueError("empty pairs dataset")
    for sample in samples.values():
        sample["keys"] = sorted(sample["keys"])
        if len(sample["pairs"]) != len(sample["keys"]) * (len(sample["keys"]) - 1) // 2:
            raise ValueError(f"incomplete pair graph: {sample['id']}")
        sample["ranking"] = sorted(sample["keys"], key=lambda key: (-sample["wins"][key], key))
    return rows, samples


def originals() -> dict[str, dict]:
    path = REVIEW_DIR / "originals.jsonl"
    if not path.exists():
        raise ValueError("run `python -m src.rank review-prepare` with the original ranking results first")
    return {row["id"]: row for row in iter_jsonl(path)}


def selected(number: int) -> list[dict]:
    path = round_dir(number) / "selected.jsonl"
    if not path.exists():
        raise ValueError("select samples before making chunks or applying reviews")
    rows = list(iter_jsonl(path))
    inputs = json.loads((path.parent / "selection-inputs.json").read_text(encoding="utf-8"))
    if digest(rows) != inputs["selection_sha256"]:
        raise ValueError("selection changed after it was frozen")
    return rows


def chunk_metadata(number: int) -> tuple[dict, dict]:
    folder = round_dir(number)
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    mapping = json.loads((folder / "chunk-map.json").read_text(encoding="utf-8"))
    inputs = json.loads((folder / "chunks-inputs.json").read_text(encoding="utf-8"))
    if (digest(mapping) != inputs["mapping_sha256"] or digest(manifest) != inputs["manifest_sha256"]
            or digest(selected(number)) != inputs["selected_sha256"]):
        raise ValueError("chunk mapping, assignment or selection changed after packaging")
    return manifest, mapping
