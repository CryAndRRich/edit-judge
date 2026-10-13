"""Normalize the dataset images in place, so that every image has the same format and size range.

    python -m src.process normalize [--max-side 1024] [--quality 92] [--workers 3] [--dry-run]

Every image of data/samples (originals and edits) becomes:
- its first frame only (GIF, animated WebP);
- rotated by its EXIF orientation;
- converted to sRGB when it embeds a colour profile (ICC), then to 8-bit RGB, with transparent areas
  composited on white;
- at most `--max-side` pixels on its longest side (never enlarged);
- a JPEG of quality `--quality` with no metadata (EXIF, GPS, colour profile, XMP and comments removed).

A file keeps its name and gets the .jpg extension (edits/02.png -> edits/02.jpg); sample.json,
samples.jsonl and the provenance index are updated. Edit keys (the file stems, used by the ranking) do
not change. A JPEG that already meets every rule is left untouched, so the command can be stopped and
run again: it continues where it stopped.
"""

import io
import json
import logging
import math
import os
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from PIL import Image, ImageCms, ImageOps

from ..paths import PROVENANCE, SAMPLES_DIR, SAMPLES_JSONL

log = logging.getLogger(__name__)

Image.MAX_IMAGE_PIXELS = None       # some community images are very large; they are only scaled down
MAX_SIDE = 1024
QUALITY = 92
METADATA = ("exif", "icc_profile", "xmp", "XML:com.adobe.xmp", "photoshop", "comment")
SRGB = ImageCms.createProfile("sRGB")


def is_normalized(path: Path, max_side: int) -> bool:
    if path.suffix != ".jpg":
        return False
    with Image.open(path) as im:
        return (im.format == "JPEG" and im.mode == "RGB" and max(im.size) <= max_side
                and not any(k in im.info for k in METADATA))


def to_rgb(im: Image.Image, icc: bytes | None) -> Image.Image:
    """8-bit sRGB RGB; transparent areas on white."""
    if im.mode in ("I;16", "I;16B", "I;16L", "I", "F"):           # 16-bit and float greyscale
        im = im.convert("I").point(lambda v: v * (1 / 256)).convert("L")
    if icc:
        try:
            out = "RGBA" if im.mode in ("RGBA", "LA", "PA") else "RGB"
            im = ImageCms.profileToProfile(im, ImageCms.ImageCmsProfile(io.BytesIO(icc)), SRGB, outputMode=out)
        except Exception:                                          # unusable profile: keep the pixel values
            pass
    if im.mode in ("RGBA", "LA", "PA") or "transparency" in im.info:
        rgba = im.convert("RGBA")
        im = Image.alpha_composite(Image.new("RGBA", rgba.size, (255, 255, 255, 255)), rgba)
    return im.convert("RGB")


def convert(path: Path, max_side: int, quality: int) -> Path:
    """Write the normalized image next to `path` (as .jpg) and remove `path` if its name changes."""
    with Image.open(path) as im:
        im.seek(0)
        icc = im.info.get("icc_profile")
        scale = max_side / max(im.size)
        if im.format == "JPEG" and scale < 1:                      # decode large JPEGs at a lower scale
            im.draft("RGB", (math.ceil(im.width * scale), math.ceil(im.height * scale)))
        img = to_rgb(ImageOps.exif_transpose(im), icc)
    img.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    img.info = {}                                                  # Pillow would copy a JPEG comment over
    new = path.with_suffix(".jpg")
    tmp = new.with_name(f".{new.name}.tmp")
    img.save(tmp, "JPEG", quality=quality, optimize=True)
    os.replace(tmp, new)
    if new != path:
        path.unlink()
    return new


def normalize_sample(sample_dir: str, max_side: int, quality: int, dry_run: bool) -> dict:
    d = Path(sample_dir)
    meta = json.loads((d / "sample.json").read_text(encoding="utf-8"))
    out = {"id": meta["id"], "renamed": {}, "converted": 0, "kept": 0, "errors": [], "before": Counter(),
           "bytes_before": 0, "bytes_after": 0}
    for rel in [meta["original"], *(e["image"] for e in meta["edits"])]:
        path = d / rel
        if not path.exists() and path.with_suffix(".jpg").exists():  # converted before an interruption
            out["renamed"][rel] = str(Path(rel).with_suffix(".jpg"))
            out["kept"] += 1
            continue
        try:
            size = path.stat().st_size
            if is_normalized(path, max_side):
                out["kept"] += 1
                continue
            with Image.open(path) as im:
                out["before"][im.format or path.suffix] += 1
            out["converted"] += 1
            out["bytes_before"] += size
            if dry_run:
                continue
            new = convert(path, max_side, quality)
            out["bytes_after"] += new.stat().st_size
            if new != path:
                out["renamed"][rel] = str(new.relative_to(d))
        except Exception as exc:
            out["errors"].append(f"{meta['id']}/{rel}: {exc}")
    if out["renamed"] and not dry_run:
        r = out["renamed"]
        meta["original"] = r.get(meta["original"], meta["original"])
        for e in meta["edits"]:
            e["image"] = r.get(e["image"], e["image"])
        tmp = d / ".sample.json.tmp"
        tmp.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, d / "sample.json")
    return out


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def sync_indexes() -> None:
    """Rewrite samples.jsonl and the image paths in the provenance index from every sample.json."""
    with open(SAMPLES_JSONL, encoding="utf-8") as fh:
        ids = [json.loads(line)["id"] for line in fh if line.strip()]
    metas = {sid: json.loads((SAMPLES_DIR / sid / "sample.json").read_text(encoding="utf-8")) for sid in ids}
    _write_jsonl(SAMPLES_JSONL, [{**metas[sid], "dir": f"samples/{sid}"} for sid in ids])
    if PROVENANCE.exists():
        with open(PROVENANCE, encoding="utf-8") as fh:
            prov = [json.loads(line) for line in fh if line.strip()]
        for p in prov:
            if p["id"] in metas:
                by_stem = {Path(e["image"]).stem: e["image"] for e in metas[p["id"]]["edits"]}
                for e in p["edits"]:
                    e["image"] = by_stem.get(Path(e["image"]).stem, e["image"])
        _write_jsonl(PROVENANCE, prov)


def normalize(max_side: int = MAX_SIDE, quality: int = QUALITY, workers: int | None = None,
              dry_run: bool = False) -> dict:
    with open(SAMPLES_JSONL, encoding="utf-8") as fh:
        dirs = [str(SAMPLES_DIR / json.loads(line)["id"]) for line in fh if line.strip()]
    workers = workers or max(1, (os.cpu_count() or 2) - 1)
    total, before, errors = Counter(), Counter(), []
    started = time.monotonic()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        jobs = pool.map(normalize_sample, dirs, [max_side] * len(dirs), [quality] * len(dirs),
                        [dry_run] * len(dirs), chunksize=8)
        for n, r in enumerate(jobs, 1):
            total.update({k: r[k] for k in ("converted", "kept", "bytes_before", "bytes_after")})
            total["renamed"] += len(r["renamed"])
            before.update(r["before"])
            errors += r["errors"]
            if n % 500 == 0 or n == len(dirs):
                rate = n / (time.monotonic() - started)
                log.info("%s/%s samples, %s images converted, about %.0f min left", n, len(dirs),
                         total["converted"], (len(dirs) - n) / rate / 60)
    if not dry_run:
        sync_indexes()
    return {"samples": len(dirs), "images_converted" if not dry_run else "images_to_convert": total["converted"],
            "images_already_normalized": total["kept"], "files_renamed_to_jpg": total["renamed"],
            "formats_converted": dict(before.most_common()),
            "gb_before": round(total["bytes_before"] / 1e9, 2),
            "gb_after": None if dry_run else round(total["bytes_after"] / 1e9, 2),
            "errors": errors, "minutes": round((time.monotonic() - started) / 60, 1),
            "rules": {"max_side": max_side, "quality": quality, "format": "JPEG, sRGB, no metadata"}}
