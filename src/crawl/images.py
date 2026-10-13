"""Download one image and describe it with its format, size and hash.

Every image format is kept as downloaded (JPEG, PNG, WEBP, GIF, TIFF, BMP, AVIF, HEIC, ...).
Nothing is re-encoded here; `python -m src.process normalize` converts the dataset images later.
"""

import hashlib
import io
import mimetypes
from pathlib import Path

from PIL import Image

from .config import DATA_DIR
from .net import PoliteSession

MIN_BYTES = 10_000          # smaller files are usually deleted-image placeholders or thumbnails [Ref: PS-Battles]

# We only read image headers here, never decode full pixels, so large images are safe to open.
Image.MAX_IMAGE_PIXELS = None


def classify(data: bytes, ctype: str = "") -> dict:
    """Describe image bytes: sha256, size, format, dimensions and a status.

    status: ok              readable by Pillow, any format
            ok_unverified   served as an image but Pillow cannot read it (e.g. HEIC); kept as is
            too_small       under MIN_BYTES, usually a deleted-image placeholder
    """
    meta = {"filesize": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    try:
        with Image.open(io.BytesIO(data)) as im:
            meta.update(format=im.format, width=im.width, height=im.height)
        meta["status"] = "too_small" if len(data) < MIN_BYTES else "ok"
    except Exception:
        subtype = ctype.split(";")[0].split("/")[-1].strip().upper() if ctype.startswith("image/") else None
        meta.update(format=subtype or None, width=None, height=None)
        meta["status"] = "too_small" if len(data) < MIN_BYTES else "ok_unverified"
    return meta


def extension(fmt: str | None, ctype: str = "") -> str:
    if fmt:
        return {"JPEG": "jpg", "MPO": "jpg", "TIFF": "tif", "SVG+XML": "svg"}.get(fmt.upper(), fmt.lower())
    return (mimetypes.guess_extension(ctype.split(";")[0].strip()) or ".bin").lstrip(".")


def fetch_image(session: PoliteSession, url: str, dest_stem: Path) -> dict:
    """Download `url` to `dest_stem` + extension. Returns metadata including a `status`."""
    resp = session.get(url)
    if resp.status_code != 200:
        return {"status": f"http_{resp.status_code}", "downloaded_url": url}
    ctype = resp.headers.get("Content-Type", "")
    if not ctype.startswith("image/"):
        return {"status": "not_an_image", "downloaded_url": url}
    data = resp.content
    meta = {"downloaded_url": resp.url, **classify(data, ctype)}
    path = dest_stem.with_suffix("." + extension(meta["format"], ctype))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    meta["local_path"] = str(path.relative_to(DATA_DIR))  # relative to data/, so the data works on every machine
    return meta
