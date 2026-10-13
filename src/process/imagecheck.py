"""Read image headers and hashes, with a cache keyed by path, size and mtime (data/index/image_cache.json)."""

import hashlib
import json
from pathlib import Path

from PIL import Image

from ..paths import INDEX_DIR

Image.MAX_IMAGE_PIXELS = None  # headers only; never decodes full pixels

CACHE_PATH = INDEX_DIR / "image_cache.json"


class ImageChecker:
    def __init__(self):
        self.cache: dict = json.loads(CACHE_PATH.read_text()) if CACHE_PATH.exists() else {}

    def info(self, path: Path) -> dict:
        """{'readable', 'format', 'width', 'height', 'sha256', 'bytes'}; readable=False if missing or unreadable."""
        if not path.exists():
            return {"readable": False, "reason": "missing_file"}
        st = path.stat()
        key = f"{path}|{st.st_size}|{int(st.st_mtime)}"
        if key in self.cache:
            return self.cache[key]
        data = path.read_bytes()
        out = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
        try:
            with Image.open(path) as im:
                out.update(readable=True, format=im.format, width=im.width, height=im.height)
        except Exception:
            out.update(readable=False, reason="unreadable")
        self.cache[key] = out
        return out

    def save(self) -> None:
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        CACHE_PATH.write_text(json.dumps(self.cache))
