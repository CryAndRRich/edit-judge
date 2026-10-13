"""The dataset as seen by later stages: samples and their images.

An image is named by its `image_id`, "<sample id>/<key>", where the key is "original" or an edit key ("01", "02",
...). Edit keys are the ones used by the ranking, so results of every stage can be joined on image_id.
"""

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from ..paths import PROVENANCE, SAMPLES_DIR, SAMPLES_JSONL
from .io import iter_jsonl


@dataclass(frozen=True)
class ImageRef:
    sample_id: str
    key: str            # "original" or an edit key such as "01"
    path: Path

    @property
    def image_id(self) -> str:
        return f"{self.sample_id}/{self.key}"


def iter_samples() -> Iterator[dict]:
    yield from iter_jsonl(SAMPLES_JSONL)


def images_of(sample: dict) -> list[ImageRef]:
    folder = SAMPLES_DIR / sample["id"]
    refs = [ImageRef(sample["id"], "original", folder / sample["original"])]
    refs += [ImageRef(sample["id"], Path(e["image"]).stem, folder / e["image"]) for e in sample["edits"]]
    return refs


def iter_images(scope: str = "all") -> Iterator[ImageRef]:
    """Every image of the dataset; scope is "all", "originals" or "edits"."""
    if scope not in ("all", "originals", "edits"):
        raise ValueError(f"unknown scope {scope!r}")
    for sample in iter_samples():
        for ref in images_of(sample):
            if scope == "all" or (scope == "originals") == (ref.key == "original"):
                yield ref


def not_trainable() -> set[str]:
    """Sample IDs whose source forbids model training (VOZ: robots.txt ai-train=no)."""
    return {p["id"] for p in iter_jsonl(PROVENANCE) if not p.get("trainable", True)}
