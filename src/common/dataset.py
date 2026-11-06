"""The dataset as seen by later stages: samples and their images.

An image is named by its `image_id`, "<sample id>/<key>", where the key is "original" or an edit key ("01", "02",
...); its file is data/images/<image_id>.jpg. Edit keys are the ones used by the ranking, so results of every stage
can be joined on image_id. The samples are those of data/pairs.jsonl (every sample with a final ranking).
"""

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from ..paths import FINAL_PAIRS, PROVENANCE, SAMPLES_DIR
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
    """Samples as {"id", "request", "original", "edits": [{"image"}], "pairs": [(key a, key b, p)]}, edits in key
    order (never in ranked order, so no stage sees the ranking by accident)."""
    samples = {}
    for p in iter_jsonl(FINAL_PAIRS):
        sid, a = p["a"].split("/")
        b = p["b"].split("/")[1]
        s = samples.setdefault(sid, {"id": sid, "request": p["request"], "keys": set(), "pairs": []})
        s["keys"].update((a, b))
        s["pairs"].append((a, b, p["p"]))
    for s in samples.values():
        keys = sorted(s.pop("keys"))
        yield {**s, "original": "original.jpg", "edits": [{"image": f"{k}.jpg"} for k in keys]}


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
    if not PROVENANCE.exists():                       # the final dataset has no VOZ sample
        return set()
    return {p["id"] for p in iter_jsonl(PROVENANCE) if not p.get("trainable", True)}
