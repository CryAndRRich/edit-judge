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
    """Samples as {"id", "request", "instruction", "original", "edits": [{"image"}], "pairs": [(key a, key b, p)]},
    edits in key order (never in ranked order, so no stage sees the ranking by accident). `instruction` is the
    simplified instruction of the request (RealEdit's or the VLM's), None when the dataset has none."""
    samples = {}
    for pair in iter_jsonl(FINAL_PAIRS):
        sample_id, first = pair["a"].split("/")
        second = pair["b"].split("/")[1]
        sample = samples.setdefault(sample_id, {"id": sample_id, "request": pair["request"],
                                                "instruction": pair.get("instruction"), "keys": set(), "pairs": []})
        sample["keys"].update((first, second))
        sample["pairs"].append((first, second, pair["p"]))
    for sample in samples.values():
        keys = sorted(sample.pop("keys"))
        yield {**sample, "original": "original.jpg", "edits": [{"image": f"{key}.jpg"} for key in keys]}


def text_of(sample: dict, kind: str = "instruction") -> str:
    """The text a model reads for a sample: its simplified instruction (falling back to the request when the
    dataset has none) or the request as written, whitespace collapsed."""
    if kind not in ("instruction", "request"):
        raise ValueError(f"unknown text kind {kind!r}; expected instruction or request")
    text = (sample.get("instruction") if kind == "instruction" else None) or sample["request"]
    return " ".join(text.split())


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
