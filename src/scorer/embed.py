"""Embed the images and requests of the ranked samples with a frozen backbone, once, so that every scorer head and
baseline is trained and evaluated from the cache without a GPU.

    data/scorer/embeddings/<backbone>/images.npy + images.json   one row per image_id (originals and edits)
                                      texts.npy  + texts.json    one row per sample ID (the request as written)
                                      meta.json                  model, device, time per image

Only samples that have preference pairs are embedded (ranked and trainable). Vectors are L2-normalised and
stored as float16. A run embeds everything again (a few minutes per backbone on a T4); it is not resumable.
"""

import json
import os
import time
from datetime import datetime, timezone

import numpy as np
from PIL import Image

from ..common.dataset import images_of, iter_samples
from ..common.device import check_heavy_run, pick_device
from ..common.io import write_json
from ..paths import SCORER_DIR
from .data import load_pairs

DECODE_SIZE = (512, 512)        # JPEG draft mode decodes at a reduced scale, never below this size


def embedding_dir(backbone_name: str):
    return SCORER_DIR / "embeddings" / backbone_name


def save_embeddings(backbone_name: str, kind: str, ids: list[str], vectors: np.ndarray) -> None:
    folder = embedding_dir(backbone_name)
    folder.mkdir(parents=True, exist_ok=True)
    tmp = folder / f".{kind}.npy.tmp"
    with open(tmp, "wb") as fh:
        np.save(fh, np.asarray(vectors, dtype=np.float16))
    os.replace(tmp, folder / f"{kind}.npy")
    write_json(folder / f"{kind}.json", ids)


def load_embeddings(backbone_name: str, kind: str) -> dict[str, np.ndarray]:
    """image_id (kind "images") or sample ID (kind "texts") -> float32 vector; empty if not embedded."""
    folder = embedding_dir(backbone_name)
    if not (folder / f"{kind}.npy").exists():
        return {}
    ids = json.loads((folder / f"{kind}.json").read_text(encoding="utf-8"))
    vectors = np.load(folder / f"{kind}.npy").astype(np.float32)
    return dict(zip(ids, vectors))


def scope_samples(version: str = "v1", limit: int | None = None) -> list[dict]:
    """Samples that have preference pairs, in dataset order."""
    ids = {p["id"] for p in load_pairs(version)}
    samples = [s for s in iter_samples() if s["id"] in ids]
    return samples[:limit] if limit is not None else samples


def request_text(sample: dict) -> str:
    return " ".join(sample["request"].split())[:2000]


class _Images:
    """Map-style dataset for torch's DataLoader: decoding and preprocessing run in its workers."""

    def __init__(self, paths, backbone):
        self.paths, self.backbone = paths, backbone

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        with Image.open(self.paths[i]) as im:
            im.draft("RGB", DECODE_SIZE)
            return self.backbone.preprocess(im.convert("RGB"))


def embed(backbone, version: str = "v1", limit: int | None = None, device: str | None = None,
          allow_cpu: bool = False, workers: int | None = None) -> dict:
    """Embed the images (and requests) of the samples in scope. `limit` caps the number of samples."""
    from torch.utils.data import DataLoader

    device = pick_device(device)
    samples = scope_samples(version, limit)
    refs = [ref for s in samples for ref in images_of(s)]
    check_heavy_run(device, len(refs), allow_cpu)
    if not refs:
        return {"backbone": backbone.name, "images": 0}
    backbone.load(device)
    workers = (min(4, os.cpu_count() or 1) if device == "cuda" else 0) if workers is None else workers
    loader = DataLoader(_Images([r.path for r in refs], backbone), batch_size=backbone.batch_size,
                        num_workers=workers)
    started = time.monotonic()
    vectors = np.concatenate([backbone.encode_images(pixels) for pixels in loader])
    image_seconds = time.monotonic() - started
    save_embeddings(backbone.name, "images", [r.image_id for r in refs], vectors)
    result = {"backbone": backbone.name, "samples": len(samples), "images": len(refs), "dim": int(vectors.shape[1])}
    if backbone.has_text:
        texts = [request_text(s) for s in samples]
        text_vectors = np.concatenate([backbone.encode_texts(texts[i:i + 256]) for i in range(0, len(texts), 256)])
        save_embeddings(backbone.name, "texts", [s["id"] for s in samples], text_vectors)
        result["texts"] = len(texts)
    import torch
    gpu = torch.cuda.get_device_name(0) if device == "cuda" else None
    write_json(embedding_dir(backbone.name) / "meta.json",
               {**result, "model_id": backbone.model_id, "version": version, "device": device, "gpu": gpu,
                "seconds": round(time.monotonic() - started, 1),
                "seconds_per_image": round(image_seconds / len(refs), 4),
                "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds")})
    return {**result, "device": device, "seconds_per_image": round(image_seconds / len(refs), 4)}
