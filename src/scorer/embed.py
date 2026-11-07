"""Embed every image and text of the ranked samples with the frozen backbones, once, so that every scorer head and
similarity baseline is trained and evaluated from the cache without a GPU.

    work/scorer/embeddings/<backbone>/images[.part-I-of-N].npy + .json   one row per image_id
                                      texts-instruction.npy + .json      one row per sample ID
                                      texts-request.npy + .json
                                      texts.json                         digest of the texts embedded
                                      meta[.part-I-of-N].json            model, device, time per image

Speed: every image is decoded once for all backbones (JPEG draft mode, EXIF orientation), padded to a square on
grey and resized to SQUARE_SIZE px in DataLoader workers; each backbone then resizes and normalises the uint8
batch on the GPU (backbones/base.py). All backbones sit on one GPU together (about 7 GB in float16), and
`embed_on_gpus` runs one process per GPU, each on its own shard of the images (`shard=(i, n)`); readers load the
part files as one table. A backbone whose embeddings already cover the images in scope is skipped, so cached
embeddings copied from an earlier run are reused; texts are embedded again when they changed (texts.json).
"""

import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np
from PIL import Image, ImageOps

from ..common.dataset import images_of, iter_samples, text_of
from ..common.device import check_heavy_run, pick_device
from ..common.gpus import run_shards
from ..common.io import write_json
from ..paths import REPO_ROOT, SCORER_DIR
from .backbones import BACKBONES, get_backbone
from .data import load_pairs

SQUARE_SIZE = 448               # above every backbone's input size (SigLIP: 384)
GREY = (127, 127, 127)
TEXT_KINDS = ("instruction", "request")
BATCH_SIZE = 64


def embedding_dir(backbone_name: str):
    return SCORER_DIR / "embeddings" / backbone_name


def part_suffix(shard: tuple[int, int] | None) -> str:
    return f".part-{shard[0]}-of-{shard[1]}" if shard else ""


def texts_digests(samples: list[dict]) -> dict[str, str]:
    """A digest of the sample IDs and texts of each text kind: cached text embeddings are used only when the
    texts are the same (an older dataset version without instructions falls back to the requests)."""
    return {kind: hashlib.sha1("\n".join(f"{sample['id']}\t{text_of(sample, kind)}" for sample in samples)
                               .encode()).hexdigest() for kind in TEXT_KINDS}


def stored_digests(backbone_name: str) -> dict | None:
    path = embedding_dir(backbone_name) / "texts.json"
    return json.loads(path.read_text()) if path.exists() else None


def save_embeddings(backbone_name: str, kind: str, ids: list[str], vectors: np.ndarray) -> None:
    folder = embedding_dir(backbone_name)
    folder.mkdir(parents=True, exist_ok=True)
    temporary = folder / f".{kind}.npy.tmp"
    with open(temporary, "wb") as handle:
        np.save(handle, np.asarray(vectors, dtype=np.float16))
    os.replace(temporary, folder / f"{kind}.npy")
    write_json(folder / f"{kind}.json", ids)


def load_embeddings(backbone_name: str, kind: str) -> dict[str, np.ndarray]:
    """image_id (kind "images") or sample ID (kind "texts-<text kind>") -> float32 vector, from the file of the
    kind and its shard files; empty if not embedded."""
    folder = embedding_dir(backbone_name)
    table = {}
    for vectors_path in sorted(folder.glob(f"{kind}.npy")) + sorted(folder.glob(f"{kind}.part-*.npy")):
        ids = json.loads(vectors_path.with_suffix(".json").read_text(encoding="utf-8"))
        table.update(zip(ids, np.load(vectors_path).astype(np.float32)))
    return table


def scope_samples(version: str = "all", limit: int | None = None) -> list[dict]:
    """Samples that have preference pairs, in dataset order."""
    ids = {pair["id"] for pair in load_pairs(version)}
    samples = [sample for sample in iter_samples() if sample["id"] in ids]
    return samples[:limit] if limit is not None else samples


def square_image(path, size: int = SQUARE_SIZE) -> np.ndarray:
    """The image as RGB uint8 (size, size, 3): whole, aspect kept, padded on grey."""
    with Image.open(path) as image:
        image.draft("RGB", (size, size))                 # JPEG: decode at a reduced scale, never below size
        image = ImageOps.exif_transpose(image).convert("RGB")
    image.thumbnail((size, size), Image.Resampling.BICUBIC)
    canvas = Image.new("RGB", (size, size), GREY)
    canvas.paste(image, ((size - image.width) // 2, (size - image.height) // 2))
    return np.array(canvas)


class SquareImages:
    """Map-style dataset for torch's DataLoader: decoding runs in its workers."""

    def __init__(self, paths):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        import torch
        return torch.from_numpy(square_image(self.paths[index])).permute(2, 0, 1)


def embed(backbone_names: list[str] | None = None, version: str = "all", limit: int | None = None,
          device: str | None = None, allow_cpu: bool = False, workers: int | None = None,
          shard: tuple[int, int] | None = None, batch_size: int = BATCH_SIZE) -> dict:
    """Embed the images in scope (one shard of them) with every backbone that does not cover them yet, and the
    texts of the samples (unsharded run or shard 0). `limit` caps the number of samples."""
    import torch
    from torch.utils.data import DataLoader

    device = pick_device(device)
    backbone_names = backbone_names or list(BACKBONES)
    samples = scope_samples(version, limit)
    refs = [ref for sample in samples for ref in images_of(sample)]
    if shard:
        refs = refs[shard[0]::shard[1]]
    wanted = {ref.image_id for ref in refs}
    todo = [name for name in backbone_names if not wanted <= set(load_embeddings(name, "images"))]
    with_texts = not shard or shard[0] == 0
    digests = texts_digests(samples)
    text_todo = [name for name in backbone_names if with_texts and get_backbone(name).has_text
                 and stored_digests(name) != digests]
    summary = {"images": len(refs), "samples": len(samples), "cached": sorted(set(backbone_names) - set(todo)),
               "shard": "%d/%d" % shard if shard else None}
    if not todo and not text_todo:
        return {**summary, "embedded": []}
    check_heavy_run(device, len(refs) if todo else len(samples), allow_cpu)
    backbones = {name: get_backbone(name) for name in sorted(set(todo) | set(text_todo))}
    started = time.monotonic()
    for name, backbone in backbones.items():
        backbone.load(device)
    print(f"loaded {', '.join(backbones)} on {device} in {time.monotonic() - started:.0f} s", flush=True)

    image_seconds = 0.0
    if todo:
        workers = (min(4, os.cpu_count() or 1) if device == "cuda" else 0) if workers is None else workers
        loader = DataLoader(SquareImages([ref.path for ref in refs]), batch_size=batch_size, num_workers=workers,
                            pin_memory=device == "cuda")
        vectors = {name: [] for name in todo}
        started, done = time.monotonic(), 0
        for index, batch in enumerate(loader):
            batch = batch.to(device, non_blocking=True)
            for name in todo:
                vectors[name].append(backbones[name].encode_images(batch))
            done += len(batch)
            if (index + 1) % 20 == 0 or done == len(refs):
                rate = done / max(time.monotonic() - started, 1e-9)
                print(f"embed: {done}/{len(refs)} images ({100 * done // len(refs)}%), {rate:.1f} img/s, "
                      f"about {(len(refs) - done) / rate / 60:.0f} min left", flush=True)
        image_seconds = time.monotonic() - started
        for name in todo:
            save_embeddings(name, f"images{part_suffix(shard)}", [ref.image_id for ref in refs],
                            np.concatenate(vectors[name]))

    for name in text_todo:
        for kind in TEXT_KINDS:
            texts = [text_of(sample, kind) for sample in samples]
            text_vectors = np.concatenate([backbones[name].encode_text_batch(texts[start:start + 256])
                                           for start in range(0, len(texts), 256)])
            save_embeddings(name, f"texts-{kind}", [sample["id"] for sample in samples], text_vectors)
        write_json(embedding_dir(name) / "texts.json", digests)

    gpu = torch.cuda.get_device_name(0) if device == "cuda" else None
    per_image = round(image_seconds / len(refs), 5) if todo and refs else None
    for name in todo:
        write_json(embedding_dir(name) / f"meta{part_suffix(shard)}.json",
                   {"backbone": name, "model_id": backbones[name].model_id, "images": len(refs),
                    "samples": len(samples), "version": version, "device": device, "gpu": gpu,
                    "backbones_together": len(todo), "seconds": round(image_seconds, 1),
                    "seconds_per_image_all_backbones": per_image,
                    "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds")})
    return {**summary, "embedded": todo, "texts": text_todo, "device": device,
            "seconds_per_image_all_backbones": per_image}


def embed_on_gpus(backbone_names: list[str] | None = None, limit: int | None = None,
                  gpus: int | None = None) -> dict:
    """`embed` in one process per GPU, each on its own shard; logs in work/scorer/logs/."""
    command = [sys.executable, "-m", "src.scorer", "embed"]
    command += ["--backbones", ",".join(backbone_names)] if backbone_names else []
    command += ["--limit", str(limit)] if limit else []
    return run_shards(command, "embed", SCORER_DIR / "logs", gpus, cwd=REPO_ROOT)
