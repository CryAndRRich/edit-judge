"""The interface every detector implements, and the runner that scores dataset images with it.

A detector returns, for each image, the probability that the image is positive for its task (sensitive content,
or a watermark). Scores go to data/filters/<task>/<name>.jsonl as {"image_id", "score"}, written after every
batch, so a run can be stopped and started again. data/filters/<task>/<name>.meta.json records the device and
the time per image, used to compare speed.
"""

import json
import time
from datetime import datetime, timezone

from PIL import Image

from ..common.dataset import iter_images
from ..common.device import check_heavy_run, pick_device
from ..common.io import append_jsonl, iter_jsonl, write_json
from ..paths import FILTERS_DIR


class Detector:
    task: str = ""          # "nsfw" or "watermark"
    name: str = ""          # short unique name; also the name of the score file
    model_id: str = ""      # where the weights come from (Hugging Face ID, URL, package version)
    batch_size: int = 16

    def load(self, device: str) -> None:
        """Load the model onto `device` ("cuda", "mps" or "cpu")."""
        raise NotImplementedError

    def score(self, images: list[Image.Image]) -> list[float]:
        """Probability in [0, 1] that each image is positive."""
        raise NotImplementedError


def score_path(task: str, name: str):
    return FILTERS_DIR / task / f"{name}.jsonl"


def run(detector: Detector, scope: str = "all", limit: int | None = None, device: str | None = None,
        allow_cpu: bool = False) -> dict:
    device = pick_device(device)
    out = score_path(detector.task, detector.name)
    done = {r["image_id"] for r in iter_jsonl(out)}
    todo = [ref for ref in iter_images(scope) if ref.image_id not in done]
    if limit is not None:
        todo = todo[:limit]
    check_heavy_run(device, len(todo), allow_cpu)
    if not todo:
        return {"model": detector.name, "scored": 0, "already_done": len(done)}
    detector.load(device)
    started = time.monotonic()
    for i in range(0, len(todo), detector.batch_size):
        batch = todo[i:i + detector.batch_size]
        images = []
        for ref in batch:
            with Image.open(ref.path) as im:
                images.append(im.convert("RGB"))
        scores = detector.score(images)
        append_jsonl(out, ({"image_id": ref.image_id, "score": round(float(s), 5)} for ref, s in zip(batch, scores)))
    seconds = time.monotonic() - started
    meta_path = out.with_suffix(".meta.json")
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {"images": 0, "seconds": 0.0}
    meta.update(model=detector.name, model_id=detector.model_id, task=detector.task, device=device,
                images=meta["images"] + len(todo), seconds=round(meta["seconds"] + seconds, 1),
                updated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"))
    meta["seconds_per_image"] = round(meta["seconds"] / meta["images"], 4)
    write_json(meta_path, meta)
    return {"model": detector.name, "scored": len(todo), "already_done": len(done), "device": device,
            "seconds_per_image": round(seconds / len(todo), 4)}
