"""Run a VLM client over the requests: one call per request (the request text, its original image, and the
prompt), answers parsed and appended to data/instructions/<client>/<prompt>.jsonl after every batch, so a run can
be stopped and started again. <prompt>.meta.json records the device and the time per request."""

import json
import random
import time
from datetime import datetime, timezone

from ..common.dataset import iter_samples
from ..common.device import check_heavy_run, pick_device
from ..common.io import append_jsonl, iter_jsonl, write_json
from ..paths import INSTRUCTIONS_DIR, SAMPLES_DIR
from . import prompts
from .parse import parse_answer


def output_path(client_name: str, prompt_id: str):
    return INSTRUCTIONS_DIR / client_name / f"{prompt_id}.jsonl"


def select(samples: list[dict], sample: int | None, seed: int) -> list[dict]:
    """All samples, or a fixed random subset of `sample` of them (the same subset for every client and prompt,
    so prompt versions and models are compared on the same requests)."""
    if sample is None:
        return samples
    ids = sorted(s["id"] for s in samples)
    keep = set(random.Random(seed).sample(ids, min(sample, len(ids))))
    return [s for s in samples if s["id"] in keep]


def run(client, prompt_name: str = "analyze", version: int = 1, limit: int | None = None,
        sample: int | None = None, seed: int = 2026, device: str | None = None, allow_cpu: bool = False,
        max_tokens: int = 256) -> dict:
    device = pick_device(device)
    template, prompt_id = prompts.load(prompt_name, version), f"{prompt_name}_v{version}"
    out = output_path(client.name, prompt_id)
    done = {r["sample_id"] for r in iter_jsonl(out)}
    todo = [s for s in select(list(iter_samples()), sample, seed) if s["id"] not in done]
    if limit is not None:
        todo = todo[:limit]
    check_heavy_run(device, len(todo), allow_cpu)
    if not todo:
        return {"client": client.name, "prompt": prompt_id, "analysed": 0, "already_done": len(done)}
    client.load(device)
    started, failed = time.monotonic(), 0
    for i in range(0, len(todo), client.batch_size):
        batch = todo[i:i + client.batch_size]
        answers = client.generate_batch([prompts.fill(template, s["request"]) for s in batch],
                                        [SAMPLES_DIR / s["id"] / s["original"] for s in batch], max_tokens)
        rows = [parse_answer(a, s["id"], client.name, prompt_id) for a, s in zip(answers, batch)]
        failed += sum(bool(r.problems) for r in rows)
        append_jsonl(out, (r.to_dict() for r in rows))
    seconds = time.monotonic() - started
    meta_path = out.with_suffix(".meta.json")
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {"requests": 0, "seconds": 0.0}
    meta.update(client=client.name, model_id=client.model_id, prompt=prompt_id, device=device,
                requests=meta["requests"] + len(todo), seconds=round(meta["seconds"] + seconds, 1),
                updated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"))
    meta["seconds_per_request"] = round(meta["seconds"] / meta["requests"], 3)
    write_json(meta_path, meta)
    return {"client": client.name, "prompt": prompt_id, "analysed": len(todo), "with_problems": failed,
            "already_done": len(done), "device": device, "seconds_per_request": round(seconds / len(todo), 3)}
