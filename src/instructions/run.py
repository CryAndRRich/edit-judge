"""Run a VLM client over the requests: one call per request (the request text, its original image, and the
prompt), answers parsed and appended to data/instructions/<client>/<prompt>.jsonl after every batch, so a run can
be stopped and started again. <prompt>.meta.json records the device and the time per request.

Long runs can be split:
- shards: `shard=(i, n)` takes every n-th request, so n processes (one per GPU) share the work; shard i writes
  <prompt>.part-<i>-of-<n>.jsonl. Requests already in any result file of the prompt are skipped.
- sessions: `max_hours` stops before the next batch once the time is used, so a Kaggle session ends cleanly and
  keeps its output; the next session copies the results back and goes on with the requests that are left.
"""

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

BROKEN = ("no JSON object", "invalid JSON")     # answers asked again on their own (e.g. "!!!!" from a batch)


def output_path(client_name: str, prompt_id: str, shard: tuple[int, int] | None = None):
    name = f"{prompt_id}.part-{shard[0]}-of-{shard[1]}" if shard else prompt_id
    return INSTRUCTIONS_DIR / client_name / f"{name}.jsonl"


def result_files(client_name: str, prompt_id: str) -> list:
    """Every result file of a run: the main file and the shard files."""
    folder = INSTRUCTIONS_DIR / client_name
    return sorted(folder.glob(f"{prompt_id}.jsonl")) + sorted(folder.glob(f"{prompt_id}.part-*.jsonl"))


def select(samples: list[dict], sample: int | None, seed: int) -> list[dict]:
    """All samples, or a fixed random subset of `sample` of them (the same subset for every client and prompt,
    so prompt versions and models are compared on the same requests)."""
    if sample is None:
        return samples
    ids = sorted(s["id"] for s in samples)
    keep = set(random.Random(seed).sample(ids, min(sample, len(ids))))
    return [s for s in samples if s["id"] in keep]


def gpu_info() -> dict:
    try:
        import torch
        if torch.cuda.is_available():
            return {"gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}
    except ImportError:
        pass
    return {}


def run(client, prompt_name: str = "analyze", version: int = 1, limit: int | None = None,
        sample: int | None = None, seed: int = 2026, device: str | None = None, allow_cpu: bool = False,
        max_tokens: int = 256, shard: tuple[int, int] | None = None, max_hours: float | None = None) -> dict:
    device = pick_device(device)
    template, prompt_id = prompts.load(prompt_name, version), f"{prompt_name}_v{version}"
    out = output_path(client.name, prompt_id, shard)
    done = {r["sample_id"] for f in result_files(client.name, prompt_id) for r in iter_jsonl(f)}
    todo = select(list(iter_samples()), sample, seed)
    if shard:
        todo = todo[shard[0]::shard[1]]
    todo = [s for s in todo if s["id"] not in done]
    if limit is not None:
        todo = todo[:limit]
    check_heavy_run(device, len(todo), allow_cpu)
    summary = {"client": client.name, "prompt": prompt_id, "shard": "%d/%d" % shard if shard else None}
    if not todo:
        return {**summary, "analysed": 0, "already_done": len(done), "left": 0}
    client.load(device)
    started, analysed, failed, retried = time.monotonic(), 0, 0, 0
    for i in range(0, len(todo), client.batch_size):
        if max_hours is not None and time.monotonic() - started >= max_hours * 3600:
            break
        batch = todo[i:i + client.batch_size]
        prompts_ = [prompts.fill(template, s["request"]) for s in batch]
        images = [SAMPLES_DIR / s["id"] / s["original"] for s in batch]
        answers = client.generate_batch(prompts_, images, max_tokens)
        rows = []
        for prompt, image, answer, s in zip(prompts_, images, answers, batch):
            row = parse_answer(answer, s["id"], client.name, prompt_id)
            if len(batch) > 1 and any(p in row.problems for p in BROKEN):     # again alone, without padding
                retried += 1
                again = parse_answer(client.generate(prompt, image, max_tokens), s["id"], client.name, prompt_id)
                row = again if len(again.problems) < len(row.problems) else row
            rows.append(row)
        analysed += len(batch)
        failed += sum(bool(r.problems) for r in rows)
        append_jsonl(out, (r.to_dict() for r in rows))
    seconds = time.monotonic() - started
    meta_path = out.with_suffix(".meta.json")
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {"requests": 0, "seconds": 0.0}
    meta.update(client=client.name, model_id=client.model_id, prompt=prompt_id, device=device, **gpu_info(),
                batch_size=client.batch_size, requests=meta["requests"] + analysed,
                seconds=round(meta["seconds"] + seconds, 1),
                updated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"))
    meta["seconds_per_request"] = round(meta["seconds"] / max(meta["requests"], 1), 3)
    write_json(meta_path, meta)
    return {**summary, "analysed": analysed, "with_problems": failed, "retried_alone": retried,
            "already_done": len(done), "left": len(todo) - analysed, "device": device,
            "seconds_per_request": round(seconds / max(analysed, 1), 3)}
