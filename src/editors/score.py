"""The scorer's opinion of every output: each model's edit is scored together with the human edits of its sample.

The scorer (a trained run of src/scorer, `mlp-siglip` by default: the run selected on dev) gives one score per edit
of a request, comparable only within that request. For every sample of the task, the original, every human edit
and every model output with status ok are encoded once by the run's backbones (padded to a square, as in
training), and every edit gets the run's score: the features of scorer/train.py:backbone_blocks, standardised, then
the mean of the seed heads (as scorer/latency.py:TrainedRun.score, which handles two edits; here any number).

Caveat: the scorer learned from human (Photoshop) edits; AI outputs are another domain, so its scores are a first
opinion to be checked by eye on the contact sheets of report.py, not a verdict.

Output: work/editors/<task>/scores.jsonl, one row per edit: {"id", "edit" ("human:<key>" or "model:<name>"),
"score", "top_human" (the key of the best human edit in the final ranking)}.
"""

import numpy as np

from ..common.dataset import iter_samples, text_of
from ..common.device import check_heavy_run, pick_device
from ..common.io import iter_jsonl, write_jsonl
from ..paths import EDITORS_DIR, SAMPLES_DIR
from . import editors_of
from .base import load_runs, output_path
from .samples import load_task_samples

DEFAULT_RUN = "mlp-siglip"


def scores_path(task: str):
    return EDITORS_DIR / task / "scores.jsonl"


def load_scores(task: str) -> dict[str, dict[str, float]]:
    """Sample ID -> {edit: score}."""
    scores = {}
    for row in iter_jsonl(scores_path(task)):
        scores.setdefault(row["id"], {})[row["edit"]] = row["score"]
    return scores


def top_human(pairs: list[tuple[str, str, float]]) -> str:
    """The edit that wins most pairs of the final ranking (its first place); `a` of every pair is the winner."""
    wins = {}
    for a, b, _ in pairs:
        wins[a] = wins.get(a, 0) + 1
        wins.setdefault(b, 0)
    return max(sorted(wins), key=wins.get)


def head_scores(run, encoded: dict) -> list[float]:
    """The run's score of every edit (rows 1.. of each backbone's image vectors; row 0 is the original)."""
    import torch

    from ..scorer.train import backbone_blocks

    blocks = []
    for name in run.cfg["backbones"]:
        images, text = encoded[name]
        count = len(images) - 1
        blocks += backbone_blocks(images[1:], images[[0] * count], None if text is None else text[[0] * count],
                                  run.cfg["inputs"])
    features = torch.from_numpy(np.concatenate(blocks, axis=1).astype(np.float32)).to(run.device)
    features = (features - run.feature_mean) / run.feature_std
    with torch.no_grad():
        scores = torch.stack([(model(features)[0] - mean) / std for model, mean, std in run.heads]).mean(0)
    return scores.cpu().tolist()


def score(task: str, run_name: str = DEFAULT_RUN, device: str | None = None, allow_cpu: bool = False) -> dict:
    import torch

    from ..scorer.backbones import get_backbone
    from ..scorer.embed import square_image
    from ..scorer.latency import TrainedRun

    samples = load_task_samples(task)
    dataset = {sample["id"]: sample for sample in iter_samples()}
    runs = {name: load_runs(name) for name in editors_of(task)}
    device = pick_device(device)
    check_heavy_run(device, len(samples), allow_cpu)
    run = TrainedRun(run_name, device)
    backbones = {name: get_backbone(name) for name in run.cfg["backbones"]}
    for backbone in backbones.values():
        backbone.load(device)
    rows = []
    for sample in samples:
        data = dataset.get(sample["id"])
        if data is None:
            continue
        best = top_human(data["pairs"])
        edits = [(f"human:{e['image'].rsplit('.', 1)[0]}", SAMPLES_DIR / sample["id"] / e["image"])
                 for e in data["edits"]]
        edits += [(f"model:{name}", output_path(name, sample["id"])) for name, model_runs in runs.items()
                  if model_runs.get(sample["id"], {}).get("status") == "ok"]
        paths = [SAMPLES_DIR / sample["id"] / "original.jpg"] + [path for _, path in edits]
        batch = torch.stack([torch.from_numpy(square_image(path)).permute(2, 0, 1) for path in paths]).to(device)
        text = text_of(data, run.cfg["text"])
        encoded = {name: (backbone.encode_images(batch),
                          backbone.encode_text_batch([text]) if backbone.has_text else None)
                   for name, backbone in backbones.items()}
        for (edit, _), value in zip(edits, head_scores(run, encoded)):
            rows.append({"id": sample["id"], "edit": edit, "score": round(float(value), 4), "top_human": best})
    write_jsonl(scores_path(task), rows)
    return {"task": task, "run": run_name, "samples": len({r["id"] for r in rows}), "edits": len(rows),
            "model_outputs": sum(r["edit"].startswith("model:") for r in rows)}
