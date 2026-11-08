"""Reward models and VLM judges compared with the trained scorer (baselines that need a GPU). Each scores every edit
of the test split from (original image, text, edited image); see base.py for the runner. Register each as
name -> "module:Class"; modules are imported only when a model is used.

`run_all` runs several of them one after the other, each in its own process and Python environment (their
libraries pin conflicting versions), sharing one time budget, and prints their output as it comes. With
`shard=(i, n)` every model scores only its i-th of n parts of the edits, so a model too slow for one Kaggle
session is split between sessions (team members) that run side by side.
"""

import importlib
import shutil
import subprocess
import sys
import time
from pathlib import Path

from ...common.envs import ensure_env
from ...common.gpus import log_progress, log_tail
from ...common.io import write_json
from ...paths import REPO_ROOT, SCORER_DIR

REWARD_MODELS: dict[str, str] = {
    "imagereward": ".imagereward:ImageRewardModel",   # BLIP reward model for text-to-image (NeurIPS 2023)
    "editreward": ".editreward:EditReward",           # Qwen2.5-VL-7B reward model for edits (ICLR 2026)
    "qwen-judge": ".vlm_judge:QwenJudge",             # Qwen2.5-VL-7B zero-shot, VIEScore prompts
    "editscore": ".vlm_judge:EditScoreJudge",         # the same with the EditScore-7B LoRA (ICLR 2026)
}


def get_reward_model(name: str):
    target = REWARD_MODELS.get(name)
    if target is None:
        raise SystemExit(f"no reward model {name!r}; registered: {sorted(REWARD_MODELS)}")
    module, class_name = target.split(":")
    return getattr(importlib.import_module(module, __name__), class_name)()


def stream(command: list[str]) -> int:
    """Run a command, printing its output line by line as it comes; returns its exit code."""
    process = subprocess.Popen(command, cwd=REPO_ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                               bufsize=1)
    for line in process.stdout:
        print(line, end="", flush=True)
    return process.wait()


def run_all(models: list[str], split: str = "test", text_kind: str = "instruction", max_hours: float = 10.0,
            limit: int | None = None, shard: tuple[int, int] | None = None) -> dict:
    """Each model in its own environment and process, in order, until `max_hours` have passed. The summary
    (work/scorer/baselines/summary-<models>[.part-I-of-N].json) says, per model, how many edits of this part are
    scored and how many are left."""
    from ..embed import part_suffix
    from .base import edit_items, scored_ids

    started = time.monotonic()
    logs = SCORER_DIR / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    exit_codes = {}
    for name in models:
        hours_left = max_hours - (time.monotonic() - started) / 3600
        if hours_left < 0.05:
            log_progress(f"no time left for {name}")
            break
        model = get_reward_model(name)
        log_progress(f"environment for {name} (pip install, a few minutes the first time; log: env-{name}.log)")
        env_log = logs / f"env-{name}.log"
        try:
            with open(env_log, "a") as log:
                python = ensure_env(model.install, log, model.isolated)
        except subprocess.CalledProcessError as error:
            log_progress(f"FAILED environment for {name} ({' '.join(map(str, error.cmd))[:200]}), end of the log:")
            print(log_tail(env_log, 25), flush=True)
            exit_codes[name] = "environment failed"
            continue
        free = shutil.disk_usage("/tmp" if Path("/tmp").exists() else REPO_ROOT).free / 1e9
        log_progress(f"start {name} ({hours_left:.2f} h left, {free:.0f} GB free on disk)")
        command = [python, "-u", "-m", "src.scorer", "reward", "--model", name, "--split", split, "--text", text_kind,
                   "--max-hours", f"{hours_left:.3f}"] + (["--limit", str(limit)] if limit else [])
        command += ["--shard", f"{shard[0]}/{shard[1]}"] if shard else []
        exit_codes[name] = stream(command)
        log_progress(f"{'done' if exit_codes[name] == 0 else 'FAILED'} {name}")
    items = edit_items(split, text_kind)
    wanted = {item.image_id for item in (items[shard[0]::shard[1]] if shard else items)}
    status = {name: {"scored": len(scored_ids(name) & wanted), "left": len(wanted - scored_ids(name)),
                     "exit_code": exit_codes.get(name)} for name in models}
    summary = {"split": split, "text": text_kind, "shard": "%d/%d" % shard if shard else None,
               "edits": len(wanted), "models": status,
               "complete": all(entry["left"] == 0 for entry in status.values()),
               "hours": round((time.monotonic() - started) / 3600, 2)}
    write_json(SCORER_DIR / "baselines" / f"summary-{'-'.join(models)}{part_suffix(shard)}.json", summary)
    return summary
