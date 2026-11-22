"""Scores the outputs of the Enhance benchmark with the repository's reward models (ImageReward, EditReward) on
Kaggle. The input is the output of the kernel stevenpaulli/enhance-models-benchmark (kernel source). Each reward
model scores (original image, instruction, edited image), each in its own Python environment (src/common/envs.py),
as `python -m src.scorer rewards` does for the test split. The unedited input is scored too, as the "do nothing"
baseline (model name "original").

Output: /kaggle/working/scores/<reward model>.jsonl, rows {"image_id": "<model>/<sample id>", "score", ...}.
"""

import argparse
import glob
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_URL = "https://github.com/CryAndRRich/edit-judge.git"
BRANCH = "curr"
CODE = Path("/tmp/ej/edit-judge")
OUT = Path("/kaggle/working/scores")
REWARDS = ["imagereward", "editreward"]
BASELINE = "original"


def find_editors() -> Path:
    for root, _, files in os.walk("/kaggle/input"):
        if "samples.json" in files and root.endswith("work/editors"):
            return Path(root)
    raise SystemExit("no work/editors/samples.json under /kaggle/input: add the benchmark kernel as input")


def build_items(editors: Path) -> list[dict]:
    items = []
    for runs in sorted(editors.glob("*/runs.jsonl")):
        model = runs.parent.name
        for line in runs.read_text().splitlines():
            row = json.loads(line)
            if row["status"] != "ok":
                continue
            sample = row["sample_id"]
            items.append({"image_id": f"{model}/{sample}", "original": str(editors / "_inputs" / f"{sample}.png"),
                          "edit": str(runs.parent / f"{sample}.png"), "text": row["instruction"]})
            items.append({"image_id": f"{BASELINE}/{sample}", "original": str(editors / "_inputs" / f"{sample}.png"),
                          "edit": str(editors / "_inputs" / f"{sample}.png"), "text": row["instruction"]})
    unique = {item["image_id"]: item for item in items}          # the baseline of a sample is added once
    return list(unique.values())


def worker(name: str, items_path: str) -> None:
    sys.path.insert(0, str(CODE))
    import math
    from src.scorer.rewards import get_reward_model
    from src.scorer.rewards.base import EditItem, read_image, score_batch
    import torch

    items = [EditItem(i["image_id"], Path(i["original"]), Path(i["edit"]), i["text"])
             for i in json.loads(Path(items_path).read_text())]
    model = get_reward_model(name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    started = time.monotonic()
    model.load(device)
    print(f"[{name}] loaded in {time.monotonic() - started:.0f} s", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / f"{name}.jsonl", "w", encoding="utf-8") as out:
        for start in range(0, len(items), model.batch_size):
            batch_items = items[start:start + model.batch_size]
            batch = [(item, read_image(item.original), read_image(item.edit)) for item in batch_items]
            results, _ = score_batch(model, batch)
            for item, result in zip(batch_items, results):
                score = result.get("score")
                out.write(json.dumps({"image_id": item.image_id, **{k: v for k, v in result.items()
                          if isinstance(v, (int, float, str, bool, type(None)))}, "score":
                          score if score is not None and math.isfinite(score) else None}) + "\n")
            out.flush()
            print(f"[{name}] {min(start + model.batch_size, len(items))}/{len(items)}", flush=True)


def orchestrate() -> None:
    CODE.parent.mkdir(parents=True, exist_ok=True)
    if CODE.exists():
        subprocess.run(["rm", "-rf", str(CODE)], check=True)
    subprocess.run(["git", "clone", "--depth", "1", "--branch", BRANCH, REPO_URL, str(CODE)], check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "uv"], check=True)
    subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv"])
    editors = find_editors()
    items = build_items(editors)
    items_path = Path("/tmp/ej/items.json")
    items_path.write_text(json.dumps(items))
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"{len(items)} items from {editors}", flush=True)
    sys.path.insert(0, str(CODE))
    from src.common.envs import ensure_env
    from src.scorer.rewards import get_reward_model
    codes = {}
    for name in REWARDS:
        model = get_reward_model(name)
        with open(OUT / f"env-{name}.log", "a") as log:
            try:
                python = ensure_env(model.install, log, model.isolated)
            except subprocess.CalledProcessError as error:
                codes[name] = f"environment failed: {error}"
                continue
        env = {**os.environ, "EJ_OUTPUT_DIR": "/tmp/ej/out", "EJ_MODELS_DIR": "/tmp/ej/models",
               "HF_HOME": "/tmp/ej/hf", "PYTHONPATH": str(CODE)}
        process = subprocess.Popen([python, "-u", os.path.abspath(__file__), "--worker", name, "--items", str(items_path)],
                                   cwd=CODE, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                   bufsize=1)
        for line in process.stdout:
            print(line, end="", flush=True)
        codes[name] = process.wait()
    (OUT / "summary.json").write_text(json.dumps({"items": len(items), "exit_codes": codes}, indent=1))
    print(json.dumps(codes), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker")
    parser.add_argument("--items")
    cli = parser.parse_args()
    if cli.worker:
        worker(cli.worker, cli.items)
    else:
        orchestrate()
