"""Helpers for runs over all GPUs of the machine (Kaggle: 2 x T4), shared by the image filters and the scorer.

`run_shards` starts one process per GPU (CUDA_VISIBLE_DEVICES), each on its own shard ("--shard i/n"), writes each
process's output to a log file, and prints the last progress line of every running process every STATUS_SECONDS,
so a notebook shows live progress without mixing the outputs.
"""

import os
import shutil
import subprocess
import time
from pathlib import Path

STATUS_SECONDS = 60


def log_progress(text: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {text}", flush=True)


def last_line(path) -> str:
    """The last progress or error line of a log (the JSON summary a run prints at the end is skipped)."""
    try:
        lines = [line for line in Path(path).read_text(errors="ignore").splitlines()
                 if line.strip() and not line.startswith((" ", "{", "}"))]
    except FileNotFoundError:
        return ""
    return lines[-1][-160:] if lines else ""


def gpu_count() -> int:
    if not shutil.which("nvidia-smi"):
        return 0
    listing = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout
    return sum(line.startswith("GPU ") for line in listing.splitlines())


def run_shards(command: list[str], name: str, log_dir: Path, gpus: int | None = None, cwd=None) -> dict:
    """Run `command + ["--shard", "i/n"]` once per GPU, side by side; returns the exit code and log of each shard.
    Without a GPU (or with one) the command runs once, unsharded."""
    gpus = gpu_count() if gpus is None else gpus
    log_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    if gpus <= 1:
        jobs = [(command, None, log_dir / f"{name}.log")]
    else:
        jobs = [(command + ["--shard", f"{index}/{gpus}"], index, log_dir / f"{name}-{index}.log")
                for index in range(gpus)]
    processes = []
    for job_command, gpu, log_path in jobs:
        environment = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu)) if gpu is not None else None
        log_file = open(log_path, "a")
        processes.append((subprocess.Popen(job_command, cwd=cwd, env=environment, stdout=log_file,
                                           stderr=subprocess.STDOUT, text=True), log_file, log_path))
        log_progress(f"start {name}" + (f" shard {gpu}/{gpus} on GPU {gpu}" if gpu is not None else ""))
    last_status = time.monotonic()
    while any(process.poll() is None for process, _, _ in processes):
        time.sleep(2)
        if time.monotonic() - last_status >= STATUS_SECONDS:
            last_status = time.monotonic()
            for process, _, log_path in processes:
                if process.poll() is None:
                    log_progress(f"  {log_path.stem}: {last_line(log_path) or 'starting'}")
    results = {}
    for process, log_file, log_path in processes:
        log_file.close()
        results[log_path.stem] = {"returncode": process.returncode, "log": str(log_path)}
        outcome = "done" if process.returncode == 0 else f"FAILED: {last_line(log_path)}"
        log_progress(f"{log_path.stem} {outcome}")
    return {"name": name, "gpus": gpus, "seconds": round(time.monotonic() - started), "shards": results,
            "failed": [stem for stem, result in results.items() if result["returncode"]]}
