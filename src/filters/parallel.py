"""Run many detectors, or one detector on many images, over all GPUs of the machine (Kaggle: 2 x T4).

    python -m src.filters multirun --task nsfw --on benchmark [--models a,b] [--latency 50]
    python -m src.filters multirun --task nsfw --on dataset --models mobilenet,falconsai

- Every detector runs in its own process (`python -m src.filters run ...`), with its own environment (envs.py),
  on one GPU (CUDA_VISIBLE_DEVICES), so one model never holds memory another one needs, and conflicting libraries
  never meet. One worker per GPU takes the next job from a queue.
- On the benchmark (many models, few images) the models run side by side, slow models first. On the dataset (few
  models, 56,568 images) every model is split into one shard per GPU, and its models run one after the other. The
  second shard starts once the first one has loaded the model, so a download happens once.
Logs: work/filters/logs/<task>-<model>[-<shard>].log. Progress is printed as it happens: when an environment is
ready, when a job starts and ends, and every STATUS_SECONDS the last progress line of every running job.
"""

import os
import shutil
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from queue import Empty, Queue

from ..paths import FILTERS_DIR, REPO_ROOT
from . import get_detector, registry
from .envs import ensure_env

SLOW_FIRST = ["nemo", "opennsfw2", "ocr", "yolo", "nudenet", "clip", "laion", "efficientnet"]
STATUS_SECONDS = 60


def log_progress(text: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {text}", flush=True)


def last_line(path) -> str:
    try:
        lines = [line for line in path.read_text(errors="ignore").splitlines() if line.strip()]
    except FileNotFoundError:
        return ""
    return lines[-1][-160:] if lines else ""


def gpu_count() -> int:
    if not shutil.which("nvidia-smi"):
        return 0
    listing = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout
    return sum(line.startswith("GPU ") for line in listing.splitlines())


def prepare_envs(task: str, models: list[str], logs) -> dict[str, str | Exception]:
    """Model -> Python of its environment, or the error that prevented it; environments built 3 at a time."""
    installs = {model: get_detector(task, model).install for model in models}
    unique = sorted(set(installs.values()))
    log_progress(f"building {sum(bool(install) for install in unique)} environments for {', '.join(models)}")

    def build(install):
        group = "-".join(model for model in models if installs[model] == install)
        started = time.monotonic()
        with open(logs / f"env-{group}.log", "a") as log:
            python = ensure_env(install, log)
        if install:
            log_progress(f"environment for {group} ready in {time.monotonic() - started:.0f} s")
        return python

    with ThreadPoolExecutor(3) as pool:
        futures = {install: pool.submit(build, install) for install in unique}
    pythons = {}
    for model in models:
        try:
            pythons[model] = futures[installs[model]].result()
        except Exception as error:                       # noqa: BLE001  (reported, other models go on)
            pythons[model] = error
            log_progress(f"environment for {model} FAILED: {error}")
    return pythons


def multirun(task: str, on: str, models: list[str] | None = None, gpus: int | None = None, limit: int | None = None,
             latency: int = 0, sets: list[str] | None = None, shard: bool | None = None) -> dict:
    models = models or sorted(registry(task))
    gpus = gpu_count() if gpus is None else gpus
    shard = (on == "dataset") if shard is None else shard
    logs = FILTERS_DIR / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    pythons = prepare_envs(task, models, logs)
    env_seconds = round(time.monotonic() - started)
    failed = {model: f"environment: {python}" for model, python in pythons.items() if isinstance(python, Exception)}
    ready = [model for model in models if model not in failed]
    n_workers = max(gpus, 1)
    if shard and n_workers > 1:
        jobs = [(model, (index, n_workers)) for model in ready for index in range(n_workers)]
    else:
        slow_first = sorted(ready, key=lambda model: SLOW_FIRST.index(model) if model in SLOW_FIRST else 99)
        jobs = [(model, None) for model in slow_first]
    queue: Queue = Queue()
    for job in jobs:
        queue.put(job)
    loaded = {model: threading.Event() for model in ready}
    results, lock, running = {}, threading.Lock(), {}
    log_progress(f"{len(jobs)} jobs on {n_workers} {'GPU' if gpus else 'CPU worker'}{'s' if n_workers > 1 else ''}: "
                 + ", ".join(model + (f" {part[0]}/{part[1]}" if part else "") for model, part in jobs))

    def worker(gpu: int | None):
        while True:
            try:
                model, part = queue.get_nowait()
            except Empty:
                return
            if part and part[0] > 0:
                loaded[model].wait()
            name = f"{task}-{model}" + (f"-{part[0]}" if part else "")
            command = [pythons[model], "-m", "src.filters", "run", "--task", task, "--model", model, "--on", on]
            command += ["--shard", f"{part[0]}/{part[1]}"] if part else []
            command += ["--limit", str(limit)] if limit else []
            command += ["--latency", str(latency)] if latency and not part else []
            command += ["--sets", ",".join(sets)] if sets else []
            environment = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu)) if gpu is not None else None
            log_path = logs / f"{name}.log"
            job_started = time.monotonic()
            log_progress(f"start {name} on GPU {gpu}")
            with lock:
                running[name] = log_path
            with open(log_path, "a") as log:
                process = subprocess.Popen(command, cwd=REPO_ROOT, env=environment, stdout=log,
                                           stderr=subprocess.STDOUT, text=True)
                while process.poll() is None:                # let other shards start once the model is loaded
                    if not loaded[model].is_set() and "loaded " in log_path.read_text(errors="ignore"):
                        loaded[model].set()
                    time.sleep(2)
            loaded[model].set()
            with lock:
                running.pop(name, None)
                results[name] = {"gpu": gpu, "returncode": process.returncode,
                                 "seconds": round(time.monotonic() - job_started), "log": str(log_path)}
            outcome = "done" if process.returncode == 0 else "FAILED"
            log_progress(f"{outcome} {name} on GPU {gpu} in {results[name]['seconds']} s"
                         + ("" if process.returncode == 0 else f": {last_line(log_path)}"))

    def monitor(stop: threading.Event):
        while not stop.wait(STATUS_SECONDS):
            with lock:
                snapshot = dict(running)
            for name, path in snapshot.items():
                log_progress(f"  {name}: {last_line(path) or 'starting'}")

    workers = [threading.Thread(target=worker, args=(gpu if gpus else None,)) for gpu in range(n_workers)]
    stop = threading.Event()
    threading.Thread(target=monitor, args=(stop,), daemon=True).start()
    for thread in workers:
        thread.start()
    for thread in workers:
        thread.join()
    stop.set()
    failed.update({name: f"exit {result['returncode']}, see {result['log']}"
                   for name, result in results.items() if result["returncode"]})
    return {"task": task, "on": on, "gpus": gpus, "environments_seconds": env_seconds,
            "seconds": round(time.monotonic() - started), "jobs": results, "failed": failed}
