"""Run many detectors, or one detector on many images, over all GPUs of the machine (Kaggle: 2 x T4).

    python -m src.filters multirun --task nsfw --on benchmark [--models a,b] [--latency 50]
    python -m src.filters multirun --task nsfw --on dataset --models mobilenet,falconsai

- Every detector runs in its own process (`python -m src.filters run ...`), with its own environment (envs.py),
  on one GPU (CUDA_VISIBLE_DEVICES), so one model never holds memory another one needs, and conflicting libraries
  never meet. One worker per GPU takes the next job from a queue.
- On the benchmark (many models, few images) the models run side by side, slow models first. On the dataset (few
  models, 56,568 images) every model is split into one shard per GPU, and its models run one after the other. The
  second shard starts once the first one has loaded the model, so a download happens once.
Logs: work/filters/logs/<task>-<model>[-<shard>].log.
"""

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


def gpu_count() -> int:
    if not shutil.which("nvidia-smi"):
        return 0
    out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout
    return sum(line.startswith("GPU ") for line in out.splitlines())


def prepare_envs(task: str, models: list[str], logs) -> dict[str, str | Exception]:
    """Model -> Python of its environment, or the error that prevented it; environments built 3 at a time."""
    installs = {m: get_detector(task, m).install for m in models}
    unique = sorted(set(installs.values()))
    with ThreadPoolExecutor(3) as pool:
        futures = {}
        for inst in unique:
            log = open(logs / f"env-{'-'.join(m for m in models if installs[m] == inst)}.log", "a")
            futures[inst] = pool.submit(ensure_env, inst, log)
    out = {}
    for m in models:
        try:
            out[m] = futures[installs[m]].result()
        except Exception as e:                           # noqa: BLE001  (reported, other models go on)
            out[m] = e
    return out


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
    failed = {m: f"environment: {p}" for m, p in pythons.items() if isinstance(p, Exception)}
    ready = [m for m in models if m not in failed]
    n = max(gpus, 1)
    if shard and n > 1:
        jobs = [(m, (i, n)) for m in ready for i in range(n)]
    else:
        jobs = [(m, None) for m in sorted(ready, key=lambda m: SLOW_FIRST.index(m) if m in SLOW_FIRST else 99)]
    queue: Queue = Queue()
    for j in jobs:
        queue.put(j)
    loaded = {m: threading.Event() for m in ready}
    results, lock = {}, threading.Lock()

    def worker(gpu: int | None):
        while True:
            try:
                model, part = queue.get_nowait()
            except Empty:
                return
            if part and part[0] > 0:
                loaded[model].wait()
            name = f"{task}-{model}" + (f"-{part[0]}" if part else "")
            cmd = [pythons[model], "-m", "src.filters", "run", "--task", task, "--model", model, "--on", on]
            cmd += ["--shard", f"{part[0]}/{part[1]}"] if part else []
            cmd += ["--limit", str(limit)] if limit else []
            cmd += ["--latency", str(latency)] if latency and not part else []
            cmd += ["--sets", ",".join(sets)] if sets else []
            env = None
            if gpu is not None:
                import os
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
            log_path = logs / f"{name}.log"
            t = time.monotonic()
            with open(log_path, "a") as log:
                proc = subprocess.Popen(cmd, cwd=REPO_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, text=True)
                while proc.poll() is None:                   # let other shards start once the model is loaded
                    if not loaded[model].is_set() and "loaded " in log_path.read_text(errors="ignore"):
                        loaded[model].set()
                    time.sleep(2)
            loaded[model].set()
            with lock:
                results[name] = {"gpu": gpu, "returncode": proc.returncode, "seconds": round(time.monotonic() - t),
                                 "log": str(log_path)}
                print(f"{name} on GPU {gpu}: {'ok' if proc.returncode == 0 else 'FAILED'} "
                      f"in {results[name]['seconds']} s", flush=True)

    threads = [threading.Thread(target=worker, args=(g if gpus else None,)) for g in range(n)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    failed.update({k: f"exit {r['returncode']}, see {r['log']}" for k, r in results.items() if r["returncode"]})
    return {"task": task, "on": on, "gpus": gpus, "environments_seconds": env_seconds,
            "seconds": round(time.monotonic() - started), "jobs": results, "failed": failed}
