"""Image filters that remove samples before training: sensitive content (nsfw/) and watermarks (watermark/).

Several detectors run per task; each writes its scores to data/filters/<task>/<model>.jsonl (base.py). The scores
are then combined (voting.py) and compared with hand labels (evaluate.py), to decide between a vote of several
models and the single best model (faster). Commands: python -m src.filters --help.
"""

import importlib

TASKS = ("nsfw", "watermark")


def registry(task: str) -> dict[str, str]:
    """Detector name -> "module:Class" for a task, read from <task>/__init__.py (DETECTORS)."""
    if task not in TASKS:
        raise ValueError(f"unknown task {task!r}; expected one of {TASKS}")
    return importlib.import_module(f".{task}", __name__).DETECTORS


def get_detector(task: str, name: str):
    """A new detector object. Modules are imported only here, so heavy libraries load only when needed."""
    target = registry(task).get(name)
    if target is None:
        raise SystemExit(f"no detector {name!r} for {task}; registered: {sorted(registry(task)) or 'none yet'}")
    module, cls = target.split(":")
    return getattr(importlib.import_module(module, f"{__name__}.{task}"), cls)()
