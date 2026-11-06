"""One Python environment per group of extra packages, so detectors whose libraries conflict (TensorFlow, NeMo
Curator, ultralytics, ...) can run side by side on two GPUs without reinstalling over each other.

Detectors with no `install` run with the current Python. Every other `install` tuple gets its own virtual
environment in EJ_ENVS_DIR (default /tmp/ej-envs), created with `--system-site-packages`: it sees the packages
already installed (Kaggle's PyTorch, Transformers, ...) and only adds or overrides what the detector needs. `uv`
makes this fast (pip install uv); without it, venv and pip are used.
"""

import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

ENVS_DIR = Path(os.environ.get("EJ_ENVS_DIR") or "/tmp/ej-envs")


def env_name(install: tuple[str, ...]) -> str | None:
    if not install:
        return None
    return hashlib.sha1("\n".join(install).encode()).hexdigest()[:10]


def env_python(install: tuple[str, ...]) -> str:
    name = env_name(install)
    return sys.executable if name is None else str(ENVS_DIR / name / "bin" / "python")


def ensure_env(install: tuple[str, ...], log=None) -> str:
    """Create the environment of `install` if needed; returns its Python."""
    name = env_name(install)
    if name is None:
        return sys.executable
    folder = ENVS_DIR / name
    python = folder / "bin" / "python"
    if (folder / ".ready").exists():
        return str(python)
    shutil.rmtree(folder, ignore_errors=True)
    uv = shutil.which("uv")
    run = lambda cmd: subprocess.run(cmd, check=True, stdout=log, stderr=subprocess.STDOUT, text=True)
    if uv:
        run([uv, "venv", "--system-site-packages", "--python", sys.executable, str(folder)])
    else:
        run([sys.executable, "-m", "venv", "--system-site-packages", str(folder)])
    for args in install:
        cmd = [uv, "pip", "install", "--python", str(python)] if uv else [str(python), "-m", "pip", "install", "-q"]
        run(cmd + args.replace('"', "").split())
    (folder / ".ready").write_text("\n".join(install) + "\n")
    return str(python)
