"""One Python environment per group of extra packages, so detectors whose libraries conflict (TensorFlow, NeMo
Curator, ultralytics, ...) can run side by side on two GPUs without reinstalling over each other.

Detectors with no `install` run with the current Python. Every other `install` tuple gets its own virtual
environment in EJ_ENVS_DIR (default /tmp/ej-envs), created with `--system-site-packages`: it sees the packages
already installed (Kaggle's PyTorch, Transformers, ...) and only adds or overrides what the detector needs. An
`isolated` detector gets an environment without the machine's packages, for libraries that bring their own torch:
a venv that sees the machine's torchaudio or torchvision next to another torch fails to load them. `uv` makes
this fast (pip install uv); without it, venv and pip are used.
"""

import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

ENVS_DIR = Path(os.environ.get("EJ_ENVS_DIR") or "/tmp/ej-envs")


def env_name(install: tuple[str, ...], isolated: bool = False) -> str | None:
    if not install:
        return None
    key = "\n".join(install) + ("\nisolated" if isolated else "")
    return hashlib.sha1(key.encode()).hexdigest()[:10]


def env_python(install: tuple[str, ...], isolated: bool = False) -> str:
    name = env_name(install, isolated)
    return sys.executable if name is None else str(ENVS_DIR / name / "bin" / "python")


def ensure_env(install: tuple[str, ...], log=None, isolated: bool = False) -> str:
    """Create the environment of `install` if needed; returns its Python."""
    name = env_name(install, isolated)
    if name is None:
        return sys.executable
    folder = ENVS_DIR / name
    python = folder / "bin" / "python"
    if (folder / ".ready").exists():
        return str(python)
    shutil.rmtree(folder, ignore_errors=True)
    uv = shutil.which("uv")

    def run_logged(command: list[str]) -> None:
        subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT, text=True)

    inherit = [] if isolated else ["--system-site-packages"]
    if uv:
        run_logged([uv, "venv", *inherit, "--python", sys.executable, str(folder)])
    else:
        run_logged([sys.executable, "-m", "venv", *inherit, str(folder)])
    for pip_args in install:
        pip = [uv, "pip", "install", "--python", str(python)] if uv else [str(python), "-m", "pip", "install", "-q"]
        run_logged(pip + pip_args.replace('"', "").split())
    (folder / ".ready").write_text("\n".join(install) + "\n")
    return str(python)
