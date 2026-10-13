"""Compute device, and the guard against heavy runs on a machine without a GPU.

Heavy tasks (detectors over the dataset, LLM/VLM inference, feature extraction, editing models) run on Kaggle.
Without a GPU, `check_heavy_run` refuses runs of more than CPU_LIMIT items unless --allow-cpu is given, so a
local test stays a small smoke test.
"""

CPU_LIMIT = 20


def pick_device(prefer: str | None = None) -> str:
    """"cuda" if a GPU is available, else "mps" on Apple silicon, else "cpu". `prefer` forces a device."""
    if prefer:
        return prefer
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def check_heavy_run(device: str, n_items: int | None, allow_cpu: bool = False) -> None:
    """Stop a large run on a machine without a GPU. n_items=None means "everything"."""
    if device == "cuda" or allow_cpu:
        return
    if n_items is None or n_items > CPU_LIMIT:
        raise SystemExit(f"no GPU here ({device}): heavy runs go to Kaggle. Use --limit {CPU_LIMIT} or less for a "
                         f"local smoke test (or --allow-cpu if the run was explicitly approved).")
