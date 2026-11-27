"""Downloads the models of the live demo on the GPU server, and nothing else: the editors the assistant can run
(assistant.downloadable_models: 20 of the 23 benchmarked editors, without FLUX.2 klein, InstructPix2Pix and Paint
by Inpaint), Qwen2.5-VL-7B (reads the request and plans the edit), Grounding DINO and SAM 2.1 (find the region) and
SigLIP so400m (our scorer's encoder). No GPU is used.

Where it goes: everything into the repository's git-ignored work/: Hugging Face repositories into HF_HOME
(default work/hf_cache), repository code and release files into EJ_MODELS_DIR (default work/models). A second run
only checks what is there.

Usage (from the repository root, in the server env):
    python apps/web/download.py --dry-run      # the list, the sizes and the folders
    python apps/web/download.py                # everything
    python apps/web/download.py --only qwen,lama
"""

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]          # edit-judge/
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "apps" / "web"))
os.environ.setdefault("HF_HOME", str(ROOT / "work" / "hf_cache"))   # before any Hugging Face import

SUPPORT = {                                          # name -> (Hugging Face repository, what it does)
    "qwen": ("Qwen/Qwen2.5-VL-7B-Instruct", "reads the request, plans the edit"),
    "grounding-dino": ("IDEA-Research/grounding-dino-base", "finds the region (boxes)"),
    "sam2": ("facebook/sam2.1-hiera-large", "finds the region (mask)"),
    "siglip": ("google/siglip-so400m-patch14-384", "our scorer's encoder (mlp-siglip)"),
}
NOTHING = {"plain-code"}                             # editors without weights (PIL code)
SKIP_FORMATS = ["*.msgpack", "*.h5", "*.onnx", "*.ot", "flax_model*", "tf_model*", "*.gguf"]


def repo_patterns(repo: str) -> list[str]:
    """Files not needed by transformers: other frameworks, and .bin weights when safetensors exist."""
    from huggingface_hub import list_repo_files
    files = list_repo_files(repo)
    skip = list(SKIP_FORMATS)
    if any(name.endswith(".safetensors") for name in files):
        skip += ["*.bin", "*.pth", "*.pt"]
    return skip


def repo_gb(repo: str) -> float:
    from huggingface_hub import HfApi
    info = HfApi().model_info(repo, files_metadata=True)
    skip_bin = any(item.rfilename.endswith(".safetensors") for item in info.siblings)
    total = 0
    for item in info.siblings:
        name = item.rfilename
        if any(Path(name).match(pattern) for pattern in SKIP_FORMATS):
            continue
        if skip_bin and name.endswith((".bin", ".pth", ".pt")):
            continue
        total += item.size or 0
    return total / 1e9


def items(only: set[str] | None) -> list[dict]:
    from src.editors import get_editor
    from src.editors.base import Editor
    from studio.assistant import downloadable_models

    rows = []
    for name, (repo, role) in SUPPORT.items():
        rows.append({"name": name, "kind": "support", "repo": repo, "role": role})
    counted = set()
    for name in downloadable_models():
        editor = get_editor(name)
        own_fetch = type(editor).fetch is not Editor.fetch
        shared = bool(editor.repos) and set(editor.repos) <= counted     # SDXL Outpainting: Inpainting's weights
        counted.update(editor.repos)
        rows.append({"name": name, "kind": "editor", "editor": editor, "fetch": own_fetch,
                     "gb": 0.0 if shared else editor.download_gb, "shared": shared, "role": ", ".join(editor.tasks)})
    return [row for row in rows if not only or row["name"] in only]


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the live demo's models (no GPU)")
    parser.add_argument("--dry-run", action="store_true", help="list the models, sizes and folders only")
    parser.add_argument("--only", help="comma-separated names (see --dry-run)")
    args = parser.parse_args()
    only = set(args.only.split(",")) if args.only else None

    from src.paths import MODELS_DIR
    print(f"HF_HOME       = {os.environ['HF_HOME']}")
    print(f"EJ_MODELS_DIR = {MODELS_DIR}")
    rows = items(only)
    total = 0.0
    print(f"\n{'name':16} {'kind':8} {'GB':>6}  what")
    for row in rows:
        if row["kind"] == "support":
            try:
                row["gb"] = repo_gb(row["repo"])
            except Exception as error:               # offline or rate-limited: the size only
                row["gb"] = 0.0
                print(f"  (size of {row['repo']} unknown: {type(error).__name__})")
            what = f"{row['repo']}: {row['role']}"
        elif row["name"] in NOTHING:
            what = row["role"] + ": nothing to download"
        elif row["shared"]:
            what = row["role"] + f": the same weights as above ({', '.join(row['editor'].repos)})"
        else:
            what = row["role"]
        total += row["gb"]
        size = f"{row['gb']:6.2f}"
        print(f"{row['name']:16} {row['kind']:8} {size}  {what}")
    print(f"\n{len(rows)} items, about {total:.0f} GB")
    if args.dry_run:
        return

    from huggingface_hub import snapshot_download
    started = time.monotonic()
    failed = []
    for row in rows:
        began = time.monotonic()
        try:
            if row["kind"] == "support":
                snapshot_download(row["repo"], ignore_patterns=repo_patterns(row["repo"]))
            elif row["fetch"]:
                row["editor"].fetch()
            else:
                continue                             # plain code: nothing to download
            print(f"ok      {row['name']:16} {time.monotonic() - began:6.0f} s", flush=True)
        except Exception as error:                   # report and go on: a second run resumes
            failed.append(row["name"])
            print(f"FAILED  {row['name']:16} {type(error).__name__}: {error}", flush=True)
    print(f"\ndone in {(time.monotonic() - started) / 60:.0f} min; failed: {', '.join(failed) or 'none'}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
