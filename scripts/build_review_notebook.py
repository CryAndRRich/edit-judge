"""Build a self-contained Kaggle notebook from source; no weights, data, tokens or git files are bundled."""

import base64
import hashlib
import io
import json
import textwrap
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def cell(kind, source):
    result = {"cell_type": kind, "metadata": {}, "source": textwrap.dedent(source).lstrip().splitlines(keepends=True)}
    if kind == "code":
        result.update(execution_count=None, outputs=[])
    return result


def build():
    paths = sorted((ROOT / "src").rglob("*.py"))
    paths += sorted((ROOT / "configs").glob("*.toml"))
    paths += [ROOT / "apps/ranking" / name for name in ("app.py", "requirements.txt", "run_mac.command", "run_windows.bat")]
    paths += [ROOT / name for name in ("LICENSE", "README.md", "IMPROVE_RANKING.md", "REVIEW_WORKFLOW.md", "requirements.txt")]
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in paths:
            info = zipfile.ZipInfo(path.relative_to(ROOT).as_posix(), date_time=(2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, path.read_bytes())
    source = buffer.getvalue()
    encoded = base64.b64encode(source).decode()
    checksum = hashlib.sha256(source).hexdigest()
    cells = [cell("markdown", """
        # T-045: ranking-label review on Kaggle

        Keep this notebook and its outputs **private**. Attach `cryandrrich/editjudge` and grant the notebook
        access to the `HF_TOKEN` secret (read access to `IntroDS/RankingResults`). Enable Internet.
        GPU is unnecessary for `prepare`, `select`, `merge`, `apply` and `report`; use a T4 for `score`,
        `retrain` and reward-model phases. Nothing downloads to your computer.

        This notebook bundles the implementation, so an unpublished Git branch is sufficient.
        The default phase imports original ranking results only; it does not start model downloads or training.
        Run phases in order: **prepare → score → select → human review → merge → apply → retrain → report**.
        Optional second opinions run between score and select. Reviewer adjudications are always human decisions.

        Use Save Version to preserve outputs. In the next session attach that saved output as an Input;
        the setup restores it on Kaggle. You do not need to download out.zip to your own machine.
        Read `REVIEW_WORKFLOW.md` in the bundled code for schemas and command examples.
    """), cell("code", """
        ROUND = 1
        PHASE = "prepare"  # prepare, score, select, merge, apply, retrain, report,
                           # imagereward, shortlist, editreward, refresh-oof
        REVIEW_SIZE = 250
        CHUNK_SIZE = 25
        CALIBRATION_SIZE = 10
        REVIEWERS = []     # fill from prepare's reviewer names; use all available team reviewers
        MAX_HOURS = 9.0    # OOF/reward work resumes from preserved output
        DATA_PATH = None   # auto-detect a Kaggle Input containing pairs.jsonl AND images/
        RESTORE_FROM = None  # previous out/ Input path; auto-detected if there is exactly one
        BACKUP_REPO = None  # private HF dataset with write permission; required for a durable remote backup
        BACKUP_PREFIX = None  # unique directory, e.g. checkpoints/my-team/round-01
        RESTORE_BACKUP = True
        REVIEW_RESULTS = None  # mounted Input directory of this round's returned review JSONs/ZIPs
        ADJUDICATIONS = None   # optional mounted adjudications.jsonl, written by the meeting participants
        ALIASES_PATH = None   # optional JSON mapping spelling variants of original reviewer names
    """)]
    cells.append(cell("code", f'''
        import base64, hashlib, io, os, shutil, subprocess, sys, zipfile
        from pathlib import Path

        if not Path("/kaggle/working").is_dir():
            raise RuntimeError("Run this notebook on Kaggle; local execution is disabled.")
        CODE = Path("/kaggle/working/edit-judge-review")
        OUT = Path("/kaggle/working/out")
        SOURCE_B64 = "{encoded}"
        bundled = base64.b64decode(SOURCE_B64)
        assert hashlib.sha256(bundled).hexdigest() == "{checksum}"
        with zipfile.ZipFile(io.BytesIO(bundled)) as archive:
            archive.extractall(CODE)
        OUT.mkdir(exist_ok=True)
        print("Bundled source SHA256:", "{checksum}")
    '''))
    cells.append(cell("code", r'''
        import json

        def output_inputs():
            matches = []
            for root, dirs, _ in os.walk("/kaggle/input"):
                folder = Path(root)
                if ((folder / "review/prepare.json").is_file()
                    or (folder / "scorer/embeddings").is_dir()
                    or (folder / "scorer/runs").is_dir()):
                    matches.append(folder)
                    dirs[:] = []
                    continue
                dirs[:] = [name for name in dirs if name not in
                           {"images", "samples", "models", "hf-cache", "dist", "_display"}]
                if len(folder.parts) > 9:
                    dirs[:] = []
            return matches

        previous = [Path(RESTORE_FROM)] if RESTORE_FROM else output_inputs()
        if len(previous) > 1:
            raise RuntimeError(f"Several saved outputs found: {previous}. Set RESTORE_FROM explicitly.")
        if previous:
            marker = OUT / "restored-input.json"
            source = previous[0]
            if marker.exists() and json.loads(marker.read_text())["source"] != str(source):
                raise RuntimeError("A different output was restored in this session. Start a fresh Kaggle session.")
            if not marker.exists():
                shutil.copytree(source, OUT, dirs_exist_ok=True,
                                ignore=shutil.ignore_patterns("dist", "hf-cache", "models", "__pycache__"))
                marker.write_text(json.dumps({"source": str(source)}))
                print("Restored saved output from", source)

        candidates = []
        if DATA_PATH:
            candidates = [Path(DATA_PATH)]
        else:
            for pattern in ("*/pairs.jsonl", "*/*/pairs.jsonl", "*/*/*/pairs.jsonl", "*/*/*/*/pairs.jsonl"):
                candidates.extend(path.parent for path in Path("/kaggle/input").glob(pattern)
                                  if (path.parent / "images").is_dir())
        candidates = sorted(set(candidates))
        if len(candidates) != 1:
            raise RuntimeError(f"Set DATA_PATH to the final dataset folder: found {candidates}")
        DATA = candidates[0]
        os.environ["EJ_DATA_DIR"] = str(DATA)
        os.environ["EJ_OUTPUT_DIR"] = str(OUT)
        os.environ["EJ_WORK_DIR"] = str(OUT / "coordinator")
        current_round = OUT / "review" / f"round-{ROUND:02d}"
        active = DATA / "pairs.jsonl"
        if ROUND > 1:
            active = OUT / "review" / f"round-{ROUND - 1:02d}" / f"labels-r{ROUND - 1}" / "pairs.jsonl"
        if PHASE in {"retrain", "report"}:
            active = current_round / f"labels-r{ROUND}" / "pairs.jsonl"
        if PHASE == "editreward":
            active = current_round / "baseline-pairs.jsonl"
        if not active.is_file():
            raise RuntimeError(f"Missing active label version: {active}. Restore the preceding phase's output.")
        os.environ["EJ_PAIRS_FILE"] = str(active)
        try:
            from kaggle_secrets import UserSecretsClient
            os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
        except Exception:
            if PHASE == "prepare":
                raise RuntimeError("Grant this notebook access to the HF_TOKEN Kaggle secret.") from None
        sys.path.insert(0, str(CODE))
        if BACKUP_REPO or BACKUP_PREFIX:
            os.environ['EJ_BACKUP_REPO'] = BACKUP_REPO or ''
            os.environ['EJ_BACKUP_PREFIX'] = BACKUP_PREFIX or ''
        else:
            os.environ.pop('EJ_BACKUP_REPO', None)
            os.environ.pop('EJ_BACKUP_PREFIX', None)
        from src.common.checkpoint import configured_backup
        BACKUP = configured_backup(OUT)
        if BACKUP and RESTORE_BACKUP:
            print(BACKUP.restore())
        if BACKUP:
            BACKUP.sync('preflight')  # prove persistent write access before expensive GPU work
        else:
            print('No HF backup configured. Save a private Kaggle Version to preserve outputs.')
        print("Dataset:", DATA, "\nLabels:", active, "\nPhase:", PHASE)

        def command(module, *args):
            subprocess.run([sys.executable, "-u", "-m", module, *map(str, args)], cwd=CODE, check=True)
            if BACKUP:
                BACKUP.sync(module + '-' + str(args[0]))
    '''))
    cells.append(cell("code", r'''
        if PHASE in {"score", "retrain", "imagereward", "editreward", "refresh-oof"}:
            import torch
            if not torch.cuda.is_available():
                raise RuntimeError("Enable GPU for this phase. No large CPU run is allowed.")
        if PHASE == "score":
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers>=4.56,<5", "sentencepiece"], check=True)
            command("src.scorer", "embed", "--backbones", "siglip", "--gpus", torch.cuda.device_count())
            baseline_name = f"mlp-siglip-r{ROUND - 1}"
            metric_path = OUT / "scorer/runs" / baseline_name / "metrics.json"
            expected = hashlib.sha256(active.read_bytes()).hexdigest()
            if not metric_path.exists():
                command("src.scorer", "train", "--run", "mlp-siglip", "--output-name", baseline_name)
            elif json.loads(metric_path.read_text()).get("labels_sha256") != expected:
                raise RuntimeError("Existing baseline belongs to another label version.")
            command("src.scorer", "oof", "--round", ROUND, "--max-hours", MAX_HOURS)
        elif PHASE == "prepare":
            command("src.rank", "review-prepare", "--hf-results", "IntroDS/RankingResults",
                    *(["--aliases", ALIASES_PATH] if ALIASES_PATH else []))
        elif PHASE == "select":
            if len(REVIEWERS) < 2:
                raise RuntimeError("Fill REVIEWERS with the team's names shown by review-prepare first.")
            command("src.rank", "review-select", "--round", ROUND, "--size", REVIEW_SIZE)
            command("src.rank", "review-chunks", "--round", ROUND, "--reviewers", ",".join(REVIEWERS),
                    "--chunk-size", CHUNK_SIZE, "--calibration", CALIBRATION_SIZE)
        elif PHASE == "merge":
            if not REVIEW_RESULTS:
                raise RuntimeError("Set REVIEW_RESULTS to the mounted returned-review Input directory.")
            if ADJUDICATIONS:
                shutil.copyfile(ADJUDICATIONS, current_round / "adjudications.jsonl")
            command("src.rank", "review-merge", "--round", ROUND, "--results", REVIEW_RESULTS)
        elif PHASE == "apply":
            command("src.rank", "review-apply", "--round", ROUND)
        elif PHASE == "retrain":
            command("src.scorer", "train", "--run", "mlp-siglip", "--output-name", f"mlp-siglip-r{ROUND}")
        elif PHASE == "report":
            command("src.rank", "review-report", "--round", ROUND,
                    "--before-run", f"mlp-siglip-r{ROUND - 1}", "--after-run", f"mlp-siglip-r{ROUND}")
        elif PHASE == "imagereward":
            command("src.scorer", "rewards", "--models", "imagereward", "--split", "all", "--max-hours", MAX_HOURS)
        elif PHASE == "shortlist":
            command("src.rank", "review-shortlist", "--round", ROUND, "--size", 1000)
        elif PHASE == "editreward":
            command("src.scorer", "rewards", "--models", "editreward", "--split", "all", "--max-hours", MAX_HOURS)
        elif PHASE == "refresh-oof":
            command("src.scorer", "oof", "--round", ROUND, "--max-hours", MAX_HOURS)
        else:
            raise RuntimeError("Unknown PHASE")
    '''))
    cells.append(cell("markdown", """
        Save a private Version after each phase. Review ZIPs are in `out/review/round-NN/dist/` and must only
        be shared with assigned team members. Keep `chunk-map.json`, selection scores and manifest private
        to the coordinator. Return human review JSONs as a private Kaggle Input for the merge phase.
        If pending reviews remain, complete the meeting's adjudications and rerun merge before apply.
        `report.md` and `report.json` are under the round folder. Preserve the entire `out/` output for the next round.
    """))
    notebook = {"nbformat": 4, "nbformat_minor": 5, "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"}, "review_source_sha256": checksum}, "cells": cells}
    for index, item in enumerate(cells):
        item["id"] = f"review-{index:02d}"
    destination = ROOT / "notebooks/ranking_review.ipynb"
    destination.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"Wrote {destination.name}: {destination.stat().st_size:,} bytes; bundled source {len(source):,} bytes")


if __name__ == "__main__":
    build()
