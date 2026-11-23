"""Editing model commands (MODEL_LIST.md, all 7 lists). TASK is remove, add_replace, enhance, restore, change_color,
formatting or open_ended.

    python -m src.editors list                                   registered editors and their tasks
    python -m src.editors samples --task TASK [--size 30]        the dev samples of the task (laptop)
    python -m src.editors plan --task TASK [--limit N]           Qwen2.5-VL-7B: targets, box, prompt (GPU; Kaggle)
    python -m src.editors masks --task TASK [--limit N]          Grounding DINO + SAM 2.1 masks (GPU; Kaggle)
    python -m src.editors run --task TASK --model NAME [--limit N]
                                                                 one model, in this Python (GPU; Kaggle)
    python -m src.editors runall --task TASK [--models a,b] [--limit N]
                                                                 every model of the task, each in its own process
                                                                 (and environment, src/common/envs.py), in order
    python -m src.editors score --task TASK [--run mlp-siglip]   the scorer on human edits and model outputs (GPU)
    python -m src.editors report --task TASK [--no-sheets]       work/editors/<task>/report.md and contact sheets
    python -m src.editors summary                                work/editors/summary.md: every task and model

`--limit 5` is the smoke run of MODEL_LIST.md; without it every sample of the task (30 by default). Every step
resumes where it stopped. Without a GPU, plan, masks, run and score refuse more than a few samples (see
src/common/device.py).
"""

import argparse
import json
import subprocess
import sys

from ..paths import EDITORS_DIR, REPO_ROOT
from . import EDITORS, TASKS, editors_of, get_editor


def stream(command: list[str]) -> int:
    process = subprocess.Popen(command, cwd=REPO_ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                               bufsize=1)
    for line in process.stdout:
        print(line, end="", flush=True)
    return process.wait()


def run_all(task: str, models: list[str], limit: int | None = None, extra: list[str] | None = None) -> dict:
    """Each model in its own process (a crash or a CUDA error of one model cannot stop the others), in its own
    environment when it has an `install`."""
    from ..common.envs import ensure_env

    codes = {}
    for name in models:
        editor = get_editor(name)
        log_path = EDITORS_DIR / "logs" / f"env-{name}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(log_path, "w") as log:
                python = ensure_env(editor.install, log, editor.isolated)
        except subprocess.CalledProcessError:
            print(f"FAILED environment for {name}; see {log_path}", flush=True)
            codes[name] = "environment failed"
            continue
        print(f"=== {name} ({task})", flush=True)
        command = [python, "-u", "-m", "src.editors", "run", "--task", task, "--model", name]
        command += (["--limit", str(limit)] if limit is not None else []) + (extra or [])
        codes[name] = stream(command)
        print(f"=== {'done' if codes[name] == 0 else 'FAILED'} {name}", flush=True)
    return codes


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m src.editors", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["list", "samples", "plan", "masks", "run", "runall", "score", "report",
                                                "summary"])
    parser.add_argument("--task", choices=TASKS)
    parser.add_argument("--model", help="run: the editor")
    parser.add_argument("--models", help="runall: comma-separated editors (default: every editor of the task)")
    parser.add_argument("--size", type=int, default=30, help="samples: samples per task")
    parser.add_argument("--limit", type=int, help="plan, masks, run, runall: the first N samples only (smoke: 5)")
    parser.add_argument("--client", default="qwen2.5-vl-7b", help="plan: the VLM (src/instructions/clients)")
    parser.add_argument("--prompt-version", type=int,
                        help="plan, masks, run, score, report: the plan prompt prompts/plan_v<k>.txt (default: the "
                             "task's, plan.TASK_VERSIONS: remove 2, add_replace 3)")
    parser.add_argument("--run", default="mlp-siglip", help="score: the scorer run (work/scorer/runs/<run>)")
    parser.add_argument("--no-sheets", action="store_true", help="report: no contact sheets")
    parser.add_argument("--device")
    parser.add_argument("--allow-cpu", action="store_true", help="allow a large run without a GPU (approved only)")
    args = parser.parse_args(argv)
    if args.command == "summary":
        from .report import overview
        print(overview())
        return
    if args.command != "list" and not args.task:
        parser.error("--task is required")
    common = {"device": args.device, "allow_cpu": args.allow_cpu}
    plan_args = {"client_name": args.client, "version": args.prompt_version}

    if args.command == "list":
        for name in EDITORS:
            editor = get_editor(name)
            print(f"{name:14} {','.join(editor.tasks):12} {editor.model_id}"
                  + (f"  [env: {'; '.join(editor.install)}]" if editor.install else ""))
        return
    if args.command == "samples":
        from .samples import select
        rows = select(args.task, args.size)
        counts = {}
        for row in rows:
            counts[row["category"]] = counts.get(row["category"], 0) + 1
        print(json.dumps({"task": args.task, "samples": len(rows), "categories": counts}, indent=1))
        return
    if args.command == "plan":
        from .plan import run
        result = run(args.task, limit=args.limit, **plan_args, **common)
    elif args.command == "masks":
        from .grounding import run
        result = run(args.task, limit=args.limit, **plan_args, **common)
    elif args.command == "run":
        from .base import run
        if not args.model:
            parser.error("run needs --model")
        result = run(get_editor(args.model), args.task, limit=args.limit, **plan_args, **common)
    elif args.command == "runall":
        models = args.models.split(",") if args.models else editors_of(args.task)
        extra = (["--device", args.device] if args.device else []) + (["--allow-cpu"] if args.allow_cpu else [])
        extra += ["--client", args.client] + (["--prompt-version", str(args.prompt_version)] if args.prompt_version else [])
        result = run_all(args.task, models, args.limit, extra)
        if any(code != 0 for code in result.values()):
            print(json.dumps(result, indent=1))
            sys.exit(1)
    elif args.command == "score":
        from .score import score
        result = score(args.task, args.run, version=args.prompt_version, **common)
    else:
        from .report import report
        print(report(args.task, sheets=not args.no_sheets, **plan_args))
        return
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
