"""Editing model commands (MODEL_LIST.md, all 7 lists). TASK is remove, add_replace, enhance, restore, change_color,
formatting or open_ended.

    python -m src.editors list                                   registered editors and their tasks
    python -m src.editors info                                   the same as JSON, with downloads and GPUs
    python -m src.editors fetch --model NAME                     download its code and weights only (no GPU)
    python -m src.editors samples --task TASK [--size 30]        the dev samples of the task (laptop)
    python -m src.editors plan --task TASK [--limit N]           Qwen2.5-VL-7B: targets, box, prompt (GPU; Kaggle)
    python -m src.editors masks --task TASK [--limit N]          Grounding DINO + SAM 2.1 masks (GPU; Kaggle)
    python -m src.editors run --task TASK --model NAME [--limit N]
                                                                 one model, in this Python (GPU; Kaggle)
    python -m src.editors runall --task TASK [--models a,b] [--limit N] [--per-editor N]
                                                                 every model of the task, each in its own process
                                                                 (and environment, src/common/envs.py), in order
    run and runall also take --tasks a,b,... instead of --task: the tasks one after another in the same process,
    so each model loads once
    python -m src.editors score --task TASK [--run mlp-siglip]   the scorer on human edits and model outputs (GPU)
    python -m src.editors report --task TASK [--no-sheets]       work/editors/<task>/report.md and contact sheets
    python -m src.editors summary                                work/editors/summary.md: every task and model

`--per-editor N` runs each editor on the first N samples it applies to (a quick test of every model);
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


def run_all(tasks: list[str], models: list[str], limit: int | None = None, extra: list[str] | None = None) -> dict:
    """Each model in its own process (a crash or a CUDA error of one model cannot stop the others), in its own
    environment when it has an `install`; one process runs all `tasks` of the model (it loads once)."""
    from ..common.envs import ensure_env

    codes = {}
    for name in models:
        editor = get_editor(name)
        own_tasks = [task for task in tasks if task in editor.tasks]
        if not own_tasks:
            print(f"=== {name}: none of its tasks in {', '.join(tasks)}", flush=True)
            codes[name] = "no task"
            continue
        log_path = EDITORS_DIR / "logs" / f"env-{name}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(log_path, "w") as log:
                python = ensure_env(editor.install, log, editor.isolated)
        except subprocess.CalledProcessError:
            print(f"FAILED environment for {name}; see {log_path}", flush=True)
            codes[name] = "environment failed"
            continue
        print(f"=== {name} ({', '.join(own_tasks)})", flush=True)
        command = [python, "-u", "-m", "src.editors", "run", "--tasks", ",".join(own_tasks), "--model", name]
        command += (["--limit", str(limit)] if limit is not None else []) + (extra or [])
        codes[name] = stream(command)
        print(f"=== {'done' if codes[name] == 0 else 'FAILED'} {name}", flush=True)
    return codes


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m src.editors", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["list", "info", "fetch", "samples", "plan", "masks", "run",
                                                "runall", "score", "report", "summary"])
    parser.add_argument("--task", choices=TASKS)
    parser.add_argument("--tasks", help="run, runall: comma-separated tasks, one after another in one process")
    parser.add_argument("--model", help="run: the editor")
    parser.add_argument("--models", help="runall: comma-separated editors (default: every editor of the task)")
    parser.add_argument("--size", type=int, default=30, help="samples: samples per task")
    parser.add_argument("--limit", type=int, help="plan, masks, run, runall: the first N samples only (smoke: 5)")
    parser.add_argument("--per-editor", type=int,
                        help="run, runall: the first N samples each editor applies to (quick test)")
    parser.add_argument("--client", default="qwen2.5-vl-7b", help="plan: the VLM (src/instructions/clients)")
    parser.add_argument("--prompt-version", type=int,
                        help="plan, masks, run, score, report: the plan prompt prompts/plan_v<k>.txt (default: the "
                             "task's, plan.TASK_VERSIONS: remove 2, add_replace 3)")
    parser.add_argument("--run", default="mlp-siglip", help="score: the scorer run (work/scorer/runs/<run>)")
    parser.add_argument("--no-sheets", action="store_true", help="report: no contact sheets")
    parser.add_argument("--device")
    parser.add_argument("--allow-cpu", action="store_true", help="allow a large run without a GPU (approved only)")
    args = parser.parse_args(argv)
    if args.command == "info":
        rows = []
        for name in EDITORS:
            editor = get_editor(name)
            rows.append({"name": name, "tasks": list(editor.tasks), "ops": editor.ops, "uses_mask": editor.uses_mask,
                         "uses_plan": editor.uses_plan, "repos": list(editor.repos),
                         "local_dirs": list(editor.local_dirs), "download_gb": editor.download_gb,
                         "gpus": editor.gpus, "install": list(editor.install), "isolated": editor.isolated})
        print(json.dumps(rows))
        return
    if args.command == "fetch":
        if not args.model:
            parser.error("fetch needs --model")
        get_editor(args.model).fetch()
        print(f"fetched {args.model}")
        return
    if args.command == "summary":
        from .report import overview
        print(overview())
        return
    tasks = args.tasks.split(",") if args.tasks else [args.task] if args.task else []
    if any(task not in TASKS for task in tasks):
        parser.error(f"unknown task in {args.tasks}")
    if args.tasks and args.command not in ("run", "runall"):
        parser.error("--tasks is for run and runall; the other commands take --task")
    if args.command != "list" and not tasks:
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
        editor, result, failed = get_editor(args.model), {}, False
        for task in tasks:              # one editor object: it loads once (base.run)
            try:
                result[task] = run(editor, task, limit=args.limit, per_editor=args.per_editor, **plan_args,
                                   **common)
            except SystemExit as stop:  # e.g. no plans yet for this task: go on with the next one
                print(f"{args.model} {task}: {stop}", flush=True)
                result[task], failed = str(stop), True
            print(json.dumps(result[task], indent=1), flush=True)
        if failed:
            sys.exit(1)
        return
    elif args.command == "runall":
        models = args.models.split(",") if args.models else \
            [name for name in EDITORS if any(name in editors_of(task) for task in tasks)]
        extra = (["--device", args.device] if args.device else []) + (["--allow-cpu"] if args.allow_cpu else [])
        extra += ["--per-editor", str(args.per_editor)] if args.per_editor is not None else []
        extra += ["--client", args.client] + (["--prompt-version", str(args.prompt_version)] if args.prompt_version else [])
        result = run_all(tasks, models, args.limit, extra)
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
