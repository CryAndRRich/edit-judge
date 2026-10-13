"""Image filter commands.

    python -m src.filters list                                   registered detectors per task
    python -m src.filters run --task nsfw --model NAME [--scope all|originals|edits] [--limit N]
                              [--device cuda] [--allow-cpu]      score dataset images (GPU; on Kaggle)
    python -m src.filters vote --task nsfw [--config configs/filters.toml]
                                                                 combined score and flag per image
    python -m src.filters evaluate --task nsfw [--config configs/filters.toml]
                                                                 models and voting vs hand labels

Without a GPU, `run` refuses more than a few images (see src/common/device.py).
"""

import argparse
import json
import tomllib

from ..common.io import write_jsonl
from ..paths import FILTERS_DIR, REPO_ROOT
from . import TASKS, get_detector, registry
from .base import run
from .evaluate import evaluate
from .voting import combine, load_scores


def load_config(path: str, task: str) -> dict:
    with open(REPO_ROOT / path if not path.startswith("/") else path, "rb") as fh:
        return tomllib.load(fh).get(task, {})


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m src.filters", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["list", "run", "vote", "evaluate"])
    parser.add_argument("--task", choices=TASKS)
    parser.add_argument("--model")
    parser.add_argument("--scope", default="all", choices=["all", "originals", "edits"])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--device")
    parser.add_argument("--allow-cpu", action="store_true", help="run: allow a large run without a GPU (approved runs only)")
    parser.add_argument("--config", default="configs/filters.toml")
    args = parser.parse_args(argv)

    if args.command == "list":
        print(json.dumps({t: sorted(registry(t)) for t in TASKS}, indent=1))
        return
    if not args.task:
        parser.error("--task is required")
    if args.command == "run":
        if not args.model:
            parser.error("run needs --model")
        result = run(get_detector(args.task, args.model), args.scope, args.limit, args.device, args.allow_cpu)
    elif args.command == "vote":
        cfg = load_config(args.config, args.task)
        scores = load_scores(args.task, cfg.get("models", []))
        combined = combine(scores, cfg.get("method", "majority"), cfg.get("thresholds"), cfg.get("weights"))
        t = cfg.get("threshold", 0.5)
        write_jsonl(FILTERS_DIR / args.task / "decisions.jsonl",
                    ({"image_id": i, "score": round(s, 5), "flagged": s >= t} for i, s in sorted(combined.items())))
        result = {"images": len(combined), "flagged": sum(s >= t for s in combined.values()),
                  "method": cfg.get("method", "majority"), "models": cfg.get("models", [])}
    else:
        result = evaluate(args.task, load_config(args.config, args.task))
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
