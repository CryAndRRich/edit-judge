"""Image filter commands.

    python -m src.filters list                                   registered detectors and their extra pip packages
    python -m src.filters prepare --task nsfw [--sets a,b]       download the public benchmark sets (on Kaggle)
    python -m src.filters run --task nsfw --model NAME [--on dataset|benchmark] [--scope all|originals|edits]
                              [--shard I/N] [--latency N] [--sets a,b] [--limit N] [--device cuda] [--allow-cpu]
                                                                 score images with one model (GPU; on Kaggle)
    python -m src.filters multirun --task nsfw --on benchmark|dataset [--models a,b] [--gpus N] [--latency N]
                                                                 many models / shards over all GPUs (parallel.py)
    python -m src.filters vote --task nsfw [--models a,b] [--method mean] [--threshold 0.7] [--config ...]
                                                                 combined score and flag per dataset image
    python -m src.filters evaluate --task nsfw --on benchmark [--models a,b] [--config ...]
                                                                 models and voting (every subset) vs labels,
                                                                 also as evaluation.md (tables for the report)

`--on benchmark` uses the public set of `prepare`; the default is the final dataset. Without a GPU, `run` refuses
more than a few images (see src/common/device.py).
"""

import argparse
import json
import tomllib

from ..common.io import write_jsonl
from ..paths import FILTERS_DIR, REPO_ROOT
from . import TASKS, get_detector, registry
from .base import SETS, run
from .benchmark import prepare
from .evaluate import evaluate, write_markdown
from .voting import METHODS, combine, load_scores


def load_config(path: str, task: str) -> dict:
    with open(REPO_ROOT / path if not path.startswith("/") else path, "rb") as handle:
        return tomllib.load(handle).get(task, {})


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m src.filters", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["list", "prepare", "run", "multirun", "vote", "evaluate"])
    parser.add_argument("--task", choices=TASKS)
    parser.add_argument("--model")
    parser.add_argument("--models", help="vote, evaluate: comma-separated models instead of the config's")
    parser.add_argument("--method", choices=METHODS, help="vote: instead of the config's")
    parser.add_argument("--threshold", type=float, help="vote: flag at this combined score instead of the config's")
    parser.add_argument("--on", default="dataset", choices=SETS, help="image set: the final dataset or the benchmark")
    parser.add_argument("--scope", default="all", choices=["all", "originals", "edits"])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--device")
    parser.add_argument("--allow-cpu", action="store_true",
                        help="run: allow a large run without a GPU (approved runs only)")
    parser.add_argument("--config", default="configs/filters.toml")
    parser.add_argument("--allow-local", action="store_true", help="prepare: allow the download outside Kaggle")
    parser.add_argument("--sets", help="prepare, run: comma-separated benchmark sets (default: all of the task)")
    parser.add_argument("--shard", help="run: I/N, every N-th image starting at I")
    parser.add_argument("--latency", type=int, default=0, help="run: also time N images one by one (batch 1)")
    parser.add_argument("--gpus", type=int, help="multirun: GPUs to use (default: all)")
    args = parser.parse_args(argv)

    if args.command == "list":
        print(json.dumps({task: {name: list(get_detector(task, name).install) for name in sorted(registry(task))}
                          for task in TASKS}, indent=1))
        return
    if not args.task:
        parser.error("--task is required")
    sets = args.sets.split(",") if args.sets else None
    models = args.models.split(",") if args.models else None
    if args.command == "prepare":
        result = prepare(args.task, args.allow_local, sets)
    elif args.command == "run":
        if not args.model:
            parser.error("run needs --model")
        shard = tuple(int(part) for part in args.shard.split("/")) if args.shard else None
        if shard and not (len(shard) == 2 and 0 <= shard[0] < shard[1]):
            parser.error("--shard must be I/N with 0 <= I < N")
        result = run(get_detector(args.task, args.model), args.scope, args.limit, args.device, args.allow_cpu, args.on,
                     shard, args.latency, sets)
    elif args.command == "multirun":
        from .parallel import multirun
        result = multirun(args.task, args.on, models, args.gpus, args.limit, args.latency, sets)
    elif args.command == "vote":
        cfg = load_config(args.config, args.task)
        models = models or cfg.get("models", [])
        method = args.method or cfg.get("method", "majority")
        if not models:
            parser.error("vote needs models (--models or the config)")
        scores = load_scores(args.task, models)
        combined = scores[models[0]] if len(models) == 1 else combine(scores, method, cfg.get("thresholds"),
                                                                       cfg.get("weights"))
        if args.threshold is not None:
            threshold = args.threshold
        elif len(models) > 1:
            threshold = cfg.get("threshold", 0.5)
        else:
            threshold = cfg.get("thresholds", {}).get(models[0], 0.5)
        flagged = sorted(image_id for image_id, score in combined.items() if score >= threshold)
        write_jsonl(FILTERS_DIR / args.task / "decisions.jsonl",
                    ({"image_id": image_id, "score": round(score, 5), "flagged": score >= threshold}
                     for image_id, score in sorted(combined.items())))
        originals = [image_id for image_id in flagged if image_id.endswith("/original")]
        result = {"models": models, "method": method if len(models) > 1 else "single model", "threshold": threshold,
                  "images": len(combined), "flagged": len(flagged), "flagged_originals": len(originals),
                  "flagged_edits": len(flagged) - len(originals),
                  "samples_touched": len({image_id.split("/")[0] for image_id in flagged})}
    else:
        result = evaluate(args.task, load_config(args.config, args.task), args.on, models)
        print(write_markdown(result))
        return
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
