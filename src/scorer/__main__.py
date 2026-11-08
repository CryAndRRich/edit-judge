"""Scorer commands (design and results: ../docs/scorer.md).

    python -m src.scorer list                         backbones, configured runs, reward models
    python -m src.scorer embed [--gpus N] [--backbones a,b] [--limit N] [--shard i/n] [--device D] [--allow-cpu]
                                                      cache embeddings (GPU; Kaggle); --gpus runs one process per GPU
    python -m src.scorer baselines                    similarity baselines (from embeddings), human ceiling
    python -m src.scorer train [--run NAME ...]       heads of configs/scorer.toml, every seed
    python -m src.scorer latency [--run NAME ...] [--pairs N]
                                                      time one request (2 edits of a pair) per trained run and
                                                      similarity baseline (GPU)
    python -m src.scorer report                       work/scorer/report.md: runs, baselines, paired differences,
                                                      latency
    python -m src.scorer reward --model NAME [--split test] [--limit N] [--max-hours H] [--shard i/n]
                                                      one reward model or VLM judge, in the current environment
    python -m src.scorer rewards --models a,b [--max-hours H] [--limit N] [--shard i/n]
                                                      several, each in its own environment and process

Without a GPU, `embed`, `latency` and `reward` refuse more than a few items (src/common/device.py). `train` is
fast on a GPU and takes minutes on a CPU; `baselines` and `report` need no GPU.
"""

import argparse
import json


def parse_shard(text: str | None) -> tuple[int, int] | None:
    if not text:
        return None
    index, count = (int(part) for part in text.split("/"))
    if not 0 <= index < count:
        raise SystemExit(f"bad shard {text!r}: expected i/n with 0 <= i < n")
    return index, count


def split_names(text: str | None) -> list[str] | None:
    return [name.strip() for name in text.split(",") if name.strip()] if text else None


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m src.scorer", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["list", "embed", "baselines", "train", "latency", "report", "reward",
                                                      "rewards"])
    parser.add_argument("--backbones", help="embed: comma-separated backbones (default: all)")
    parser.add_argument("--gpus", type=int, help="embed: one process per GPU (default: run here)")
    parser.add_argument("--model", help="reward: the reward model")
    parser.add_argument("--models", help="rewards: comma-separated reward models, in this order")
    parser.add_argument("--split", default="test", help="reward, rewards: the split to score (default test)")
    parser.add_argument("--text", help="instruction or request (default: the config's)")
    parser.add_argument("--version", help="ranking version: all, v1 or v2 (default: the config's; v1, v2 need work/)")
    parser.add_argument("--limit", type=int, help="embed: samples; reward, rewards: edits per model")
    parser.add_argument("--max-hours", type=float, help="reward, rewards: stop before the next batch after this")
    parser.add_argument("--shard", help="embed, reward, rewards: i/n, this process's share")
    parser.add_argument("--device")
    parser.add_argument("--allow-cpu", action="store_true",
                        help="embed, latency, reward: allow a large run without a GPU")
    parser.add_argument("--workers", type=int, help="embed: DataLoader workers (default 4 on a GPU, else 0)")
    parser.add_argument("--run", action="append", help="train, latency: only these runs")
    parser.add_argument("--pairs", type=int, help="latency: requests timed (default 50)")
    parser.add_argument("--config", default="configs/scorer.toml")
    args = parser.parse_args(argv)

    from .train import load_config
    cfg = load_config(args.config)
    version = args.version or cfg.get("version", "all")
    text_kind = args.text or cfg.get("train", {}).get("text", "instruction")
    if args.command == "list":
        from .backbones import BACKBONES
        from .rewards import REWARD_MODELS
        result = {"backbones": sorted(BACKBONES), "runs": [run["name"] for run in cfg.get("runs", [])],
                  "reward_models": list(REWARD_MODELS)}
    elif args.command == "embed":
        from .embed import embed, embed_on_gpus
        backbones = split_names(args.backbones) or cfg.get("embed", {}).get("backbones")
        if args.gpus:
            result = embed_on_gpus(backbones, args.limit, args.gpus)
        else:
            result = embed(backbones, version, args.limit, args.device, args.allow_cpu, args.workers,
                           parse_shard(args.shard))
    elif args.command == "baselines":
        from .evaluate import baselines
        report = baselines(version, text_kind)
        result = {"human": report["human"],
                  "baselines": {name: {split: metrics[split]["accuracy"] for split in metrics}
                                for name, metrics in report["baselines"].items()}}
    elif args.command == "train":
        from .train import train_run
        runs = [run for run in cfg.get("runs", []) if not args.run or run["name"] in args.run]
        if args.run and len(runs) != len(set(args.run)):
            parser.error(f"unknown run in {args.run}; configured: {[run['name'] for run in cfg.get('runs', [])]}")
        result = []
        for run in runs:
            result.append(train_run(run, cfg.get("train"), version, args.device))
            print(json.dumps(result[-1]), flush=True)
    elif args.command == "latency":
        from .latency import LATENCY_PAIRS, scorer_latency
        result = scorer_latency(args.run, args.pairs or LATENCY_PAIRS, args.device, args.allow_cpu,
                                text_kind=text_kind)
    elif args.command == "report":
        from .report import report
        result = report(version)
    elif args.command == "reward":
        if not args.model:
            parser.error("reward needs --model")
        from .rewards import get_reward_model
        from .rewards.base import run
        result = run(get_reward_model(args.model), args.split, text_kind, args.limit, args.device, args.allow_cpu,
                     parse_shard(args.shard), args.max_hours)
    else:
        from .rewards import REWARD_MODELS, run_all
        result = run_all(split_names(args.models) or list(REWARD_MODELS), args.split, text_kind,
                         args.max_hours or 10.0, args.limit, parse_shard(args.shard))
    print(json.dumps(result, indent=1, ensure_ascii=False))
    if isinstance(result, dict) and result.get("failed"):
        raise SystemExit(f"failed: {result['failed']}")


if __name__ == "__main__":
    main()
