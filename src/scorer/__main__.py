"""Scorer commands (pilot of the architecture in docs/scorer-survey.md).

    python -m src.scorer list                                   backbones and configured runs
    python -m src.scorer embed --backbone NAME [--limit N] [--device cuda] [--allow-cpu] [--workers K]
                                                                cache embeddings of the ranked samples (GPU; Kaggle)
    python -m src.scorer baselines                              zero-shot baselines and human ceiling
    python -m src.scorer train [--run NAME ...] [--config configs/scorer.toml]
                                                                train heads on the cached embeddings
    python -m src.scorer report                                 one table of baselines and runs

`embed` without a GPU refuses more than a few images (src/common/device.py); `--limit N` takes N samples.
`train` and `report` need no GPU (minutes on a CPU once the embeddings exist).
"""

import argparse
import json

from .backbones import BACKBONES, get_backbone


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m src.scorer", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["list", "embed", "baselines", "train", "report"])
    parser.add_argument("--backbone")
    parser.add_argument("--version", help="ranking version (default: the config's, else v1)")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--device")
    parser.add_argument("--allow-cpu", action="store_true", help="embed: allow a large run without a GPU (approved runs only)")
    parser.add_argument("--workers", type=int, help="embed: DataLoader workers (default 4 on a GPU, else 0)")
    parser.add_argument("--run", action="append", help="train: only these runs of the config")
    parser.add_argument("--config", default="configs/scorer.toml")
    args = parser.parse_args(argv)

    from .train import load_config
    cfg = load_config(args.config)
    version = args.version or cfg.get("version", "v1")
    if args.command == "list":
        result = {"backbones": sorted(BACKBONES), "runs": [r["name"] for r in cfg.get("runs", [])]}
    elif args.command == "embed":
        if not args.backbone:
            parser.error("embed needs --backbone")
        from .embed import embed
        result = embed(get_backbone(args.backbone), version, args.limit, args.device, args.allow_cpu, args.workers)
    elif args.command == "baselines":
        from .evaluate import baselines
        report = baselines(version)
        result = {"human": report["human"],
                  "baselines": {n: {s: m[s]["accuracy"] for s in m} for n, m in report["baselines"].items()}}
    elif args.command == "train":
        from .train import train_run
        runs = [r for r in cfg.get("runs", []) if not args.run or r["name"] in args.run]
        if args.run and len(runs) != len(set(args.run)):
            parser.error(f"unknown run in {args.run}; configured: {[r['name'] for r in cfg.get('runs', [])]}")
        result = []
        for r in runs:
            result.append(train_run(r, cfg.get("train"), version, args.device))
            print(json.dumps(result[-1]), flush=True)
    else:
        from .train import report
        result = report()
    print(json.dumps(result, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
