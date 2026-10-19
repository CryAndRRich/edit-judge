"""Request analysis commands (RealEdit's simplification and taxonomy, one VLM call per request).

    python -m src.instructions list                                  registered clients and prompt versions
    python -m src.instructions run --client NAME [--prompt analyze --version K]
                                   [--sample N] [--limit N] [--device cuda] [--allow-cpu]
                                   [--shard I/N] [--max-hours H]     analyse requests (GPU; on Kaggle)
    python -m src.instructions stats --run CLIENT:PROMPT             distribution vs RealEdit's test set
    python -m src.instructions agree --run A --run B                 agreement of two runs

`--sample N` takes a fixed random subset (the same for every client and prompt), for tuning prompts.
`--shard I/N` runs every N-th request (one process per GPU); `--max-hours` stops cleanly before a session limit.
Without a GPU, `run` refuses more than a few requests (see src/common/device.py).
"""

import argparse
import json

from . import prompts
from .clients import CLIENTS, get_client
from .run import run
from .stats import agree, stats


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m src.instructions", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["list", "run", "stats", "agree"])
    parser.add_argument("--client")
    parser.add_argument("--prompt", default="analyze")
    parser.add_argument("--version", type=int, help="prompt version (default: the latest)")
    parser.add_argument("--sample", type=int, help="run: fixed random subset of N requests")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--device")
    parser.add_argument("--allow-cpu", action="store_true", help="run: allow a large run without a GPU (approved runs only)")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--shard", help="run: I/N, this process takes requests I, I+N, I+2N, ... (0 <= I < N)")
    parser.add_argument("--max-hours", type=float, help="run: stop before the next batch after H hours")
    parser.add_argument("--run", action="append", help="stats/agree: CLIENT:PROMPT, e.g. qwen2.5-vl-7b:analyze_v2")
    args = parser.parse_args(argv)

    if args.command == "list":
        result = {"clients": sorted(CLIENTS), "prompts": {"analyze": prompts.versions("analyze")}}
    elif args.command == "run":
        args.version = args.version or prompts.versions(args.prompt)[-1]
        if not args.client:
            parser.error("run needs --client")
        shard = None
        if args.shard:
            i, n = (int(x) for x in args.shard.split("/"))
            if not 0 <= i < n:
                parser.error("--shard must be I/N with 0 <= I < N")
            shard = (i, n)
        result = run(get_client(args.client), args.prompt, args.version, args.limit, args.sample, args.seed,
                     args.device, args.allow_cpu, args.max_tokens, shard, args.max_hours)
    elif args.command == "stats":
        if not args.run:
            parser.error("stats needs --run CLIENT:PROMPT")
        result = stats(args.run[0])
    else:
        if not args.run or len(args.run) != 2:
            parser.error("agree needs two --run")
        result = agree(*args.run)
    print(json.dumps(result, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
