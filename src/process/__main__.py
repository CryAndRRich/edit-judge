"""Build the unified dataset in data/ from data/raw/, and normalize its images.

    python -m src.process build [--dry-run] [--rebuild] [--min-edits N] [--min-side PX] [--regenerate-mineo-pairing]
    python -m src.process prune [--source S ...]       delete raw files of decided candidates
    python -m src.process cycle --source S [--batch N] [--min-free-gb G] [--max-minutes M] [--round-minutes R]
    python -m src.process normalize [--max-side PX] [--quality Q] [--workers N] [--dry-run]
"""

import argparse
import json
import logging
import sys

from .build import MIN_EDITS, MIN_SIDE, Builder, prune


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m src.process", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["build", "prune", "cycle", "normalize"])
    parser.add_argument("--dry-run", action="store_true", help="build, normalize: report without writing")
    parser.add_argument("--rebuild", action="store_true",
                        help="build: recreate data/samples from all of data/raw/ (only valid while no raw files were pruned)")
    parser.add_argument("--min-edits", type=int, default=MIN_EDITS)
    parser.add_argument("--min-side", type=int, default=MIN_SIDE, help="minimum shorter side of every image, in px")
    parser.add_argument("--regenerate-mineo-pairing", action="store_true",
                        help="rebuild data/raw/mineo/pairing.csv from the rules (discards manual corrections)")
    parser.add_argument("--source", action="append", help="prune/cycle: source name (prune: repeatable)")
    parser.add_argument("--batch", type=int, default=100, help="cycle: threads per round")
    parser.add_argument("--min-free-gb", type=float, default=15.0, help="cycle: stop below this free disk space")
    parser.add_argument("--max-minutes", type=float, default=30.0, help="cycle: total time budget")
    parser.add_argument("--round-minutes", type=float, default=3.0,
                        help="cycle: download time per round before building and pruning")
    parser.add_argument("--max-side", type=int, default=1024, help="normalize: longest image side, in px")
    parser.add_argument("--quality", type=int, default=92, help="normalize: JPEG quality")
    parser.add_argument("--workers", type=int, help="normalize: parallel processes (default: CPU count - 1)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")

    if args.command == "build":
        if args.dry_run:
            print("note: --dry-run does not record decisions; counts include already-decided candidates only on rebuild")
        stats = Builder(args.min_edits, args.min_side, args.regenerate_mineo_pairing).run(
            write=not args.dry_run, rebuild=args.rebuild)
    elif args.command == "prune":
        stats = prune(args.source)
    elif args.command == "normalize":
        from .normalize import normalize
        stats = normalize(args.max_side, args.quality, args.workers, args.dry_run)
    else:
        if not args.source or len(args.source) != 1:
            parser.error("cycle needs exactly one --source")
        from .cycle import cycle
        stats = cycle(args.source[0], args.batch, args.min_free_gb, args.max_minutes, args.round_minutes)
    print(json.dumps(stats, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
