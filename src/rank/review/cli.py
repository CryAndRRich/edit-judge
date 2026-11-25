import argparse
import json
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m src.rank")
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("review-prepare", help="import original v1/v2 JSONs or the small HF result ZIPs")
    prepare.add_argument("--results", type=Path)
    prepare.add_argument("--hf-results")
    prepare.add_argument("--aliases", type=Path, help="JSON mapping old annotator names to roster names")
    select = commands.add_parser("review-select")
    select.add_argument("--round", required=True, type=int)
    select.add_argument("--size", default=250, type=int)
    select.add_argument("--seed", default=42, type=int)
    shortlist = commands.add_parser("review-shortlist", help="top candidates for the expensive reward-model second opinion")
    shortlist.add_argument("--round", required=True, type=int)
    shortlist.add_argument("--size", default=1000, type=int)
    shortlist.add_argument("--seed", default=42, type=int)
    chunks = commands.add_parser("review-chunks")
    chunks.add_argument("--round", required=True, type=int)
    chunks.add_argument("--reviewers", required=True, help="comma-separated roster, matching imported names")
    chunks.add_argument("--chunk-size", default=25, type=int)
    chunks.add_argument("--seed", default=42, type=int)
    chunks.add_argument("--calibration", default=10, type=int)
    merge = commands.add_parser("review-merge")
    merge.add_argument("--round", required=True, type=int)
    merge.add_argument("--results", required=True, type=Path)
    apply = commands.add_parser("review-apply")
    apply.add_argument("--round", required=True, type=int)
    approve = commands.add_parser('review-approve', help='record explicit user approval of a group proposal')
    approve.add_argument('--round', required=True, type=int)
    approve.add_argument('--approver', required=True)
    approve.add_argument('--confirmation', required=True)
    report = commands.add_parser("review-report")
    report.add_argument("--round", required=True, type=int)
    report.add_argument("--before-run")
    report.add_argument("--after-run")
    args = parser.parse_args(argv)
    try:
        if args.command == "review-prepare":
            from .prepare import prepare
            result = prepare(args.results, args.hf_results, args.aliases)
        elif args.command in ("review-select", "review-shortlist"):
            from .select import select
            result = select(args.round, args.size, args.seed, args.command == "review-shortlist")
        elif args.command == "review-chunks":
            from .chunks import make_chunks
            result = make_chunks(args.round, args.reviewers.split(","), args.chunk_size, args.seed, args.calibration)
        elif args.command == "review-merge":
            from .merge import merge
            result = merge(args.round, args.results)
        elif args.command == 'review-approve':
            from .approve import approve
            result = approve(args.round, args.approver, args.confirmation)
        elif args.command == "review-apply":
            from .apply import apply
            result = apply(args.round)
        else:
            from .report import report
            result = report(args.round, args.before_run, args.after_run)
    except (ValueError, FileNotFoundError, KeyError) as error:
        parser.exit(2, f"{error}\n")
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
