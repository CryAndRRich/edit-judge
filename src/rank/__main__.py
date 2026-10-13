"""Team ranking commands.

    python -m src.rank plan | zip | app | results ...   chunks and zip files (chunks.py)
    python -m src.rank merge [--version v1]             check results, agreement (merge.py)
    python -m src.rank finalize [--version v1]          final rankings (finalize.py)

Run a command with --help for its options.
"""

import sys

from . import chunks, finalize, merge


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return
    if argv[0] == "merge":
        merge.main(argv[1:])
    elif argv[0] == "finalize":
        finalize.main(argv[1:])
    else:
        chunks.main(argv)


if __name__ == "__main__":
    main()
