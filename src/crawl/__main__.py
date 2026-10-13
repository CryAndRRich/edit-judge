"""Command line for all crawl sources. See README.md, "Crawling".

    python -m src.crawl probe [SOURCE ...]                 check access for every (or the given) source
    python -m src.crawl status                             progress counters for every source
    python -m src.crawl SOURCE discover [--section S] [--first N] [--last M] [--max-pages K] [--limit L]
    python -m src.crawl SOURCE threads  [--limit N] [--section S] [--refetch]
    python -m src.crawl SOURCE reactors [--limit N]        (XenForo sources)
    python -m src.crawl SOURCE images   [--limit N] [--min-edit-images K] [--max-minutes M] [--retry-thumbnails]
    python -m src.crawl SOURCE export
    python -m src.crawl SOURCE restatus                   re-classify downloaded images, no requests
    python -m src.crawl SOURCE reparse                    re-parse cached raw pages, no requests (fotokto)
"""

import argparse
import json
import logging
import sys
from datetime import datetime, timezone

from .config import RAW_DIR
from .net import Blocked
from .sources import SOURCES
from .sources.base import SourceUnavailable

log = logging.getLogger("crawler")


def run_probe(names: list[str], delay: float | None) -> int:
    reports = []
    for name in names or list(SOURCES):
        src = SOURCES[name](delay)
        log.info("probing %s ...", name)
        reports.append(src.probe())
    out = RAW_DIR / "_probe"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"probe-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.json"
    path.write_text(json.dumps(reports, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n{'source':16} verdict")
    for r in reports:
        print(f"{r['source']:16} {r['verdict']}")
        for c in r.get("checks", []):
            flags = {k: c[k] for k in ("robots_allows", "http_status", "bot_challenge", "logged_in", "login_required",
                                       "image_bytes", "error") if k in c}
            print(f"{'':16}   {c['what']:22} {flags}")
    print(f"\nreport saved to {path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m src.crawl", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--delay", type=float, default=None, help="seconds between requests (default: per source)")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("target", help="'probe', 'status', or a source: " + ", ".join(SOURCES))
    parser.add_argument("stage", nargs="?", help="discover | threads | reactors | images | export | status")
    parser.add_argument("extra", nargs="*", help="source names for 'probe'")
    parser.add_argument("--section")
    parser.add_argument("--first", type=int, default=1)
    parser.add_argument("--last", type=int)
    parser.add_argument("--max-pages", type=int)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--refetch", action="store_true")
    parser.add_argument("--min-edit-images", type=int, default=0)
    parser.add_argument("--max-minutes", type=float)
    parser.add_argument("--retry-thumbnails", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")

    if args.target == "probe":
        names = [n for n in [args.stage, *args.extra] if n]
        unknown = [n for n in names if n not in SOURCES]
        if unknown:
            parser.error(f"unknown source(s): {unknown}")
        return run_probe(names, args.delay)
    if args.target == "status":
        print(json.dumps([SOURCES[n](args.delay).status() for n in SOURCES], indent=2))
        return 0
    if args.target not in SOURCES:
        parser.error(f"unknown source {args.target!r}; choose from {', '.join(SOURCES)}")

    src = SOURCES[args.target](args.delay)
    stage = args.stage or "status"
    try:
        if stage == "discover":
            opts = {k: v for k, v in {"section": args.section, "first": args.first, "last": args.last,
                                      "max_pages": args.max_pages, "limit": args.limit,
                                      "refresh": args.refresh}.items() if v is not None}
            src.discover(**opts)
        elif stage == "threads":
            src.crawl_threads(args.limit, args.refetch, args.section)
        elif stage == "reactors":
            if not hasattr(src, "crawl_reactors"):
                parser.error(f"{args.target} has no reactors stage")
            src.crawl_reactors(args.limit)
        elif stage == "images":
            src.download_images(args.limit, args.retry_thumbnails, args.min_edit_images, args.max_minutes)
        elif stage == "export":
            src.export()
        elif stage == "restatus":
            src.restatus()
        elif stage == "reparse":
            if not hasattr(src, "reparse"):
                parser.error(f"{args.target} has no reparse stage")
            src.reparse()
        elif stage != "status":
            parser.error(f"unknown stage {stage!r}")
        print(json.dumps(src.status(), indent=2))
    except SourceUnavailable as exc:
        log.error("not running: %s", exc)
        return 3
    except Blocked as exc:
        log.error("stopped: the site refused access (%s). Do not bypass; see README.md (Data rules).", exc)
        return 2
    except KeyboardInterrupt:
        log.warning("interrupted; progress is saved, rerun the same command to resume")
        return 130
    finally:
        if hasattr(src, "save_cookies"):
            src.save_cookies()
    return 0


if __name__ == "__main__":
    sys.exit(main())
