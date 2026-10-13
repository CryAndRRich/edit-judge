"""Streaming loop for limited disk space: fetch a batch -> download its images -> add complete
samples to data/ -> delete the batch's raw files -> repeat, until time, disk or work runs out.

Only the raw files of the cycled source are pruned; other sources' raw data is left alone.
"""

import logging
import shutil
import time

from ..crawl.sources import SOURCES
from ..paths import DATA_DIR

from .build import Builder, prune

log = logging.getLogger(__name__)


def free_gb() -> float:
    return shutil.disk_usage(DATA_DIR).free / 1e9


def cycle(source: str, batch: int = 100, min_free_gb: float = 15.0, max_minutes: float = 30.0,
          round_minutes: float = 3.0) -> dict:
    """Each round downloads for at most `round_minutes` (threads already started are always
    finished), then builds and prunes. The loop stops once `max_minutes` have been used."""
    src = SOURCES[source]()
    src.check_runnable()
    if src.store.status()["threads_listed"] == 0:
        src.discover()
    started, rounds, summary = time.monotonic(), [], {"stopped_because": None}
    while True:
        elapsed = (time.monotonic() - started) / 60
        if free_gb() < min_free_gb:
            summary["stopped_because"] = f"free disk space below {min_free_gb} GB"
            break
        if elapsed >= max_minutes:
            summary["stopped_because"] = f"time budget of {max_minutes} min used"
            break
        before = src.store.status()
        # Fetch more threads only when the download queue runs low, so each round finishes
        # the threads already fetched instead of piling up half-downloaded ones.
        if before["images_pending"] < 2 * batch:
            src.crawl_threads(limit=batch)
        if src.store.status()["threads_fetched"] == before["threads_fetched"] and before["images_pending"] == 0:
            summary["stopped_because"] = "nothing left to fetch or download"
            break
        remaining = max_minutes - (time.monotonic() - started) / 60
        src.download_images(max_minutes=max(0.1, min(round_minutes, remaining)))
        stats = Builder().run(write=True)
        pruned = prune([source])
        rounds.append({"round": len(rounds) + 1, "minutes": round((time.monotonic() - started) / 60, 1),
                       "added_samples": stats["added_samples"],
                       "this_run": stats["this_run"].get(source, {}),
                       "pruned_files": pruned["deleted_files"], "freed_gb": pruned["freed_gb"],
                       "free_disk_gb": round(free_gb(), 2)})
        log.info("round %s: +%s samples, pruned %s files, free disk %.1f GB", len(rounds),
                 stats["added_samples"], pruned["deleted_files"], free_gb())
    summary.update(rounds=rounds, source_status=src.store.status(), free_disk_gb=round(free_gb(), 2),
                   total_minutes=round((time.monotonic() - started) / 60, 1))
    return summary
