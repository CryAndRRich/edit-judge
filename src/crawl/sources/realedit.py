"""RealEdit (CVPR 2025, CC BY 4.0): Reddit requests and edits published as a CSV of image URLs.

No page is crawled: `discover` and `threads` read raw/realedit/RealEdit_train_split_urls.csv offline
(only posts with at least 2 edits), and `images` downloads the linked files (i.redd.it, imgur
pages/albums resolved to direct files, Dropbox, Drive, ...). Edits are ordered by upvotes in the CSV.

The request is the post title plus selftext, as written. RealEdit's LLM-generated `instruction`
column is not used: it can be wrong.
"""

import csv
import logging

from ..config import RAW_DIR
from .base import Source

log = logging.getLogger(__name__)

CSV = RAW_DIR / "realedit" / "RealEdit_train_split_urls.csv"
MIN_EDITS = 2


def _edits(row: dict) -> list[str]:
    return [row[f"edited_image_{i}_url"] for i in range(1, 6) if row.get(f"edited_image_{i}_url")]


class RealEdit(Source):
    name = "realedit"
    label = "RealEdit dataset (Reddit URLs)"
    base_url = "https://i.redd.it"
    delay = 1.0          # static image hosts (i.redd.it, i.imgur.com): at most one request start per second each
    workers = 6          # parallel transfers, like a browser's per-host connection limit

    def __init__(self, delay=None):
        super().__init__(delay)
        self._rows: dict[str, dict] | None = None

    def rows(self) -> dict[str, dict]:
        if self._rows is None:
            with open(CSV, newline="", encoding="utf-8") as fh:
                self._rows = {r["input_image_name"].rsplit(".", 1)[0]: r for r in csv.DictReader(fh)
                              if len(_edits(r)) >= MIN_EDITS}
        return self._rows

    def discover(self, **_) -> None:
        """Register every post with at least 2 edits (offline)."""
        rows = self.rows()
        for post_id, r in rows.items():
            self.store.upsert_thread({"thread_id": post_id, "section": r["subreddit"], "url": r["input_url"],
                                      "title": r["title"], "meta": {"split": r["split"]}})
        self.store.mark_listing_page("csv", CSV.name)
        self.store.commit()
        log.info("[%s] %s posts with >= %s edits registered", self.name, len(rows), MIN_EDITS)

    def fetch_thread(self, row):
        """Build the request and edit posts from the CSV row (offline)."""
        r = self.rows()[row["thread_id"]]
        posts = [{"post_id": f"{row['thread_id']}-req", "author_hash": None, "text": r["selftext"],
                  "role": "request", "images": [{"full_url": r["input_url"], "thumb_url": None, "alt": None}]}]
        for rank, url in enumerate(_edits(r), 1):
            posts.append({"post_id": f"{row['thread_id']}-e{rank}", "author_hash": None, "text": None,
                          "role": "edit", "meta": {"upvote_rank": rank},
                          "images": [{"full_url": url, "thumb_url": None, "alt": None}]})
        return posts, 1, None

    def probe_targets(self):
        return [("reddit_image_host", "https://i.redd.it/m5f1xktbfh311.jpg"),
                ("imgur_image_host", "https://i.imgur.com/7nlk852.jpg")]
