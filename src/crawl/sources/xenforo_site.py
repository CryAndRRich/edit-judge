"""Shared behaviour for XenForo 2 forums (PhotoshopGurus, VOZ)."""

import logging
import re

from ..parsers import xenforo
from .base import Source

log = logging.getLogger(__name__)


class XenForoSource(Source):
    # listing name -> forum path, e.g. {"free": "/forum/forums/free-photoshop-requests.78/"}
    listings: dict[str, str] = {}
    # Optional title filter applied at discovery (None keeps every thread).
    title_filter: re.Pattern | None = None

    def discover(self, section: str | None = None, first: int = 1, last: int | None = None,
                 refresh: bool = False, **_) -> None:
        """Crawl listing pages first..last of one listing (or of all listings).
        Page 1 is always refreshed so new threads are picked up."""
        self.check_runnable()
        for name in ([section] if section else list(self.listings)):
            self._discover_listing(name, first, last, refresh)

    def _discover_listing(self, name: str, first: int, last: int | None, refresh: bool) -> None:
        page = first
        while True:
            if page != 1 and not refresh and self.store.listing_page_done(name, page):
                page += 1
                if last is not None and page > last:
                    break
                continue
            url = self.base_url + self.listings[name] + ("" if page == 1 else f"page-{page}")
            resp = self.get_page(url)
            if resp.status_code != 200:
                log.warning("[%s] listing %s page %s: HTTP %s; stopping", self.name, name, page, resp.status_code)
                break
            self.on_page(resp.text)
            threads, final_page = xenforo.parse_listing(resp.text, self.base_url)
            self.save_raw(f"listing/{name}/page-{page}.html.gz", resp.text)
            kept = 0
            for t in threads:
                if self.title_filter and not self.title_filter.search(f"{t['prefix'] or ''} {t['title']}"):
                    continue
                t["author_hash"] = self.hash_name(t.pop("author"))
                t["section"] = name
                self.store.upsert_thread(t)
                kept += 1
            self.store.mark_listing_page(name, page)
            self.store.commit()
            log.info("[%s] listing %s page %s/%s: %s threads, %s kept", self.name, name, page, final_page,
                     len(threads), kept)
            stop_at = final_page if last is None else min(last, final_page)
            if page >= stop_at:
                break
            page += 1

    def on_page(self, html: str) -> None:
        """Hook called with every fetched page (used to detect the login state)."""

    def fetch_thread(self, row):
        posts, page, last = [], 1, 1
        while page <= last:
            resp = self.get_page(row["url"] if page == 1 else f"{row['url']}page-{page}")
            if resp.status_code != 200:
                raise LookupError(f"http_{resp.status_code}")
            self.on_page(resp.text)
            page_posts, last = xenforo.parse_thread_page(resp.text, self.base_url)
            self.save_raw(f"threads/{row['thread_id']}/page-{page}.html.gz", resp.text)
            posts.extend(page_posts)
            page += 1
        for p in posts:
            p["author_hash"] = self.hash_name(p.pop("author"))
            p["reactor_hashes"] = [self.hash_name(r) for r in p.pop("reactors_shown")]
        return posts, last, None

    def crawl_reactors(self, limit: int | None = None) -> None:
        """Fetch full reactor lists for posts whose reaction bar was truncated."""
        self.check_runnable()
        todo = self.store.posts_needing_reactors(limit)
        log.info("[%s] posts needing full reactor lists: %s", self.name, len(todo))
        for row in todo:
            resp = self.get_page(f"{self.reactions_base}/posts/{row['post_id']}/reactions")
            if resp.status_code != 200:
                log.warning("[%s] reactions for post %s: HTTP %s", self.name, row["post_id"], resp.status_code)
                continue
            names = xenforo.parse_reactions(resp.text)
            self.store.set_reactors(row["post_id"], [self.hash_name(n) for n in names])
            self.store.commit()

    @property
    def reactions_base(self) -> str:
        return self.base_url

    def probe_extra(self, what, resp):
        out = super().probe_extra(what, resp)
        if "text/html" in resp.headers.get("Content-Type", ""):
            out["logged_in"] = xenforo.is_logged_in(resp.text)
        if what == "full_attachment":
            out["login_required"] = resp.status_code in (401, 403)
        return out
