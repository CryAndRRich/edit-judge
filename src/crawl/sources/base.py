"""Common machinery for every source.

A source subclass implements only what is site-specific:
    discover(**opts)          find request threads and store them (Store.upsert_thread)
    fetch_thread(row)         return (posts in display order, num_pages, thread_meta or None)
and may override:
    image_choice(row)         which URL to download for an image row
    probe_targets()           URLs checked by `probe`
    check_runnable()          refuse to run (suspended source, missing credentials)

Everything else (pacing, robots.txt, refusals, checkpointing, images, export, probe) is shared.
"""

import gzip
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import hashlib
import json
import logging
import time
from pathlib import Path
from urllib.parse import urlsplit
from urllib.robotparser import RobotFileParser

import requests

from ..config import DATA_DIR, RAW_DIR, user_agent, username_salt
from ..images import classify, fetch_image
from ..net import Blocked, Disallowed, GaveUp, PoliteSession
from ..resolve import looks_like_image, resolve
from ..store import Store, now

log = logging.getLogger(__name__)

CHALLENGE_MARKERS = ("<title>Just a moment...</title>", "Checking your browser", "百度安全验证",
                     "cf-browser-verification", "challenge-platform")


class SourceUnavailable(Exception):
    """The source must not be crawled right now (suspended, or credentials missing)."""


class Source:
    name: str = ""                 # identifier, also the data directory name
    label: str = ""                # human-readable name
    base_url: str = ""
    delay: float = 3.0             # seconds between requests
    suspended_reason: str | None = None

    def __init__(self, delay: float | None = None):
        self.out_dir: Path = RAW_DIR / self.name
        self.store = Store(self.out_dir / "crawl.sqlite")
        self.http = PoliteSession(user_agent(), delay=delay or self.delay, **self.session_options())
        self.salt = username_salt()

    # --- hooks -------------------------------------------------------------------------
    def session_options(self) -> dict:
        """Extra PoliteSession arguments (e.g. cookies)."""
        return {}

    def check_runnable(self) -> None:
        if self.suspended_reason:
            raise SourceUnavailable(f"{self.name} is suspended: {self.suspended_reason}")

    def discover(self, **opts) -> None:
        raise NotImplementedError

    def fetch_thread(self, row) -> tuple[list[dict], int, dict | None]:
        raise NotImplementedError

    def image_choice(self, row) -> tuple[str | None, bool, str | None]:
        """Return (url, is_thumbnail, status_if_skipped). Default: full size, else thumbnail.
        Share links on other hosts (imgur pages and albums, Dropbox, Drive) are resolved first."""
        if row["full_url"]:
            url = row["full_url"]
            if not url.startswith(self.base_url) and not looks_like_image(url):
                direct, method = resolve(url, self.http)
                if not direct:
                    return None, False, method
                url = direct
            return url, False, None
        if row["thumb_url"]:
            return row["thumb_url"], True, None
        return None, False, "no_url"

    def probe_targets(self) -> list[tuple[str, str]]:
        """(what, url) pairs checked by `probe`: typically a listing page, a thread and an image."""
        return [("home", self.base_url)]

    # --- helpers -----------------------------------------------------------------------
    def hash_name(self, name: str | None) -> str | None:
        if not name:
            return None
        return hashlib.sha256((self.salt + name).encode("utf-8")).hexdigest()[:16]

    def save_raw(self, rel: str, text: str) -> None:
        """Keep the raw page so parsers can be fixed without re-crawling (local, git-ignored)."""
        path = self.out_dir / "raw" / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(path, "wt", encoding="utf-8") as fh:
            fh.write(text)

    def get_page(self, url: str) -> requests.Response:
        resp = self.http.get(url)
        if resp.status_code == 200 and any(m in resp.text[:5000] for m in CHALLENGE_MARKERS[:3]):
            raise Blocked(f"bot challenge page at {url}")
        return resp

    # --- stage: threads ------------------------------------------------------------------
    def crawl_threads(self, limit: int | None = None, refetch: bool = False, section: str | None = None) -> None:
        self.check_runnable()
        todo = self.store.threads_to_fetch(limit, refetch, section)
        log.info("[%s] threads to fetch: %s", self.name, len(todo))
        for n, row in enumerate(todo, 1):
            try:
                posts, num_pages, meta = self.fetch_thread(row)
            except (LookupError, Disallowed) as exc:
                reason = "robots_disallowed" if isinstance(exc, Disallowed) else str(exc)
                self.store.mark_thread_failed(row["thread_id"], reason)
                self.store.commit()
                log.warning("[%s] thread %s: %s", self.name, row["thread_id"], reason)
                continue
            self.store.save_thread(row["thread_id"], num_pages, posts, meta)
            self.store.commit()
            log.info("[%s] thread %s (%s/%s): %s posts, %s images", self.name, row["thread_id"], n, len(todo),
                     len(posts), sum(len(p["images"]) for p in posts))

    # --- stage: images -------------------------------------------------------------------
    workers: int = 4   # parallel downloads; PoliteSession still paces every host separately

    def _download_one(self, row) -> tuple[dict, bool, Exception | None]:
        """Download one image row (runs in a worker thread; no database access here)."""
        try:
            url, is_thumb, skip_status = self.image_choice(row)
        except Disallowed:
            return {"status": "robots_disallowed"}, False, None
        except GaveUp as exc:
            return {"status": str(exc)}, False, None
        except Blocked as exc:
            # Only share links on other hosts are resolved, so this refusal is a third-party host.
            log.warning("[%s] share-link host refused for %s: %s", self.name, row["post_id"], exc)
            return {"status": "blocked_host"}, False, None
        if not url:
            return {"status": skip_status}, False, None
        stem = self.out_dir / "images" / f"{row['post_id']}_{row['idx']}"
        try:
            result = fetch_image(self.http, url, stem)
        except Disallowed:
            result = {"status": "robots_disallowed", "downloaded_url": url}
        except GaveUp as exc:
            result = {"status": str(exc), "downloaded_url": url}
        except Blocked as exc:
            if urlsplit(url).netloc != urlsplit(self.base_url).netloc:
                # A third-party image host refused: skip this image, keep crawling the source.
                log.warning("[%s] image host refused %s: %s", self.name, urlsplit(url).netloc, exc)
                result = {"status": "blocked_host", "downloaded_url": url}
            elif is_thumb or not row["thumb_url"]:
                return {}, is_thumb, exc  # the source itself refused: the caller stops (never bypass)
            else:
                log.warning("[%s] full size refused for %s; using the thumbnail", self.name, row["post_id"])
                is_thumb = True
                result = fetch_image(self.http, row["thumb_url"], stem)
        return result, is_thumb, None

    def download_images(self, limit: int | None = None, retry_thumbnails: bool = False,
                        min_edit_images: int = 0, max_minutes: float | None = None) -> None:
        """Download pending images thread by thread, in parallel across hosts.

        With max_minutes, no new thread is started once the budget is used, but every thread
        already started is finished, so no request is left half-downloaded."""
        self.check_runnable()
        self.before_images()
        todo = self.store.images_to_download(limit, retry_thumbnails, min_edit_images)
        log.info("[%s] images to download: %s (threads with >= %s edit images)", self.name, len(todo), min_edit_images)
        groups: dict[str, list] = {}
        for row in todo:
            groups.setdefault(row["thread_id"], []).append(row)
        started, done, nbytes, threads, refusal = time.monotonic(), 0, 0, set(), None

        def record(row, result, is_thumb) -> None:
            nonlocal done, nbytes
            result["is_thumbnail"] = int(is_thumb) if result.get("status") not in (None, "no_url", "needs_login") else None
            self.store.save_image_result(row["post_id"], row["idx"], result)
            self.store.commit()
            done += 1
            nbytes += result.get("filesize") or 0
            threads.add(row["thread_id"])
            log.info("[%s] image %s_%s (thread %s, %s): %s %sx%s %.2f MB%s", self.name, row["post_id"], row["idx"],
                     row["thread_id"], row["role"], result.get("status"), result.get("width"), result.get("height"),
                     (result.get("filesize") or 0) / 1e6, " (thumbnail)" if is_thumb else "")

        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            in_flight = {}
            for tid, rows in groups.items():
                if refusal or (max_minutes is not None and time.monotonic() - started >= max_minutes * 60):
                    log.info("[%s] time budget of %s min reached; finishing threads already started",
                             self.name, max_minutes)
                    break
                for row in rows:
                    in_flight[pool.submit(self._download_one, row)] = row
                while len(in_flight) >= 2 * self.workers:  # keep a short queue; save results as they arrive
                    finished, _ = wait(in_flight, return_when=FIRST_COMPLETED)
                    for fut in finished:
                        result, is_thumb, exc = fut.result()
                        row = in_flight.pop(fut)
                        if exc:
                            refusal = exc
                        else:
                            record(row, result, is_thumb)
            for fut in list(in_flight):  # finish everything already started
                result, is_thumb, exc = fut.result()
                row = in_flight.pop(fut)
                if exc:
                    refusal = exc
                else:
                    record(row, result, is_thumb)
        elapsed = time.monotonic() - started
        log.info("[%s] downloaded %s images (%.1f MB) from %s threads in %.1f min", self.name, done, nbytes / 1e6,
                 len(threads), elapsed / 60)
        if refusal:
            raise refusal

    def before_images(self) -> None:
        """Hook run once before the image stage (e.g. verify a login)."""

    # --- stage: restatus (offline) ---------------------------------------------------------
    def restatus(self) -> dict:
        """Re-classify downloaded images under the current rules in images.py, reading local files
        only (no requests). Images never saved under an older rule ('unreadable') go back to pending."""
        db, changes = self.store.db, {}
        for row in db.execute("SELECT post_id, idx, status, local_path FROM images WHERE status IS NOT NULL").fetchall():
            path = DATA_DIR / row["local_path"] if row["local_path"] else None
            if path and path.exists():
                meta = classify(path.read_bytes())
                new = meta["status"]
                db.execute("UPDATE images SET status=?, format=?, width=?, height=?, filesize=?, sha256=? "
                           "WHERE post_id=? AND idx=?", (new, meta["format"], meta["width"], meta["height"],
                                                         meta["filesize"], meta["sha256"], row["post_id"], row["idx"]))
            elif row["status"] in ("unreadable", "unsupported_format"):
                new = None  # no file was kept: download again on the next images run
                db.execute("UPDATE images SET status=NULL WHERE post_id=? AND idx=?", (row["post_id"], row["idx"]))
            else:
                continue
            if new != row["status"]:
                key = f"{row['status']} -> {new or 'pending'}"
                changes[key] = changes.get(key, 0) + 1
        self.store.commit()
        log.info("[%s] restatus: %s", self.name, changes or "no changes")
        return changes

    # --- stage: export -------------------------------------------------------------------
    def export(self) -> Path:
        """Write export/posts.jsonl (one record per request thread) and export/edits.jsonl
        (one record per edit image)."""
        db, out = self.store.db, self.out_dir / "export"
        out.mkdir(parents=True, exist_ok=True)
        img_cols = ("full_url", "thumb_url", "downloaded_url", "is_thumbnail", "local_path", "sha256", "filesize",
                    "format", "width", "height", "status")
        n_posts = n_edits = 0
        with open(out / "posts.jsonl", "w", encoding="utf-8") as fp, open(out / "edits.jsonl", "w", encoding="utf-8") as fe:
            for t in db.execute("SELECT * FROM threads WHERE fetch_status='ok' ORDER BY rowid"):
                posts = db.execute("SELECT * FROM posts WHERE thread_id=? ORDER BY position", (t["thread_id"],)).fetchall()
                if not posts:
                    continue
                imgs = {p["post_id"]: [{k: i[k] for k in img_cols} | {"idx": i["idx"]} for i in db.execute(
                    "SELECT * FROM images WHERE post_id=? ORDER BY idx", (p["post_id"],))] for p in posts}
                req = posts[0]
                op = req["author_hash"]
                replies, edit_posts = [], 0
                for p in posts[1:]:
                    if p["role"] == "requester_reply":
                        replies.append({"reply_id": p["post_id"], "created_at": p["created_at"], "text": p["text"],
                                        "quoted_post_ids": json.loads(p["quoted_post_ids"] or "[]"),
                                        "images": imgs[p["post_id"]]})
                    elif p["role"] in ("edit", "image_post"):
                        edit_posts += 1
                        reactors = json.loads(p["reactor_hashes"] or "[]")
                        for img in imgs[p["post_id"]]:
                            fe.write(json.dumps({
                                "edit_id": f"{p['post_id']}_{img['idx']}", "post_id": t["thread_id"],
                                "comment_id": p["post_id"], "source": f"{self.name}/{t['section'] or ''}".rstrip("/"),
                                "author": p["author_hash"], "created_at": p["created_at"],
                                "score": p["reaction_count"], "requester_reacted": op in reactors if op else None,
                                "role": p["role"], "is_top_level": True, "text": p["text"],
                                "quoted_post_ids": json.loads(p["quoted_post_ids"] or "[]"),
                                "meta": json.loads(p["meta"]) if p["meta"] else None, "image": img,
                            }, ensure_ascii=False) + "\n")
                            n_edits += 1
                fp.write(json.dumps({
                    "post_id": t["thread_id"], "source": f"{self.name}/{t['section'] or ''}".rstrip("/"),
                    "url": t["url"], "author": op, "created_at": req["created_at"] or t["created_at"],
                    "title": t["title"], "prefix": t["prefix"], "body": req["text"],
                    "num_comments": t["reply_count"] if t["reply_count"] is not None else len(posts) - 1,
                    "num_views": t["view_count"], "num_edit_comments": edit_posts,
                    "requester_replies": replies, "op_selected_edit_id": None,
                    "meta": json.loads(t["meta"]) if t["meta"] else None, "images": imgs[req["post_id"]],
                }, ensure_ascii=False) + "\n")
                n_posts += 1
        log.info("[%s] exported %s posts and %s edits to %s", self.name, n_posts, n_edits, out)
        return out

    def status(self) -> dict:
        return {"source": self.name, **self.store.status()}

    # --- probe -----------------------------------------------------------------------------
    def probe(self) -> dict:
        """Check, with a handful of plain requests, whether this source can be crawled:
        robots.txt, HTTP status, bot challenges, login walls and image access.
        Records the attempt even for sources we cannot crawl."""
        report = {"source": self.name, "label": self.label, "checked_at": now(), "suspended": self.suspended_reason,
                  "checks": []}
        if self.suspended_reason:
            report["verdict"] = "suspended (no requests sent)"
            return report
        ua = self.http.user_agent
        for what, url in self.probe_targets():
            entry = {"what": what, "url": url}
            try:
                origin = "{0.scheme}://{0.netloc}".format(urlsplit(url))
                rp = RobotFileParser()
                rb = self.http.session.get(origin + "/robots.txt", timeout=30)
                time.sleep(self.http.delay)
                entry["robots_status"] = rb.status_code
                rp.parse(rb.text.splitlines() if rb.status_code == 200 else [])
                entry["robots_allows"] = rp.can_fetch(ua, url) if rb.status_code in (200, 404) else None
                r = self.http.session.get(url, timeout=30)
                time.sleep(self.http.delay)
                entry["http_status"] = r.status_code
                entry["content_type"] = r.headers.get("Content-Type", "")[:40]
                entry["bot_challenge"] = any(m in r.text[:6000] for m in CHALLENGE_MARKERS)
                entry.update(self.probe_extra(what, r))
            except requests.RequestException as exc:
                entry["error"] = f"{type(exc).__name__}: {str(exc)[:120]}"
            report["checks"].append(entry)
        report["verdict"] = self.probe_verdict(report["checks"])
        return report

    def probe_extra(self, what: str, resp: requests.Response) -> dict:
        """Source-specific observations for a probe response (e.g. login wall, image size)."""
        out = {}
        if resp.headers.get("Content-Type", "").startswith("image/"):
            out["image_bytes"] = len(resp.content)
        return out

    @staticmethod
    def probe_verdict(checks: list[dict]) -> str:
        if any(c.get("error") for c in checks):
            return "error"
        if any(c.get("bot_challenge") or c.get("http_status") in (401, 403, 429) for c in checks):
            return "blocked"
        if any(c.get("robots_allows") is False for c in checks):
            return "disallowed by robots.txt"
        if any(c.get("login_required") for c in checks):
            return "login required"
        return "crawlable"
