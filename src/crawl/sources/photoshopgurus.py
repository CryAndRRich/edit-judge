"""PhotoshopGurus: Free and Paid Photoshop Requests (XenForo 2).

SUSPENDED: the team's account was banned right after a logged-in crawl (2026-10-02).
The code stays as the record of the attempt and works if access is allowed again:
set PSG_PERMISSION=granted in .env only after a forum admin explicitly permits crawling.

Guests get thumbnails only; full-size attachments need a login cookie (PSG_COOKIE).
"""

import hashlib
import json
import logging

from ..config import ENV
from ..parsers import xenforo
from .base import Source, SourceUnavailable
from .xenforo_site import XenForoSource

log = logging.getLogger(__name__)

COOKIE_DOMAIN = ".photoshopgurus.com"


def _parse_cookie_string(raw: str) -> dict[str, str]:
    pairs = (part.split("=", 1) for part in raw.split(";") if "=" in part)
    return {k.strip(): v.strip() for k, v in pairs}


class PhotoshopGurus(XenForoSource):
    name = "photoshopgurus"
    label = "PhotoshopGurus forum (Free / Paid Photoshop Requests)"
    base_url = "https://www.photoshopgurus.com"
    listings = {
        "free": "/forum/forums/free-photoshop-requests.78/",
        "paid": "/forum/forums/paid-photoshop-requests.47/",
    }
    suspended_reason = None if ENV.get("PSG_PERMISSION") == "granted" else (
        "account banned after a logged-in crawl (2026-10-02); needs explicit admin permission "
        "(then set PSG_PERMISSION=granted in .env)")

    def __init__(self, delay=None):
        self.cookie = ENV.get("PSG_COOKIE") or None
        self.logged_in: bool | None = None
        super().__init__(delay)

    # --- login cookies -------------------------------------------------------------------
    @property
    def cookie_cache(self):
        return self.out_dir / "session_cookies.json"

    def session_options(self) -> dict:
        if not self.cookie:
            return {}
        fingerprint = hashlib.sha256(self.cookie.encode()).hexdigest()[:16]
        cookies = _parse_cookie_string(self.cookie)
        if self.cookie_cache.exists():
            cached = json.loads(self.cookie_cache.read_text(encoding="utf-8"))
            if cached.get("env_fingerprint") == fingerprint:
                cookies = cached["cookies"]  # cookies the forum rotated during the last run
        return {"cookies": cookies, "cookie_domain": COOKIE_DOMAIN}

    def save_cookies(self) -> None:
        """Persist rotated session cookies (local, git-ignored) for the next run."""
        if not self.cookie:
            return
        jar = {c.name: c.value for c in self.http.session.cookies if c.name.startswith("xf_")}
        self.cookie_cache.parent.mkdir(parents=True, exist_ok=True)
        self.cookie_cache.write_text(json.dumps({"env_fingerprint": hashlib.sha256(self.cookie.encode()).hexdigest()[:16],
                                                 "cookies": jar}), encoding="utf-8")
        self.cookie_cache.chmod(0o600)

    def on_page(self, html: str) -> None:
        if self.logged_in is None:
            self.logged_in = xenforo.is_logged_in(html)
            if self.cookie and not self.logged_in:
                log.warning("PSG_COOKIE is set but the forum reports a guest session; the cookie may be invalid.")
            log.info("[%s] session: %s", self.name, "logged in" if self.logged_in else "guest (thumbnails only)")

    def before_images(self) -> None:
        if self.logged_in is None:
            self.on_page(self.get_page(self.base_url + self.listings["free"]).text)
        if self.cookie and not self.logged_in:
            raise SourceUnavailable("PSG_COOKIE is set but the forum reports a guest session; refresh the cookie.")

    def image_choice(self, row):
        is_attachment = bool(row["full_url"]) and "/attachments/" in row["full_url"] and self.base_url in row["full_url"]
        if row["full_url"] and (self.logged_in or not is_attachment):
            return row["full_url"], False, None
        if row["thumb_url"]:
            return row["thumb_url"], is_attachment, None
        # Pages fetched while logged in show full-size attachments without a thumbnail.
        return None, False, "needs_login" if is_attachment else "no_url"

    @property
    def reactions_base(self) -> str:
        return self.base_url + "/forum"

    def probe_targets(self):
        return [("listing", self.base_url + self.listings["free"])]
