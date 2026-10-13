"""Reddit (r/PhotoshopRequest, r/estoration) through the official OAuth Data API.

Since 2025-11-11 every API client needs prior approval (Responsible Builder Policy;
researchers apply through Reddit for Researchers). This module works once approved
credentials are in .env:

    REDDIT_CLIENT_ID=...
    REDDIT_CLIENT_SECRET=...

Without them it refuses to run. It never falls back to unauthenticated JSON, RSS or
old.reddit (those return HTTP 403 or would circumvent the policy; see README.md, "Data rules").

Mapping: the submission is the request; top-level comments with images are edits
(score = upvotes); replies by the original poster are requester replies.
"""

import logging
import re
import time

from ..config import ENV
from ..net import Blocked
from ..resolve import looks_like_image
from .base import Source, SourceUnavailable

log = logging.getLogger(__name__)

API = "https://oauth.reddit.com"
TOKEN_URL = "https://www.reddit.com/api/v1/access_token"
LINK = re.compile(r"https?://[^\s)\]]+")
IMAGE_HOSTS = ("i.redd.it", "imgur.com", "i.imgur.com", "ibb.co", "postimg.cc", "dropbox.com", "drive.google.com",
               "preview.redd.it")


def _image_links(text: str) -> list[str]:
    links = [u.rstrip(".,") for u in LINK.findall(text or "")]
    return [u for u in links if looks_like_image(u) or any(h in u for h in IMAGE_HOSTS)]


def _media_images(thing: dict) -> list[str]:
    """Images attached through Reddit's media_metadata (galleries and image comments)."""
    out = []
    for item in (thing.get("media_metadata") or {}).values():
        src = (item.get("s") or {}).get("u") or (item.get("s") or {}).get("gif")
        if src:
            out.append(src.replace("&amp;", "&"))
    return out


class Reddit(Source):
    name = "reddit"
    label = "Reddit r/PhotoshopRequest and r/estoration (official API)"
    base_url = "https://www.reddit.com"
    delay = 1.0                      # stays well under the 100 requests/minute OAuth limit
    subreddits = ("PhotoshopRequest", "estoration")

    def __init__(self, delay=None):
        super().__init__(delay)
        self._token, self._token_expiry = None, 0.0

    def check_runnable(self) -> None:
        super().check_runnable()
        if not (ENV.get("REDDIT_CLIENT_ID") and ENV.get("REDDIT_CLIENT_SECRET")):
            raise SourceUnavailable(
                "Reddit API credentials are missing. API access needs prior approval since 2025-11-11 "
                "(Reddit for Researchers); add REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET to .env once approved.")

    def _auth(self) -> dict:
        if not self._token or time.time() > self._token_expiry - 60:
            resp = self.http.request("POST", TOKEN_URL, data={"grant_type": "client_credentials"},
                                     auth=(ENV["REDDIT_CLIENT_ID"], ENV["REDDIT_CLIENT_SECRET"]))
            if resp.status_code != 200:
                raise Blocked(f"token request failed: HTTP {resp.status_code}")
            data = resp.json()
            self._token, self._token_expiry = data["access_token"], time.time() + data.get("expires_in", 3600)
        return {"Authorization": f"bearer {self._token}"}

    def api(self, path: str, **params):
        resp = self.http.get(API + path, robots=False, headers=self._auth(), params={"raw_json": 1, **params})
        if resp.status_code != 200:
            raise LookupError(f"api_http_{resp.status_code}")
        return resp.json()

    def discover(self, section: str | None = None, max_pages: int = 10, **_) -> None:
        """Newest submissions (100 per page). Reddit listings reach back about 1,000 items."""
        self.check_runnable()
        for sub in ([section] if section else self.subreddits):
            after, pages = None, 0
            while pages < max_pages:
                data = self.api(f"/r/{sub}/new", limit=100, **({"after": after} if after else {}))
                children = data["data"]["children"]
                for c in children:
                    d = c["data"]
                    self.store.upsert_thread({
                        "thread_id": d["id"], "section": sub, "url": "https://www.reddit.com" + d["permalink"],
                        "title": d.get("title"), "prefix": d.get("link_flair_text"),
                        "author_hash": self.hash_name(d.get("author")),
                        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(d["created_utc"])),
                        "reply_count": d.get("num_comments"),
                        "meta": {"score": d.get("score"), "over_18": d.get("over_18"), "is_gallery": d.get("is_gallery")}})
                self.store.mark_listing_page(sub, after or "first")
                self.store.commit()
                pages += 1
                after = data["data"].get("after")
                log.info("[%s] r/%s page %s: %s submissions", self.name, sub, pages, len(children))
                if not after:
                    break

    def fetch_thread(self, row):
        post_listing, comment_listing = self.api(f"/comments/{row['thread_id']}", limit=500, depth=2, sort="top")
        sub = post_listing["data"]["children"][0]["data"]
        op = sub.get("author")
        req_imgs = _media_images(sub) or ([sub["url"]] if looks_like_image(sub.get("url", "")) else []) \
            or _image_links(sub.get("selftext", ""))
        posts = [{"post_id": sub["id"], "author": op, "created_at": row["created_at"],
                  "text": f"{sub.get('title', '')}\n{sub.get('selftext', '')}".strip(), "quoted_post_ids": [],
                  "reaction_count": sub.get("score", 0), "role": "request",
                  "images": [{"full_url": u, "thumb_url": None, "alt": None} for u in req_imgs]}]
        for c in comment_listing["data"]["children"]:
            if c["kind"] != "t1":
                continue
            d = c["data"]
            imgs = _media_images(d) or _image_links(d.get("body", ""))
            posts.append({"post_id": d["id"], "author": d.get("author"),
                          "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(d["created_utc"])),
                          "text": d.get("body"), "quoted_post_ids": [], "reaction_count": d.get("score", 0),
                          "images": [{"full_url": u, "thumb_url": None, "alt": None} for u in imgs]})
            replies = (d.get("replies") or {}).get("data", {}).get("children", []) if isinstance(d.get("replies"), dict) else []
            for r in replies:
                rd = r.get("data", {})
                if r.get("kind") == "t1" and rd.get("author") == op:
                    posts.append({"post_id": rd["id"], "author": op,
                                  "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(rd["created_utc"])),
                                  "text": rd.get("body"), "quoted_post_ids": [d["id"]],
                                  "reaction_count": rd.get("score", 0), "images": [], "role": "requester_reply"})
        for p in posts:
            p["author_hash"] = self.hash_name(p.pop("author"))
        return posts, 1, {"score": sub.get("score"), "link_flair_text": sub.get("link_flair_text")}

    def probe_targets(self):
        return [("unauthenticated_json", self.base_url + "/r/PhotoshopRequest/new.json?limit=1")]

    def probe(self):
        report = super().probe()
        has_creds = bool(ENV.get("REDDIT_CLIENT_ID") and ENV.get("REDDIT_CLIENT_SECRET"))
        report["api_credentials_present"] = has_creds
        if not has_creds:
            report["verdict"] = f"{report['verdict']}; API credentials missing (approval required since 2025-11-11)"
        return report
