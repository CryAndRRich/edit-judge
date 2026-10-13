"""Zhopped (zhopped.com, dead since about 2021) via the Internet Archive's Wayback Machine.

The live domain now hosts a gambling site and must not be crawled (README.md, "Data rules"). Archived
copies are read from web.archive.org:
    discover   Wayback CDX API lists archived /z/<request>-<zhop> pages (one page per edit, "zhop")
    threads    each request groups its zhop pages; every page is fetched raw (id_ mode, no toolbar)
    images     archived image files (/350/... thumbnails; /gz_960/... larger copies when archived)

Archive.org rate-limits aggressively (HTTP 429), so the default delay is 5 s and Retry-After is honoured.
Expect mostly 350 px images; only a small share of larger copies was ever archived.
"""

import html as htmlmod
import json
import logging
import re
from collections import defaultdict

from .base import Source

log = logging.getLogger(__name__)

CDX = "https://web.archive.org/cdx/search/cdx"
ZPAGE = re.compile(r"zhopped\.com(?::80)?/z/(\w+)-(\w+)")
IMG = re.compile(r"(?:https?:)?//(?:www\.)?zhopped\.com(?::80)?/(350|gz_960|t180)/(\w+)\.(jpe?g|png|gif)", re.I)


def wayback(ts: str, url: str, mode: str = "id_") -> str:
    return f"https://web.archive.org/web/{ts}{mode}/{url}"


class Zhopped(Source):
    name = "zhopped"
    label = "Zhopped (archived on the Wayback Machine)"
    base_url = "https://web.archive.org"
    delay = 5.0

    def discover(self, limit: int | None = None, **_) -> None:
        """List archived /z/ pages with the CDX API and group them by request ID."""
        self.check_runnable()
        params = {"url": "zhopped.com/z/*", "output": "json", "fl": "timestamp,original",
                  "filter": "statuscode:200", "collapse": "urlkey"}
        if limit:
            params["limit"] = str(limit)
        resp = self.http.get(CDX + "?" + "&".join(f"{k}={v}" for k, v in params.items()))
        if resp.status_code != 200:
            raise LookupError(f"cdx_http_{resp.status_code}")
        rows = json.loads(resp.text)[1:] if resp.text.strip() else []
        groups: dict[str, list] = defaultdict(list)
        for ts, original in rows:
            m = ZPAGE.search(original)
            if m:
                groups[m.group(1)].append({"zhop": m.group(2), "ts": ts, "original": original})
        for req, zhops in groups.items():
            first = min(zhops, key=lambda z: z["ts"])
            self.store.upsert_thread({"thread_id": req, "section": "z", "url": wayback(first["ts"], first["original"], ""),
                                      "meta": {"zhops": zhops}})
        self.store.mark_listing_page("cdx", "z")
        self.store.commit()
        log.info("[%s] CDX: %s archived pages, %s requests", self.name, len(rows), len(groups))

    def fetch_thread(self, row):
        meta = json.loads(row["meta"] or "{}")
        zhops = meta.get("zhops", [])
        req_id = row["thread_id"]
        request = {"post_id": f"{req_id}", "author_hash": None, "created_at": None, "text": None,
                   "quoted_post_ids": [], "reaction_count": 0, "images": [], "role": "request"}
        edits, title = [], None
        for z in sorted(zhops, key=lambda z: z["ts"]):
            resp = self.get_page(wayback(z["ts"], z["original"]))
            if resp.status_code != 200:
                log.warning("[%s] %s: HTTP %s", self.name, z["original"], resp.status_code)
                continue
            self.save_raw(f"threads/{req_id}/{z['zhop']}.html.gz", resp.text)
            t = re.search(r"<title>(.*?)</title>", resp.text, re.S)
            if t and not title:
                title = htmlmod.unescape(t.group(1)).replace("Zhopped - ", "").strip()
            found = {(m.group(1), m.group(2)) for m in IMG.finditer(resp.text)}
            if not request["images"] and any(i == req_id for _, i in found):
                size = "gz_960" if ("gz_960", req_id) in found else "350"
                request["images"].append({"full_url": wayback(z["ts"], f"http://zhopped.com/{size}/{req_id}.jpg", "im_"),
                                          "thumb_url": wayback(z["ts"], f"http://zhopped.com/350/{req_id}.jpg", "im_"),
                                          "alt": "request"})
            zid = z["zhop"]
            # The edit image is usually not in the page HTML (loaded by script); try the archived file paths.
            edits.append({"post_id": f"{req_id}-{zid}", "author_hash": None, "created_at": z["ts"], "text": None,
                          "quoted_post_ids": [], "reaction_count": 0, "role": "edit",
                          "images": [{"full_url": wayback(z["ts"], f"http://zhopped.com/gz_960/{zid}.jpg", "im_"),
                                      "thumb_url": wayback(z["ts"], f"http://zhopped.com/350/{zid}.jpg", "im_"),
                                      "alt": "zhop"}]})
        request["text"] = title
        if not edits:
            raise LookupError("no_archived_pages")
        self.store.db.execute("UPDATE threads SET title=? WHERE thread_id=?", (title, req_id))
        return [request] + edits, len(zhops), meta

    def image_choice(self, row):
        # Larger copies were rarely archived; the thumbnail is the dependable one.
        return (row["full_url"] or row["thumb_url"]), False, None

    def probe_targets(self):
        return [("cdx", CDX + "?url=zhopped.com/z/*&output=json&limit=3"),
                ("archived_page", wayback("20190525132325", "http://zhopped.com:80/z/2WQ1n-4BSzC"))]

    def probe(self):
        report = super().probe()
        report["note"] = "zhopped.com itself now hosts a gambling site and is never requested."
        return report
