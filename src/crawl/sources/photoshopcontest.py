"""photoshopcontest.com (English): Photoshop contests.

A contest has one original image ("source image"), a title that acts as the theme, and member
entries (edits) with vote counts. Closed contests are public: the contest page shows the original
and a thumbnail of every entry. The full-size entry image has the same token as its thumbnail
(/images/thumbs/<token>.jpg -> /images/fullsize/<token>.jpg), so no entry page is requested.
Newest contests are shown to paying members only ("not yet opened for viewing"); they are skipped.

Archive pages list 16 contests each (/archives/<offset>/photoshop-contests.html); the number of
entries is printed after each title. robots.txt allows everything.
"""

import logging
import re
from urllib.parse import urljoin

from bs4 import BeautifulSoup

from .base import Source

log = logging.getLogger(__name__)

CONTEST = re.compile(r"/photoshop-contest/(\d+)/([^/?#]+)\.html")
ENTRIES = re.compile(r"\((\d+)\)")
PAGE_SIZE = 16


def _text(tag) -> str:
    return re.sub(r"\s+", " ", tag.get_text(" ", strip=True)).strip() if tag else ""


def parse_archive(html: str, base_url: str) -> list[dict]:
    """Contests of an archive page. The thumbnail link carries the title and entry count in its alt text
    ("Golf Game Puzzle, 2 entries")."""
    soup = BeautifulSoup(html, "html.parser")
    out, seen = [], set()
    for img in soup.select("a[href*='/photoshop-contest/'] img[alt]"):
        a = img.find_parent("a")
        m = CONTEST.search(a["href"])
        alt = re.match(r"(.*),\s*(\d+)\s+entr", img["alt"])
        if not m or m.group(1) in seen:
            continue
        seen.add(m.group(1))
        out.append({"thread_id": m.group(1), "title": alt.group(1) if alt else img["alt"],
                    "url": urljoin(base_url, f"/photoshop-contest/{m.group(1)}/{m.group(2)}.html"),
                    "entries": int(alt.group(2)) if alt else None})
    return out


def parse_contest(html: str, base_url: str) -> dict | None:
    """Original image and entries of a contest page, or None when the page is members-only."""
    if "has not yet opened for viewing" in html:
        return None
    soup = BeautifulSoup(html, "html.parser")
    original = None
    for a in soup.select("a[href*='/images/queue/large/']"):
        original = urljoin(base_url, a["href"])
        break
    entries = []
    for td in soup.select("td[valign='middle'][align='center']"):
        link = td.select_one("a[href*='/view-entry/']")
        img = td.select_one("img[src*='/images/thumbs/']")
        if not link or not img:
            continue
        entry_id = re.search(r"/view-entry/(\d+)", link["href"]).group(1)
        thumb = urljoin(base_url, img["src"])
        votes = re.search(r"(\d+)\s+votes?", _text(td))
        author = td.select_one("a[href*='viewprofile']")
        entries.append({"entry_id": entry_id, "title": img.get("alt") or "", "thumb": thumb,
                        "full": thumb.replace("/images/thumbs/", "/images/fullsize/"),
                        "votes": int(votes.group(1)) if votes else 0, "author": _text(author) or None})
    return {"original": original, "entries": entries}


class PhotoshopContest(Source):
    name = "photoshopcontest"
    label = "photoshopcontest.com contests"
    base_url = "https://photoshopcontest.com"
    delay = 4.0
    workers = 2

    def discover(self, max_pages: int = 20, min_entries: int = 2, **_) -> None:
        self.check_runnable()
        for page in range(max_pages):
            path = "/archives/photoshop-contests.html" if page == 0 else f"/archives/{page * PAGE_SIZE}/photoshop-contests.html"
            if self.store.listing_page_done("archive", path):
                continue
            resp = self.get_page(self.base_url + path)
            if resp.status_code != 200:
                log.warning("[%s] %s: HTTP %s", self.name, path, resp.status_code)
                break
            self.save_raw(f"listing/archive-{page + 1}.html.gz", resp.text)
            contests = parse_archive(resp.text, self.base_url)
            kept = 0
            for c in contests:
                if c["entries"] is not None and c["entries"] < min_entries:
                    continue
                self.store.upsert_thread({"thread_id": c["thread_id"], "section": "contest", "url": c["url"],
                                          "title": c["title"], "meta": {"entries_listed": c["entries"]}})
                kept += 1
            self.store.mark_listing_page("archive", path)
            self.store.commit()
            log.info("[%s] archive page %s: %s contests, %s with >= %s entries", self.name, page + 1, len(contests), kept, min_entries)
            if not contests:
                break

    def fetch_thread(self, row):
        resp = self.get_page(row["url"])
        if resp.status_code != 200:
            raise LookupError(f"http_{resp.status_code}")
        self.save_raw(f"threads/{row['thread_id']}/page-1.html.gz", resp.text)
        parsed = parse_contest(resp.text, self.base_url)
        if not parsed or not parsed["original"]:
            raise LookupError("not_viewable")
        posts = [{"post_id": f"{row['thread_id']}-source", "author_hash": None, "text": row["title"], "role": "request",
                  "images": [{"full_url": parsed["original"], "thumb_url": None, "alt": row["title"]}]}]
        for rank, e in enumerate(sorted(parsed["entries"], key=lambda x: -x["votes"]), 1):
            posts.append({"post_id": f"{row['thread_id']}-e{e['entry_id']}", "author_hash": self.hash_name(e["author"]),
                          "text": e["title"], "role": "edit", "reaction_count": e["votes"],
                          "meta": {"votes": e["votes"], "vote_rank": rank},
                          "images": [{"full_url": e["full"], "thumb_url": e["thumb"], "alt": e["title"]}]})
        return posts, 1, {"note": "the contest title is the only request text"}

    def probe_targets(self):
        return [("archive", self.base_url + "/archives/photoshop-contests.html"),
                ("contest", self.base_url + "/photoshop-contest/1006029/container-homes.html")]
