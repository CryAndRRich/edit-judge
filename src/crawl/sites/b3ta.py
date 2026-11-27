"""b3ta.com image challenges (UK, English).

Each challenge has a theme text and a source image (a link in the description, or the challenge
header image) and many member entries posted on the board. Entries are loaded through
/challenge/loadMoreFromChallenge.php (not disallowed by robots.txt, which asks for a 1 s crawl delay).
Recent entries are hosted on www2.b3ta.com/host/creative/; old ones hang on third-party hosts that are
often dead. Entries are creative composites, not always edits of the source photo.
"""

import logging
import re
from html import unescape
from urllib.parse import urljoin

from bs4 import BeautifulSoup

from .base import Source

log = logging.getLogger(__name__)

SLUG = re.compile(r'href="/challenge/([a-z0-9_]+)/?"')
IMAGE_LINK = re.compile(r"\.(?:jpe?g|png|gif)(?:\?|$)", re.I)


def _text(tag) -> str:
    return re.sub(r"\s+", " ", tag.get_text(" ", strip=True)).strip() if tag else ""


def parse_challenge(html: str, base_url: str) -> dict:
    meta = lambda prop: (re.search(rf'<meta property="{prop}" content="([^"]*)"', html) or [None, ""])[1]
    desc_html = unescape(meta("og:description"))
    desc = BeautifulSoup(desc_html, "html.parser")
    links = [a["href"] for a in desc.find_all("a", href=True) if IMAGE_LINK.search(a["href"])]
    header = re.search(r'<img[^>]+src="(/images/challenge/[^"]+)"', html)
    source = links[0] if links else (urljoin(base_url, header.group(1)) if header else None)
    api = re.search(r"loadMoreFromChallenge\.php\?id=(\d+)", html)
    return {"title": meta("og:title").removeprefix("From the B3ta challenge").strip(" #"), "text": _text(desc),
            "source": source, "api_id": api.group(1) if api else None}


def parse_entries(html: str) -> list[dict]:
    soup = BeautifulSoup(html, "html.parser")
    out = []
    for div in soup.select("div[id^='post']"):
        pid = div["id"].removeprefix("post")
        imgs = []
        for img in div.find_all("img"):
            src = img.get("src", "")
            if "board_posticon" in src or not src:
                continue
            parent = img.find_parent("a")
            full = parent["href"] if parent and parent.get("href") and IMAGE_LINK.search(parent["href"]) else src
            imgs.append({"full_url": full, "thumb_url": src if full != src else None, "alt": None})
        for a in div.find_all("a", href=IMAGE_LINK):          # plain text links to an image
            if not a.find("img") and a["href"] not in [i["full_url"] for i in imgs]:
                imgs.append({"full_url": a["href"], "thumb_url": None, "alt": None})
        user = div.select_one("a.username")
        title = _text(div.find("b"))
        when = re.search(r",\s*([A-Z][a-z]{2} \d+ [A-Z][a-z]{2} \d{4}, \d+:\d+)", _text(div))
        out.append({"post_id": pid, "title": title, "images": imgs,
                    "author": user.get("href") if user else None, "created_at": when.group(1) if when else None})
    return out


class B3ta(Source):
    name = "b3ta"
    label = "b3ta.com image challenges"
    base_url = "https://b3ta.com"
    delay = 2.0          # robots.txt: Crawl-delay 1
    workers = 3
    entries_per_page = 20

    def discover(self, max_pages: int = 70, **_) -> None:
        """The index lists the latest challenges, then pages /challenge/page61/ ... /challenge/page1/."""
        self.check_runnable()
        total = 0
        for page in [None] + list(range(max_pages, 0, -1)):
            path = "/challenge/" if page is None else f"/challenge/page{page}/"
            if self.store.listing_page_done("challenges", path):
                continue
            resp = self.get_page(self.base_url + path)
            if resp.status_code != 200:
                if page is not None:
                    continue
                raise LookupError(f"http_{resp.status_code}")
            slugs = [s for s in dict.fromkeys(SLUG.findall(resp.text)) if not re.fullmatch(r"page\d+", s)]
            for slug in slugs:
                self.store.upsert_thread({"thread_id": slug, "section": "challenge",
                                          "url": f"{self.base_url}/challenge/{slug}/", "title": slug})
            total += len(slugs)
            self.store.mark_listing_page("challenges", path)
            self.store.commit()
            log.info("[%s] %s: %s challenges", self.name, path, len(slugs))

    def fetch_thread(self, row, max_pages: int = 5):
        resp = self.get_page(row["url"])
        if resp.status_code != 200:
            raise LookupError(f"http_{resp.status_code}")
        self.save_raw(f"threads/{row['thread_id']}/page-1.html.gz", resp.text)
        ch = parse_challenge(resp.text, self.base_url)
        if not ch["source"] or not ch["api_id"]:
            raise LookupError("no_source_image")
        text = (ch["title"] + ". " + ch["text"]).strip()
        posts = [{"post_id": f"{row['thread_id']}-source", "author_hash": None, "text": text[:2000], "role": "request",
                  "images": [{"full_url": ch["source"], "thumb_url": None, "alt": ch["title"]}]}]
        for page in range(1, max_pages + 1):
            r = self.get_page(f"{self.base_url}/challenge/loadMoreFromChallenge.php?id={ch['api_id']}&page_id={page}"
                              f"&page_order=popular&challenge_entries_per_page={self.entries_per_page}&infinite_scrolling=0")
            entries = parse_entries(r.text) if r.status_code == 200 else []
            self.save_raw(f"threads/{row['thread_id']}/entries-{page}.html.gz", r.text)
            if not entries:
                break
            for rank, e in enumerate(entries, (page - 1) * self.entries_per_page + 1):
                if not e["images"]:
                    continue
                posts.append({"post_id": f"{row['thread_id']}-{e['post_id']}", "author_hash": self.hash_name(e["author"]),
                              "created_at": e["created_at"], "text": e["title"] or None, "role": "edit",
                              "meta": {"popular_rank": rank}, "images": e["images"][:1]})
        return posts, max_pages, {"title": ch["title"]}

    def probe_targets(self):
        return [("challenge_list", self.base_url + "/challenge/"), ("challenge", self.base_url + "/challenge/royalphotoshopdisasters/")]
