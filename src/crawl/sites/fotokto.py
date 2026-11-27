"""fotokto.ru (Russia), phpBB forum "Ретушь и постобработка фотографий" (f=30).

Most topics are discussion; discovery keeps topics whose title suggests a request for help
with editing or restoring a photo. Dates are Russian ("07 июн 2016, 17:26").
"""

import logging
import re
from datetime import datetime
from urllib.parse import urljoin

from bs4 import BeautifulSoup

from .base import Source

log = logging.getLogger(__name__)

KEYWORDS = re.compile(r"помог|помощ|прошу|обработ|отретуш|ретуш|реставр|восстанов|исправ|отредакт|фотошоп|photoshop",
                      re.I)
MONTHS = {"янв": 1, "фев": 2, "мар": 3, "апр": 4, "май": 5, "мая": 5, "июн": 6, "июл": 7, "авг": 8, "сен": 9,
          "окт": 10, "ноя": 11, "дек": 12}
DATE = re.compile(r"(\d{1,2})\s+([а-яё]{3})[а-яё]*\s+(\d{4}),?\s+(\d{1,2}):(\d{2})", re.I)


def parse_ru_date(text: str) -> str | None:
    m = DATE.search(text)
    if not m or m.group(2).lower() not in MONTHS:
        return None
    d, mon, y, hh, mm = int(m.group(1)), MONTHS[m.group(2).lower()], int(m.group(3)), int(m.group(4)), int(m.group(5))
    return datetime(y, mon, d, hh, mm).isoformat()  # forum local time (Moscow), no offset given


def parse_forum(html: str, base_url: str) -> tuple[list[dict], int]:
    soup = BeautifulSoup(html, "html.parser")
    topics = []
    for a in soup.select("a.topictitle"):
        m = re.search(r"[?&;]t=(\d+)", a.get("href", ""))
        if m:
            row = a.find_parent("dl") or a.find_parent("li")
            replies = row.select_one("dd.posts") if row else None
            topics.append({"thread_id": m.group(1), "url": f"{base_url}/forum/viewtopic.php?t={m.group(1)}",
                           "title": a.get_text(strip=True),
                           "reply_count": int(re.sub(r"\D", "", replies.get_text()) or 0) if replies else None})
    starts = [int(s) for s in re.findall(r"start=(\d+)", html)]
    return topics, max(starts, default=0)


def parse_topic(html: str, base_url: str) -> tuple[list[dict], list[int]]:
    soup = BeautifulSoup(html, "html.parser")
    posts = []
    for div in soup.select("div.post[id^='p']"):
        author_box = div.select_one(".author")
        author = div.select_one(".author a.username, .author a.username-coloured") or \
            (author_box.select("a")[-1] if author_box and author_box.select("a") else None)
        content = div.select_one("div.content")
        quoted = []
        if content:
            for q in content.select("blockquote"):
                q.decompose()
        imgs = []
        # Posts embed images hosted elsewhere (radikal.ru, 4put.ru, s*.fotokto.ru/photo/full, ...) as
        # plain <img>; relative images are smilies and data: URIs are icons.
        candidates = [i for i in (content.select("img") if content else [])
                      if i.get("src", "").startswith(("http://", "https://")) and "smilies" not in i.get("src", "")]
        for img in candidates + div.select("dl.attachbox img"):
            src = urljoin(base_url + "/forum/", img.get("src", ""))
            parent = img.find_parent("a")
            full = urljoin(base_url + "/forum/", parent["href"]) if parent and parent.get("href") else src
            imgs.append({"full_url": full, "thumb_url": src if full != src else None, "alt": img.get("alt")})
        posts.append({"post_id": div["id"].removeprefix("p"), "author": author.get_text(strip=True) if author else None,
                      "created_at": parse_ru_date(author_box.get_text(" ", strip=True)) if author_box else None,
                      "text": re.sub(r"\s+", " ", content.get_text(" ", strip=True)) if content else "",
                      "quoted_post_ids": quoted, "reaction_count": 0, "images": imgs})
    starts = sorted({int(s) for s in re.findall(r"start=(\d+)", html)})
    return posts, starts


class Fotokto(Source):
    name = "fotokto"
    label = "fotokto.ru retouching forum (Russia)"
    base_url = "https://fotokto.ru"
    forum_id = 30

    def discover(self, max_pages: int = 30, **_) -> None:
        self.check_runnable()
        start, pages, last_start = 0, 0, 0
        while pages < max_pages and start <= last_start:
            url = f"{self.base_url}/forum/viewforum.php?f={self.forum_id}" + (f"&start={start}" if start else "")
            resp = self.get_page(url)
            if resp.status_code != 200:
                break
            topics, last_start = parse_forum(resp.text, self.base_url)
            self.save_raw(f"listing/f{self.forum_id}/start-{start}.html.gz", resp.text)
            kept = [t for t in topics if KEYWORDS.search(t["title"])]
            for t in kept:
                self.store.upsert_thread({**t, "section": f"f{self.forum_id}"})
            self.store.mark_listing_page(f"f{self.forum_id}", start)
            self.store.commit()
            log.info("[%s] forum page start=%s: %s topics, %s kept", self.name, start, len(topics), len(kept))
            pages += 1
            start += 25

    def fetch_thread(self, row):
        posts, start, seen_starts = [], 0, {0}
        while True:
            resp = self.get_page(row["url"] + (f"&start={start}" if start else ""))
            if resp.status_code != 200:
                raise LookupError(f"http_{resp.status_code}")
            page_posts, starts = parse_topic(resp.text, self.base_url)
            self.save_raw(f"threads/{row['thread_id']}/start-{start}.html.gz", resp.text)
            posts.extend(page_posts)
            nxt = [s for s in starts if s > start and s not in seen_starts]
            if not nxt:
                break
            start = nxt[0]
            seen_starts.add(start)
        for p in posts:
            p["author_hash"] = self.hash_name(p.pop("author"))
        return posts, len(seen_starts), None

    def reparse(self) -> None:
        """Re-parse every cached topic page (raw/threads/<id>/start-N.html.gz) without any request."""
        import gzip
        n = 0
        for row in self.store.db.execute("SELECT thread_id FROM threads WHERE fetch_status='ok'").fetchall():
            pages = sorted((self.out_dir / "raw" / "threads" / row["thread_id"]).glob("start-*.html.gz"),
                           key=lambda p: int(p.name.split("-")[1].split(".")[0]))
            posts = []
            for page in pages:
                with gzip.open(page, "rt", encoding="utf-8") as fh:
                    posts.extend(parse_topic(fh.read(), self.base_url)[0])
            for p in posts:
                p["author_hash"] = self.hash_name(p.pop("author"))
            self.store.save_thread(row["thread_id"], len(pages), posts)
            n += 1
        self.store.commit()
        log.info("[%s] re-parsed %s threads from the raw cache", self.name, n)

    def probe_targets(self):
        return [("forum", f"{self.base_url}/forum/viewforum.php?f={self.forum_id}"),
                ("topic", f"{self.base_url}/forum/viewtopic.php?f=30&t=5296")]
