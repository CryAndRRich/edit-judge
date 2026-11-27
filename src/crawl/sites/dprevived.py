"""dprevived.com (Misago forum rebuilt from DPReview's community): "Edit Me An Image" challenges.

Each challenge thread has one original posted by the host, many member edits, and the
winner announced by the host (often also in the title). Guests can download full-size
attachments (/a/<token>/<id>/); thumbnails are at /a/thumb/<token>/<id>/.

robots.txt disallows /api/, /p/ and /u/, so the crawler reads category and thread HTML only.
"""

import logging
import re
from urllib.parse import urljoin

from bs4 import BeautifulSoup

from .base import Source

log = logging.getLogger(__name__)

THREAD_LINK = re.compile(r"^/t/([^/]+)/(\d+)/?$")
ATTACHMENT_ID = re.compile(r"/a/(?:thumb/)?[^/]+/(\d+)/")
LIKES = re.compile(r"favorite\s*(\d+)")
OTHERS = re.compile(r"and\s+(\d+)\s+other", re.I)


def _text(tag) -> str:
    return re.sub(r"\s+", " ", tag.get_text(" ", strip=True)).strip() if tag else ""


def parse_category(html: str, base_url: str, category_path: str) -> tuple[list[dict], str | None]:
    """Threads on a category page, plus the URL of the next (older) page if any."""
    soup = BeautifulSoup(html, "html.parser")
    threads, seen = [], set()
    for a in soup.select("a[href^='/t/']"):
        m = THREAD_LINK.match(a.get("href", ""))
        title = _text(a)
        if not m or m.group(2) in seen or not title:
            continue
        seen.add(m.group(2))
        threads.append({"thread_id": m.group(2), "url": urljoin(base_url, f"/t/{m.group(1)}/{m.group(2)}/"),
                        "title": title})
    nxt = [a["href"] for a in soup.select(f"a[href^='{category_path}?page=']")]
    return threads, (urljoin(base_url, nxt[-1]) if nxt else None)


def parse_thread_page(html: str, base_url: str) -> tuple[list[dict], int]:
    soup = BeautifulSoup(html, "html.parser")
    posts = []
    for li in soup.select("li.post[id^='post-']"):
        body = li.select_one("article.misago-markup")
        quoted_authors = []
        if body:
            for q in body.select("aside.quote-block"):
                author = q.select_one("blockquote.quote-body")
                if author and author.get("data-author"):
                    quoted_authors.append(author["data-author"].lstrip("@"))
                q.decompose()  # quoted images and text belong to someone else's post
        images, seen = [], set()

        def add(full: str | None, thumb: str | None, alt: str | None) -> None:
            key_m = ATTACHMENT_ID.search(full or thumb or "")
            key = key_m.group(1) if key_m else (full or thumb)
            if key and key not in seen:
                seen.add(key)
                images.append({"full_url": full, "thumb_url": thumb, "alt": alt})

        for img in (body.select("img") if body else []):
            src = urljoin(base_url, img.get("src", ""))
            parent = img.find_parent("a")
            href = urljoin(base_url, parent["href"]).split("?")[0] if parent and parent.get("href") else ""
            if src.startswith(base_url + "/a/thumb/"):
                full = href if "/a/" in href else src.replace("/a/thumb/", "/a/")
                add(full.split("?")[0], src.split("?")[0], img.get("alt"))
            elif src.startswith(base_url + "/a/"):           # full-size attachment embedded directly
                full = src.split("?")[0]
                add(full, full.replace("/a/", "/a/thumb/", 1), img.get("alt"))
            elif src and not src.startswith("data:"):
                add(src, None, img.get("alt"))                     # external image (Flickr, ...)
        for att in li.select(".post-attachments a.attachment-name"):
            href = urljoin(base_url, att["href"]).split("?")[0]
            add(href, href.replace("/a/", "/a/thumb/", 1), _text(att))
        footer = _text(li.select_one(".post-footer"))
        likes = LIKES.search(footer)
        others = OTHERS.search(footer)
        # Footer reads e.g. "A, B and 1 other user like this. favorite 4": names are plain text.
        liked_by = re.match(r"(.*?)\s+likes?\s+this", footer)
        shown = []
        if liked_by:
            names = re.split(r",\s*|\s+and\s+", liked_by.group(1))
            shown = [n.strip() for n in names if n.strip() and not OTHERS.search(f"and {n}") and "other" not in n]
        count = int(likes.group(1)) if likes else 0
        posted = li.select_one(".post-heading time")
        author = li.select_one(".media-heading .item-title") or li.select_one(".media-heading")
        posts.append({
            "post_id": li["id"].removeprefix("post-"),
            "author": _text(author) or None,
            "created_at": posted.get("datetime") if posted else None,
            "text": _text(body),
            "quoted_post_ids": [],
            "reaction_count": count,
            "reactors_shown": shown,
            "reactors_complete": len(shown) >= count and not others,
            "meta": {"quoted_authors": quoted_authors} if quoted_authors else None,
            "images": images,
        })
    pages = [int(m.group(1)) for a in soup.select("a[href^='/t/']")
             for m in [re.search(r"/t/[^/]+/\d+/(\d+)/?$", a.get("href", ""))] if m]
    return posts, max(pages, default=1)


class Dprevived(Source):
    name = "dprevived"
    label = "dprevived.com: Edit Me An Image challenges"
    base_url = "https://dprevived.com"
    categories = {"edit-me-an-image": "/c/edit-me-an-image/198/", "challenges": "/c/challenges/197/"}
    # Keep only image-editing challenge threads (skip meta threads such as date announcements).
    title_filter = re.compile(r"edit(\s+me)?\s+an\s+image|edit\s+challenge", re.I)

    def discover(self, section: str | None = None, max_pages: int = 50, **_) -> None:
        self.check_runnable()
        for name in ([section] if section else list(self.categories)):
            path = self.categories[name]
            url, pages = self.base_url + path, 0
            while url and pages < max_pages:
                resp = self.get_page(url)
                if resp.status_code != 200:
                    log.warning("[%s] %s: HTTP %s", self.name, url, resp.status_code)
                    break
                threads, nxt = parse_category(resp.text, self.base_url, path)
                self.save_raw(f"listing/{name}/page-{pages + 1}.html.gz", resp.text)
                kept = 0
                for t in threads:
                    if self.title_filter.search(t["title"]):
                        winner = re.search(r"winner[^a-z0-9]*(?:selected|congrats(?:ulations)?)?[\s,:!-]*(.*)$",
                                           t["title"], re.I)
                        t["meta"] = {"title_announces_winner": bool(winner)}
                        self.store.upsert_thread({**t, "section": name})
                        kept += 1
                self.store.mark_listing_page(name, url)
                self.store.commit()
                pages += 1
                log.info("[%s] %s page %s: %s threads, %s challenge threads", self.name, name, pages, len(threads), kept)
                url = nxt if nxt and nxt != url and not self.store.listing_page_done(name, nxt) else None

    def fetch_thread(self, row):
        posts, page, last = [], 1, 1
        while page <= last:
            url = row["url"] if page == 1 else f"{row['url'].rstrip('/')}/{page}/"
            resp = self.get_page(url)
            if resp.status_code != 200:
                raise LookupError(f"http_{resp.status_code}")
            page_posts, last = parse_thread_page(resp.text, self.base_url)
            self.save_raw(f"threads/{row['thread_id']}/page-{page}.html.gz", resp.text)
            posts.extend(page_posts)
            page += 1
        for p in posts:
            p["author_hash"] = self.hash_name(p.pop("author"))
            p["reactor_hashes"] = [self.hash_name(r) for r in p.pop("reactors_shown")]
            if p.get("meta") and p["meta"].get("quoted_authors"):
                p["meta"]["quoted_authors"] = [self.hash_name(a) for a in p["meta"]["quoted_authors"]]
        return posts, last, None

    def probe_targets(self):
        return [("category", self.base_url + self.categories["edit-me-an-image"]),
                ("thread", self.base_url + "/t/edit-me-an-image-57-the-nobbies-winner-congrats-softmarmotte/7936/"),
                ("full_image", self.base_url + "/a/23PsK38sqxbE7DMbLcj3N0A0QnOWpSL4RN4GA50req5sZXR3ZKFMFsgEM4UBJOb1/40073/")]
