"""mineo community board (king.mineo.jp, Japan): "みんなで写真加工(ﾌｫﾄﾚﾀｯﾁ)～王国現像所～".

One long-running board thread (report 7093, since 2016) where members post photos and
others retouch them. Requests and edits are both plain comments in the same thread, so the
crawler does not decide which is which: every comment with an image gets the neutral role
`image_post`, and reply links (comment IDs) are kept so Part II can pair originals with edits.

Comment images are shown at size M (about 520 px); the original is `fullsize_image.jpg`
(e.g. 1080 px), which the crawler downloads, keeping M as the thumbnail.
"""

import logging
import re
from urllib.parse import urljoin

from bs4 import BeautifulSoup

from .base import Source

log = logging.getLogger(__name__)

NICE = re.compile(r"ナイス！\s*(\d+)")


def _text(tag) -> str:
    return re.sub(r"\s+", " ", tag.get_text(" ", strip=True)).strip() if tag else ""


def _large(src: str) -> str:
    """Original-size variant: comment images use fullsize_image, section images use L_image."""
    if "content_comment_images" in src:
        return re.sub(r"/(?:XS|S|M|L|SQ_XS)_image\.(\w+).*$", r"/fullsize_image.\1", src)
    return re.sub(r"/(?:XS|S|M|SQ_XS)_image\.", "/L_image.", src)


def parse_report_page(html: str, base_url: str, include_opening: bool) -> tuple[list[dict], int]:
    soup = BeautifulSoup(html, "html.parser")
    posts = []
    if include_opening:
        body = soup.select_one("article.card-body")
        author = soup.select_one("a.text-body[href^='/my/']")
        posted = soup.select_one("article.card-body time") or soup.select_one("time")
        imgs = [{"full_url": _large(i["src"]), "thumb_url": i["src"], "alt": i.get("alt")}
                for i in (body.select("img[src*='content_section_images']") if body else [])]
        posts.append({"post_id": "report", "author": author.get("title") if author else None,
                      "created_at": posted.get("datetime") if posted else None, "text": _text(body)[:4000],
                      "quoted_post_ids": [], "reaction_count": 0, "images": imgs, "role": "request"})
    for body in soup.select("div[id^='comment-body-']"):
        cid = body["id"].removeprefix("comment-body-")
        box = body.find_parent("div", class_="my-4")
        author = box.select_one(".media-body a.text-body") if box else None
        posted = box.select_one("time") if box else None
        nice = NICE.search(_text(box.select_one(".my-2"))) if box else None
        replied = sorted({m.group(1) for a in body.select("a[href*='/comments/']")
                          for m in [re.search(r"/comments/(\d+)", a["href"])] if m})
        imgs = []
        for img in body.select("img[src*='content_comment_images']"):
            imgs.append({"full_url": _large(img["src"]), "thumb_url": img["src"], "alt": img.get("alt")})
        posts.append({"post_id": cid, "author": author.get("title") if author else None,
                      "created_at": posted.get("datetime") if posted else None, "text": _text(body),
                      "quoted_post_ids": replied, "reaction_count": int(nice.group(1)) if nice else 0,
                      "images": imgs, "role": "image_post" if imgs else "comment"})
    pages = [int(p) for p in re.findall(r"[?&]page=(\d+)", html)]
    return posts, max(pages, default=1)


class Mineo(Source):
    name = "mineo"
    label = "mineo community: みんなで写真加工 (Japan)"
    base_url = "https://king.mineo.jp"
    reports = {"7093": "/reports/7093"}

    def discover(self, **_) -> None:
        self.check_runnable()
        for rid, path in self.reports.items():
            self.store.upsert_thread({"thread_id": rid, "url": self.base_url + path, "section": "photo-retouch",
                                      "title": "みんなで写真加工(ﾌｫﾄﾚﾀｯﾁ)～王国現像所～"})
        self.store.commit()
        log.info("[%s] %s board thread(s) registered", self.name, len(self.reports))

    def fetch_thread(self, row):
        posts, page, last = [], 1, 1
        while page <= last:
            resp = self.get_page(f"{row['url']}?page={page}")
            if resp.status_code != 200:
                raise LookupError(f"http_{resp.status_code}")
            page_posts, last = parse_report_page(resp.text, self.base_url, include_opening=(page == 1))
            self.save_raw(f"threads/{row['thread_id']}/page-{page}.html.gz", resp.text)
            posts.extend(page_posts)
            log.info("[%s] page %s/%s: %s comments", self.name, page, last, len(page_posts))
            page += 1
        for p in posts:
            p["author_hash"] = self.hash_name(p.pop("author"))
        return posts, last, {"note": "requests and edits share one thread; pair them in Part II via replies"}

    def probe_targets(self):
        return [("thread", self.base_url + "/reports/7093"),
                ("image_fullsize", "https://img.king.mineo.jp/system/content_comment_images/images/000/000/027/236/fullsize_image.jpg")]
