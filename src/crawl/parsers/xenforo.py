"""Parsers for XenForo 2 forum pages (PhotoshopGurus, and also VOZ).

Pure functions: HTML in, plain dicts out. Usernames are returned raw; callers hash them.
"""

import re
from urllib.parse import urljoin

from bs4 import BeautifulSoup, Tag

_THREAD_ID = re.compile(r"\.(\d+)/?$")
_POST_ID = re.compile(r"post-(\d+)")
_OTHERS = re.compile(r"and\s+([\d,]+)\s+others?", re.I)


def _soup(html: str) -> BeautifulSoup:
    return BeautifulSoup(html, "html.parser")


def _text(tag: Tag | None) -> str:
    return re.sub(r"\s+", " ", tag.get_text(" ", strip=True)).strip() if tag else ""


def _count(value: str) -> int | None:
    """'1,234' -> 1234, '29K' -> 29000, '1.2M' -> 1200000."""
    value = value.strip().replace(",", "")
    m = re.fullmatch(r"([\d.]+)\s*([KkMm]?)", value)
    if not m:
        return None
    return int(float(m.group(1)) * {"": 1, "k": 1_000, "m": 1_000_000}[m.group(2).lower()])


def last_page(soup: BeautifulSoup) -> int:
    pages = [int(a.get_text(strip=True)) for a in soup.select(".pageNav-page a") if a.get_text(strip=True).isdigit()]
    return max(pages, default=1)


def is_logged_in(html: str) -> bool:
    m = re.search(r'data-logged-in="(\w+)"', html)
    return bool(m and m.group(1) == "true")


# --- forum listing -------------------------------------------------------------------

def parse_listing(html: str, base_url: str) -> tuple[list[dict], int]:
    """Return (threads on this page, last page number of the listing)."""
    soup = _soup(html)
    threads = []
    for item in soup.select("div.structItem--thread"):
        link = item.select_one(".structItem-title a[data-tp-primary]") or item.select_one(".structItem-title a[href*='/threads/']")
        if not link:
            continue
        url = urljoin(base_url, link["href"])
        m = _THREAD_ID.search(link["href"])
        if not m:
            continue
        prefix = item.select_one(".structItem-title .label")
        start = item.select_one(".structItem-startDate time")
        latest = item.select_one("time.structItem-latestDate")
        meta = {dt.get_text(strip=True): dd.get_text(strip=True)
                for dt, dd in zip(item.select(".structItem-cell--meta dt"), item.select(".structItem-cell--meta dd"))}
        threads.append({
            "thread_id": int(m.group(1)),
            "url": url,
            "title": _text(link),
            "prefix": _text(prefix) or None,
            "author": item.get("data-author"),
            "created_at": start.get("datetime") if start else None,
            "reply_count": _count(meta.get("Replies", "")),
            "view_count": _count(meta.get("Views", "")),
            "last_post_at": latest.get("datetime") if latest else None,
            "is_sticky": bool(item.select_one(".structItem-status--sticky")),
        })
    return threads, last_page(soup)


# --- thread pages --------------------------------------------------------------------

def _images(article: Tag, body: Tag | None, base_url: str) -> list[dict]:
    """Embedded images plus the attachment list under the post, de-duplicated by URL."""
    found: list[dict] = []
    seen: set[str] = set()

    def add(full: str | None, thumb: str | None, alt: str | None) -> None:
        key = full or thumb
        if key and key not in seen:
            seen.add(key)
            found.append({"full_url": full, "thumb_url": thumb, "alt": alt})

    for img in (body.select("img.bbImage") if body else []):
        src = img.get("data-src") or img.get("src") or ""
        if src.startswith("data:"):
            src = img.get("data-src", "")
        src = urljoin(base_url, src) if src else ""
        parent = img.find_parent("a")
        href = urljoin(base_url, parent["href"]) if parent and parent.get("href") else ""
        if "/attachments/" in href:
            add(href, src or None, img.get("alt"))          # full-size link wrapping a thumbnail
        elif "/attachments/" in src and "/data/attachments/" not in src:
            add(src, None, img.get("alt"))                  # lazy-loaded full-size attachment (VOZ)
        elif src:
            add(src, None, img.get("alt"))                  # external image (imgur, etc.)
    for a in article.select(".attachmentList a.file-preview, .attachmentList .file-name a"):
        href = urljoin(base_url, a.get("href", ""))
        thumb = a.select_one("img")
        add(href, urljoin(base_url, thumb["src"]) if thumb and thumb.get("src") else None,
            thumb.get("alt") if thumb else None)
    return found


def parse_thread_page(html: str, base_url: str) -> tuple[list[dict], int]:
    """Return (posts on this page, last page number of the thread)."""
    soup = _soup(html)
    posts = []
    for article in soup.select("article.message--post"):
        m = _POST_ID.search(article.get("data-content", ""))
        if not m:
            continue
        body = article.select_one(".message-body .bbWrapper")
        quoted = []
        if body:
            for q in body.select("blockquote.bbCodeBlock--quote"):
                src = q.get("data-source", "")
                qm = re.search(r"post:\s*(\d+)", src)
                if qm:
                    quoted.append(int(qm.group(1)))
        images = _images(article, body, base_url)
        if body:
            for q in body.select("blockquote"):
                q.decompose()                                # drop quoted text from the post's own text
        posted = article.select_one(".message-attribution-main time") or article.select_one("time.u-dt")
        bar = article.select_one(".reactionsBar-link")
        shown = [b.get_text(strip=True) for b in bar.select("bdi")] if bar else []
        others = _OTHERS.search(_text(bar)) if bar else None
        count_meta = article.select_one('meta[itemprop="userInteractionCount"]')
        reaction_count = int(count_meta["content"]) if count_meta and count_meta.get("content", "").isdigit() else 0
        reaction_count = max(reaction_count, len(shown) + (int(others.group(1).replace(",", "")) if others else 0))
        posts.append({
            "post_id": int(m.group(1)),
            "author": article.get("data-author"),
            "created_at": posted.get("datetime") if posted else None,
            "text": _text(body),
            "quoted_post_ids": quoted,
            "reaction_count": reaction_count,
            "reactors_shown": shown,
            # The bar lists at most a few names; the rest need the post's reactions page.
            "reactors_complete": len(shown) >= reaction_count,
            "images": images,
        })
    return posts, last_page(soup)


def parse_reactions(html: str) -> list[str]:
    """Usernames on a post's /reactions overlay page."""
    soup = _soup(html)
    names = []
    for row in soup.select(".block-row"):
        user = row.select_one(".username")
        if user:
            names.append(user.get_text(strip=True))
    return names
