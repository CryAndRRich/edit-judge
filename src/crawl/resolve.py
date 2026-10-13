"""Turn image *share* links (imgur pages and albums, Dropbox, Google Drive, ...) into direct file URLs.

Most community edits are posted as links to an image host, not as the file itself.
Album resolution needs one extra request to the album page.
"""

import re

from .net import Blocked, Disallowed, PoliteSession

IMAGE_EXT = re.compile(r"\.(jpe?g|png|gif|webp|bmp|tiff?)(\?.*)?$", re.I)
IMGUR_ALBUM = re.compile(r"https?://(?:www\.|m\.)?imgur\.com/(?:a|gallery)/(\w+)")
IMGUR_PAGE = re.compile(r"https?://(?:www\.|m\.)?imgur\.com/(\w{5,8})(?:[?#].*)?$")
GDRIVE = re.compile(r"drive\.google\.com/(?:file/d/|open\?id=|uc\?(?:export=\w+&)?id=)([\w-]+)")
OG_IMAGE = re.compile(r'<meta[^>]+(?:property|name)="(?:og:image|twitter:image)"[^>]+content="([^"]+)"')


def looks_like_image(url: str) -> bool:
    return bool(IMAGE_EXT.search(url)) or "i.redd.it/" in url or "i.imgur.com/" in url


def resolve(url: str, http: PoliteSession | None = None) -> tuple[str | None, str]:
    """Return (direct_url or None, method). `http` is needed only for album pages."""
    if IMGUR_ALBUM.match(url):
        if http is None:
            return None, "imgur_album_unresolved"
        try:
            page = http.get(url).text
        except (Blocked, Disallowed):
            return None, "imgur_album_blocked"
        m = OG_IMAGE.search(page)
        if not m:
            return None, "imgur_album_unresolved"
        # og:image is a resized copy (suffix h/l/m/...); strip the suffix for the original.
        direct = re.sub(r"https://i\.imgur\.com/(\w{7})[hlmtsb]?\.(\w+).*", r"https://i.imgur.com/\1.\2", m.group(1))
        return direct, "imgur_album_cover"
    m = IMGUR_PAGE.match(url)
    if m:
        return f"https://i.imgur.com/{m.group(1)}.jpg", "imgur_page"
    if "dropbox.com" in url:
        return re.sub(r"[?&]dl=\d", "", url) + ("&" if "?" in url else "?") + "raw=1", "dropbox_raw"
    m = GDRIVE.search(url)
    if m:
        return f"https://drive.usercontent.google.com/download?id={m.group(1)}&export=download", "gdrive"
    return url, "direct"
