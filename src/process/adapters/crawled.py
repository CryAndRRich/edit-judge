"""Adapter for every crawler database (raw/<source>/crawl.sqlite) whose threads follow the
request / edit roles assigned by the crawler (all sources except mineo, which has its own adapter)."""

import sqlite3
from collections.abc import Iterator

from ...paths import DATA_DIR, RAW_DIR

from ..model import Candidate, Edit

# Sources whose threads already distinguish the request from community edits.
SOURCES = ("dprevived", "voz", "fotokto", "photoshopgurus", "zhopped", "reddit", "realedit", "photoshopcontest", "b3ta")
NOT_TRAINABLE = {"voz"}  # robots.txt ai-train=no
USABLE = ("ok", "ok_unverified")


def _request_text(title: str | None, body: str | None) -> str:
    parts = [p.strip() for p in (title, body) if p and p.strip()]
    return "\n\n".join(parts)


def crawled_sources() -> Iterator[Candidate]:
    for name in SOURCES:
        path = RAW_DIR / name / "crawl.sqlite"
        if not path.exists():
            continue
        db = sqlite3.connect(path)
        db.row_factory = sqlite3.Row
        for t in db.execute("SELECT * FROM threads WHERE fetch_status = 'ok'"):
            posts = db.execute("SELECT * FROM posts WHERE thread_id = ? ORDER BY position", (t["thread_id"],)).fetchall()
            if not posts or posts[0]["role"] != "request":
                continue
            req = posts[0]
            req_images = db.execute("SELECT * FROM images WHERE post_id = ? ORDER BY idx", (req["post_id"],)).fetchall()
            # All request images count toward "exactly one original"; only downloaded ones have files.
            originals = [DATA_DIR / i["local_path"] if i["local_path"] and i["status"] in USABLE else None
                         for i in req_images]
            # Still downloading: an image of this thread has no result yet, so no decision is possible.
            pending = db.execute("""SELECT COUNT(*) FROM images i JOIN posts p ON p.post_id = i.post_id
                                    WHERE p.thread_id = ? AND i.status IS NULL""", (t["thread_id"],)).fetchone()[0]
            cand = Candidate(source=name, key=t["thread_id"], request=_request_text(t["title"], req["text"]),
                             originals=list(originals), trainable=name not in NOT_TRAINABLE,
                             complete=pending == 0)
            for p in posts:
                if p["role"] != "edit":
                    continue
                for i in db.execute("SELECT * FROM images WHERE post_id = ? ORDER BY idx", (p["post_id"],)):
                    if i["local_path"] and i["status"] in USABLE:
                        cand.edits.append(Edit(image=DATA_DIR / i["local_path"], note=p["text"] or None,
                                               ref=f"{p['post_id']}_{i['idx']}"))
            yield cand
        db.close()
