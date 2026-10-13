"""SQLite storage shared by all sources (one database file per source).

The database is also the checkpoint: every stage only processes rows it has not
finished, so an interrupted run resumes where it stopped.

IDs are TEXT because sources differ (forum numbers, Reddit base36 IDs, Zhopped codes).
Each post gets a `role` when its thread is fetched:
    request           the original request (first post, or the submission)
    requester_reply   a later post by the requester
    edit              a post by someone else that carries at least one image
    comment           anything else
"""

import json
import shutil
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

SCHEMA_VERSION = 2

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS listing_pages (
    listing     TEXT NOT NULL,      -- e.g. 'free', 'node-75', 'cdx'
    page        TEXT NOT NULL,      -- page number or cursor
    fetched_at  TEXT NOT NULL,
    PRIMARY KEY (listing, page)
);
CREATE TABLE IF NOT EXISTS threads (
    thread_id     TEXT PRIMARY KEY,
    section       TEXT,
    url           TEXT NOT NULL,
    title         TEXT,
    prefix        TEXT,
    author_hash   TEXT,
    created_at    TEXT,
    reply_count   INTEGER,
    view_count    INTEGER,
    last_post_at  TEXT,
    is_sticky     INTEGER NOT NULL DEFAULT 0,
    meta          TEXT,             -- JSON, source-specific (winner, price, ...)
    listed_at     TEXT NOT NULL,
    fetched_at    TEXT,
    num_pages     INTEGER,
    fetch_status  TEXT              -- NULL = not fetched, 'ok', or an error reason
);
CREATE TABLE IF NOT EXISTS posts (
    post_id           TEXT PRIMARY KEY,
    thread_id         TEXT NOT NULL REFERENCES threads(thread_id),
    position          INTEGER NOT NULL,     -- 0 = first post / submission
    role              TEXT,
    author_hash       TEXT,
    created_at        TEXT,
    text              TEXT,
    quoted_post_ids   TEXT,                 -- JSON list
    reaction_count    INTEGER NOT NULL DEFAULT 0,   -- likes / reactions / upvotes
    reactor_hashes    TEXT,                 -- JSON list; complete only if reactors_complete = 1
    reactors_complete INTEGER NOT NULL DEFAULT 1,
    meta              TEXT,
    fetched_at        TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS images (
    post_id        TEXT NOT NULL REFERENCES posts(post_id),
    idx            INTEGER NOT NULL,
    full_url       TEXT,
    thumb_url      TEXT,
    alt            TEXT,
    local_path     TEXT,
    downloaded_url TEXT,
    is_thumbnail   INTEGER,
    sha256         TEXT,
    filesize       INTEGER,
    format         TEXT,
    width          INTEGER,
    height         INTEGER,
    status         TEXT,      -- NULL = pending; ok / ok_unverified / too_small / needs_login / http_* / ...
    fetched_at     TEXT,
    PRIMARY KEY (post_id, idx)
);
CREATE INDEX IF NOT EXISTS posts_thread ON posts(thread_id);
"""


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _assign_role(position: int, author: str | None, op: str | None, has_images: bool) -> str:
    if position == 0:
        return "request"
    if author is not None and author == op:
        return "requester_reply"
    return "edit" if has_images else "comment"


def _migrate_v1(path: Path) -> None:
    """Version 1 (PhotoshopGurus only) used INTEGER ids and no role column. Convert in place,
    keeping a backup next to the database."""
    old = sqlite3.connect(path)
    cols = {r[1]: r[2] for r in old.execute("PRAGMA table_info(threads)")}
    if not cols or cols.get("thread_id") != "INTEGER":
        old.close()
        return
    old.row_factory = sqlite3.Row
    threads = [dict(r) for r in old.execute("SELECT * FROM threads")]
    posts = [dict(r) for r in old.execute("SELECT * FROM posts")]
    images = [dict(r) for r in old.execute("SELECT * FROM images")]
    pages = [dict(r) for r in old.execute("SELECT * FROM listing_pages")]
    old.close()
    shutil.copy2(path, path.with_name(path.stem + "_v1_backup.sqlite"))
    path.unlink()
    db = sqlite3.connect(path)
    db.executescript(SCHEMA)
    for p in pages:
        db.execute("INSERT INTO listing_pages VALUES (?,?,?)", (p["section"], str(p["page"]), p["fetched_at"]))
    for t in threads:
        db.execute("""INSERT INTO threads (thread_id, section, url, title, prefix, author_hash, created_at, reply_count,
                      view_count, last_post_at, is_sticky, listed_at, fetched_at, num_pages, fetch_status)
                      VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                   (str(t["thread_id"]), t["section"], t["url"], t["title"], t["prefix"], t["author_hash"],
                    t["created_at"], t["reply_count"], t["view_count"], t["last_post_at"], t["is_sticky"],
                    t["listed_at"], t["fetched_at"], t["num_pages"], t["fetch_status"]))
    has_img = {(str(i["post_id"])) for i in images}
    op = {str(p["thread_id"]): p["author_hash"] for p in posts if p["position"] == 0}
    for p in posts:
        tid, pid = str(p["thread_id"]), str(p["post_id"])
        db.execute("""INSERT INTO posts (post_id, thread_id, position, role, author_hash, created_at, text,
                      quoted_post_ids, reaction_count, reactor_hashes, reactors_complete, fetched_at)
                      VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                   (pid, tid, p["position"], _assign_role(p["position"], p["author_hash"], op.get(tid), pid in has_img),
                    p["author_hash"], p["created_at"], p["text"],
                    json.dumps([str(q) for q in json.loads(p["quoted_post_ids"] or "[]")]),
                    p["reaction_count"], p["reactor_hashes"], p["reactors_complete"], p["fetched_at"]))
    for i in images:
        i = {**i, "post_id": str(i["post_id"])}
        db.execute(f"INSERT INTO images ({','.join(i)}) VALUES ({','.join('?' * len(i))})", list(i.values()))
    db.execute("INSERT OR REPLACE INTO meta VALUES ('schema_version', ?)", (str(SCHEMA_VERSION),))
    db.commit()
    db.close()


class Store:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            _migrate_v1(path)
        self.db = sqlite3.connect(path)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)
        self.db.execute("INSERT OR IGNORE INTO meta VALUES ('schema_version', ?)", (str(SCHEMA_VERSION),))

    def commit(self) -> None:
        self.db.commit()

    # --- listing -------------------------------------------------------------------
    def listing_page_done(self, listing: str, page: str | int) -> bool:
        return self.db.execute("SELECT 1 FROM listing_pages WHERE listing=? AND page=?",
                               (listing, str(page))).fetchone() is not None

    def mark_listing_page(self, listing: str, page: str | int) -> None:
        self.db.execute("INSERT OR REPLACE INTO listing_pages VALUES (?,?,?)", (listing, str(page), now()))

    def upsert_thread(self, t: dict) -> None:
        """Insert a discovered thread, or refresh its listing fields (fetch state is preserved)."""
        row = {"section": None, "title": None, "prefix": None, "author_hash": None, "created_at": None,
               "reply_count": None, "view_count": None, "last_post_at": None, "is_sticky": 0, **t,
               "thread_id": str(t["thread_id"]), "is_sticky": int(bool(t.get("is_sticky"))),
               "meta": json.dumps(t["meta"]) if t.get("meta") is not None else None, "listed_at": now()}
        self.db.execute(
            """INSERT INTO threads (thread_id, section, url, title, prefix, author_hash, created_at, reply_count,
                                    view_count, last_post_at, is_sticky, meta, listed_at)
               VALUES (:thread_id, :section, :url, :title, :prefix, :author_hash, :created_at, :reply_count,
                       :view_count, :last_post_at, :is_sticky, :meta, :listed_at)
               ON CONFLICT(thread_id) DO UPDATE SET
                   title=COALESCE(excluded.title, title), prefix=COALESCE(excluded.prefix, prefix),
                   reply_count=COALESCE(excluded.reply_count, reply_count),
                   view_count=COALESCE(excluded.view_count, view_count),
                   last_post_at=COALESCE(excluded.last_post_at, last_post_at),
                   is_sticky=excluded.is_sticky, meta=COALESCE(excluded.meta, meta), listed_at=excluded.listed_at""",
            row,
        )

    # --- threads and posts ---------------------------------------------------------
    def threads_to_fetch(self, limit: int | None, refetch: bool, section: str | None) -> list[sqlite3.Row]:
        sql, args = "SELECT * FROM threads WHERE is_sticky = 0", []
        if not refetch:
            sql += " AND fetched_at IS NULL"
        if section:
            sql += " AND section = ?"
            args.append(section)
        sql += " ORDER BY listed_at, rowid"
        if limit:
            sql += f" LIMIT {int(limit)}"
        return self.db.execute(sql, args).fetchall()

    def save_thread(self, thread_id: str, num_pages: int, posts: list[dict], meta: dict | None = None) -> None:
        """Store a fetched thread. `posts` are in display order and already carry hashed authors."""
        stamp = now()
        op = posts[0]["author_hash"] if posts else None
        for position, p in enumerate(posts):
            role = p.get("role") or _assign_role(position, p["author_hash"], op, bool(p["images"]))
            self.db.execute(
                """INSERT OR REPLACE INTO posts VALUES
                   (:post_id, :thread_id, :position, :role, :author_hash, :created_at, :text, :quoted_post_ids,
                    :reaction_count, :reactor_hashes, :reactors_complete, :meta, :fetched_at)""",
                {"post_id": str(p["post_id"]), "thread_id": str(thread_id), "position": position, "role": role,
                 "author_hash": p["author_hash"], "created_at": p.get("created_at"), "text": p.get("text"),
                 "quoted_post_ids": json.dumps([str(q) for q in p.get("quoted_post_ids", [])]),
                 "reaction_count": p.get("reaction_count") or 0,
                 "reactor_hashes": json.dumps(p.get("reactor_hashes", [])),
                 "reactors_complete": int(p.get("reactors_complete", True)),
                 "meta": json.dumps(p["meta"]) if p.get("meta") else None, "fetched_at": stamp},
            )
            for i, img in enumerate(p["images"]):
                self.db.execute(
                    """INSERT INTO images (post_id, idx, full_url, thumb_url, alt) VALUES (?,?,?,?,?)
                       ON CONFLICT(post_id, idx) DO UPDATE SET full_url=excluded.full_url,
                           thumb_url=excluded.thumb_url, alt=excluded.alt""",
                    (str(p["post_id"]), i, img.get("full_url"), img.get("thumb_url"), img.get("alt")),
                )
        sets, args = "fetched_at=?, num_pages=?, fetch_status='ok'", [stamp, num_pages]
        if meta is not None:
            sets += ", meta=?"
            args.append(json.dumps(meta))
        self.db.execute(f"UPDATE threads SET {sets} WHERE thread_id=?", (*args, str(thread_id)))

    def mark_thread_failed(self, thread_id: str, reason: str) -> None:
        self.db.execute("UPDATE threads SET fetched_at=?, fetch_status=? WHERE thread_id=?",
                        (now(), reason, str(thread_id)))

    def posts_needing_reactors(self, limit: int | None) -> list[sqlite3.Row]:
        sql = "SELECT post_id FROM posts WHERE reactors_complete = 0 ORDER BY rowid"
        if limit:
            sql += f" LIMIT {int(limit)}"
        return self.db.execute(sql).fetchall()

    def set_reactors(self, post_id: str, hashes: list[str]) -> None:
        self.db.execute("UPDATE posts SET reactor_hashes=?, reactors_complete=1, reaction_count=? WHERE post_id=?",
                        (json.dumps(hashes), len(hashes), str(post_id)))

    # --- images --------------------------------------------------------------------
    def images_to_download(self, limit: int | None, retry_thumbnails: bool,
                           min_edit_images: int = 0) -> list[sqlite3.Row]:
        """Pending images thread by thread, request images first.

        min_edit_images: only threads with at least this many images in posts whose role is 'edit'."""
        pending = "(i.status IS NULL OR i.status = 'needs_login')"
        if retry_thumbnails:
            pending = f"({pending} OR (i.is_thumbnail = 1 AND i.full_url IS NOT NULL))"
        sql = f"""
            WITH edit_counts AS (
                SELECT p.thread_id, COUNT(*) AS n FROM images i JOIN posts p ON p.post_id = i.post_id
                WHERE p.role = 'edit' GROUP BY p.thread_id)
            SELECT i.*, p.thread_id, p.position, p.role
            FROM images i JOIN posts p ON p.post_id = i.post_id
            LEFT JOIN edit_counts e ON e.thread_id = p.thread_id
            WHERE {pending} AND COALESCE(e.n, 0) >= ?
            ORDER BY p.thread_id, p.position, i.idx"""
        if limit:
            sql += f" LIMIT {int(limit)}"
        return self.db.execute(sql, (min_edit_images,)).fetchall()

    def save_image_result(self, post_id: str, idx: int, result: dict) -> None:
        self.db.execute(
            """UPDATE images SET local_path=:local_path, downloaded_url=:downloaded_url, is_thumbnail=:is_thumbnail,
                   sha256=:sha256, filesize=:filesize, format=:format, width=:width, height=:height,
                   status=:status, fetched_at=:fetched_at WHERE post_id=:post_id AND idx=:idx""",
            {"local_path": None, "downloaded_url": None, "is_thumbnail": None, "sha256": None, "filesize": None,
             "format": None, "width": None, "height": None, **result,
             "fetched_at": now(), "post_id": str(post_id), "idx": idx},
        )

    # --- status --------------------------------------------------------------------
    def status(self) -> dict:
        q = lambda sql: self.db.execute(sql).fetchone()[0]
        return {
            "threads_listed": q("SELECT COUNT(*) FROM threads WHERE is_sticky=0"),
            "threads_fetched": q("SELECT COUNT(*) FROM threads WHERE fetch_status='ok'"),
            "threads_failed": q("SELECT COUNT(*) FROM threads WHERE fetch_status IS NOT NULL AND fetch_status!='ok'"),
            "posts": q("SELECT COUNT(*) FROM posts"),
            "edit_posts": q("SELECT COUNT(*) FROM posts WHERE role='edit'"),
            "threads_with_2plus_edit_images": q(
                """SELECT COUNT(*) FROM (SELECT p.thread_id FROM images i JOIN posts p ON p.post_id=i.post_id
                   WHERE p.role='edit' GROUP BY p.thread_id HAVING COUNT(*) >= 2)"""),
            "images": q("SELECT COUNT(*) FROM images"),
            "images_ok": q("SELECT COUNT(*) FROM images WHERE status IN ('ok','ok_unverified')"),
            "images_ok_full_size": q("SELECT COUNT(*) FROM images WHERE status IN ('ok','ok_unverified') "
                                     "AND COALESCE(is_thumbnail,0)=0"),
            "images_pending": q("SELECT COUNT(*) FROM images WHERE status IS NULL OR status='needs_login'"),
        }
