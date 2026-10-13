"""Adapter for the mineo board (raw/mineo): pair requests with edits inside one shared thread.

Requests and edits are both plain comments, with no reply links. Pairing is done by rules,
written to raw/mineo/pairing.csv for the team to review:

1. A *request* is an image comment whose text offers material to edit
   (素材 "material", 元画像 "original image", 無編集 "unedited", どうする "what to do with", ...).
2. Another image comment by a different author is an *edit* when it carries edit wording
   (加工 "processed", してみ "tried", 触らせ "touched", 編集 "edited", こんな感じ "like this", ...)
   or names a requester ("<name>さん").
3. Comments that walk through an app (画面 "screen", タップ "tap", メニュ "menu", 【 headings) are
   tutorial screenshots, not edits; 自分で ("by myself") by someone with their own recent request
   marks a self-edit.
4. An edit goes to the latest earlier request by the user it names; otherwise to the latest
   earlier request by another user posted within 3 days. Edits of one's own request are `self_edit`.

pairing.csv is generated once and then treated as the source of truth: correct it by hand,
or delete it to regenerate. Columns: comment_id, created_at, role (request / edit / self_edit /
other), request_id, rule, author (short hash), n_images, text.
"""

import csv
import gzip
import re
import sqlite3
from collections.abc import Iterator
from datetime import datetime, timedelta

from bs4 import BeautifulSoup

from ...paths import DATA_DIR, RAW_DIR

from ..model import Candidate, Edit

MINEO_DIR = RAW_DIR / "mineo"
PAIRING = MINEO_DIR / "pairing.csv"
REQUEST_CUES = re.compile(r"素材|元画像|元の画像|元写真|無編集|未加工|加工前|どうする|お願い|加工して|いじって|誰か|お題")
EDIT_CUES = re.compile(r"加工|してみ|触らせ|編集|いじっ|仕上|こんな感じ|やってみ|拝借|お借り|レタッチ|補正|変えてみ")
TUTORIAL = re.compile(r"画面|タップ|メニュ|【")   # app walkthrough screenshots ("screen", "tap", "menu", headings)
SELF = re.compile(r"自分で")                       # "by myself"
MENTION = re.compile(r"([^\s>＞、。！!？?「」()（）]{2,20}?)(?:さん|ちゃん|くん|様)")
WINDOW = timedelta(days=3)
FIELDS = ["comment_id", "created_at", "role", "request_id", "rule", "author", "n_images", "text"]


def _display_names() -> dict[str, str]:
    """comment id -> author display name, read from the raw page cache (names are not stored in the DB)."""
    names = {}
    for page in sorted((MINEO_DIR / "raw" / "threads").glob("*/page-*.html.gz")):
        soup = BeautifulSoup(gzip.open(page, "rt", encoding="utf-8").read(), "html.parser")
        for body in soup.select("div[id^='comment-body-']"):
            box = body.find_parent("div", class_="my-4")
            a = box.select_one(".media-body a.text-body") if box else None
            names[body["id"].removeprefix("comment-body-")] = a.get("title") if a else None
    return names


def _matches(mention: str, name: str | None) -> bool:
    return bool(name) and (name == mention or name.startswith(mention) or name.split("@")[0] == mention)


def generate_pairing(db: sqlite3.Connection) -> list[dict]:
    names = _display_names()
    posts = db.execute("""SELECT p.post_id, p.created_at, p.author_hash, p.text,
                                 (SELECT COUNT(*) FROM images i WHERE i.post_id = p.post_id) AS n_images
                          FROM posts p WHERE p.post_id != 'report' ORDER BY p.created_at""").fetchall()
    rows, requests = [], []  # requests: (post_id, time, author_hash, display name)
    for p in posts:
        when = datetime.fromisoformat(p["created_at"]) if p["created_at"] else None
        text = p["text"] or ""
        row = {"comment_id": p["post_id"], "created_at": p["created_at"], "role": "other", "request_id": "",
               "rule": "", "author": (p["author_hash"] or "")[:8], "n_images": p["n_images"],
               "text": re.sub(r"\s+", " ", text)[:120]}
        if p["n_images"]:
            mentioned = [m for m in MENTION.findall(text)]
            if REQUEST_CUES.search(text) and not (EDIT_CUES.search(text) and mentioned):
                row.update(role="request", rule="request_cue")
                requests.append((p["post_id"], when, p["author_hash"], names.get(p["post_id"])))
            elif TUTORIAL.search(text):
                row.update(rule="tutorial_screenshot")
            else:
                target = None
                own = [r for r in requests if r[2] == p["author_hash"] and when and r[1] and when - r[1] <= WINDOW]
                if SELF.search(text) and own:
                    row.update(role="self_edit", request_id=own[-1][0], rule="self_cue")
                    rows.append(row)
                    continue
                for m in mentioned:  # 1) a request by the user this comment names
                    hits = [r for r in requests if _matches(m, r[3]) and r[2] != p["author_hash"]]
                    if hits:
                        target, rule = hits[-1], f"names {m}"
                        break
                if target is None and EDIT_CUES.search(text):  # 2) latest recent request by someone else
                    recent = [r for r in requests if r[2] != p["author_hash"] and when and r[1] and when - r[1] <= WINDOW]
                    if recent:
                        target, rule = recent[-1], "latest_request_within_3_days"
                    if target is None and own:
                        row.update(role="self_edit", request_id=own[-1][0], rule="own_request")
                if target is not None:
                    row.update(role="edit", request_id=target[0], rule=rule)
        rows.append(row)
    return rows


def _load_or_generate(db: sqlite3.Connection, regenerate: bool) -> list[dict]:
    if PAIRING.exists() and not regenerate:
        with open(PAIRING, newline="", encoding="utf-8") as fh:
            return list(csv.DictReader(fh))
    rows = generate_pairing(db)
    with open(PAIRING, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    return rows


def mineo(regenerate_pairing: bool = False) -> Iterator[Candidate]:
    path = MINEO_DIR / "crawl.sqlite"
    if not path.exists():
        return
    db = sqlite3.connect(path)
    db.row_factory = sqlite3.Row
    rows = _load_or_generate(db, regenerate_pairing)

    def files(post_id: str) -> list:
        return [DATA_DIR / i["local_path"] if i["local_path"] and i["status"] in ("ok", "ok_unverified") else None
                for i in db.execute("SELECT * FROM images WHERE post_id = ? ORDER BY idx", (post_id,))]

    texts = {r["post_id"]: r["text"] for r in db.execute("SELECT post_id, text FROM posts")}
    for req in (r for r in rows if r["role"] == "request"):
        cand = Candidate(source="mineo", key=req["comment_id"], request=texts.get(req["comment_id"]) or "",
                         originals=files(req["comment_id"]))
        for e in (r for r in rows if r["role"] == "edit" and r["request_id"] == req["comment_id"]):
            for f in files(e["comment_id"]):
                if f is not None:
                    cand.edits.append(Edit(image=f, note=texts.get(e["comment_id"]) or None, ref=e["comment_id"]))
        yield cand
    db.close()
