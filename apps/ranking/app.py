"""Edit ranking app: rank the community edits of each sample from best to worst.

Put chunk zip files in chunks/, run `python app.py`, and send back the files in results/.
Each sample is written to disk as soon as you move on, so closing the app never loses finished work.

Images are shown from smaller copies in chunks/<chunk>/_display/ (made once, in the background, a few
samples ahead); the full-size file opens with Zoom. Ranking clicks are handled in the browser, and the
ranking is sent to the app when you move to another sample.
"""

import json
import os
import re
import threading
import time
import zipfile
from concurrent.futures import Future, ThreadPoolExecutor, wait as futures_wait
from datetime import datetime, timezone
from functools import partial
from html import escape
from pathlib import Path
from urllib.parse import quote

import gradio as gr
from PIL import Image, ImageOps

APP_VERSION = "1.3"
HERE = Path(__file__).resolve().parent
CHUNKS = HERE / "chunks"
RESULTS = HERE / "results"
TAG = re.compile(r"^\s*(?:\[[^\]\n]{1,24}\]\s*)+")   # leading flair tags such as "[Specific]"
DISPLAY_SIDE = 1600        # longest side of the copies shown in the browser
KEEP_BYTES = 1_500_000     # images at most DISPLAY_SIDE wide and this small are shown as they are
AHEAD = 4                  # samples prepared in advance
# One frame colour per rank (1 to 10); ranks 11 and lower share the last one. Red is kept for the original.
RANK_COLORS = ["#d4af37", "#a8a9ad", "#8c5a2b", "#16a34a", "#2563eb",
               "#9333ea", "#db2777", "#0891b2", "#ea580c", "#84cc16", "#64748b"]
RANK_NAMES = ["gold", "silver", "bronze", "green", "blue", "purple", "pink", "cyan", "orange", "lime"]

GUIDE = {
    "What you do": """
For every sample you see **the original photo** (red frame), **the edit request** written by the
person who posted it, and **2 to 14 edits** made by other people. Rank the edits from best to worst.

Your work is saved to `results/<chunk>__<your name>.json` every time you move to another sample. You can close
the app at any time: open the same chunk with the same name and you continue where you stopped.
The sample on screen is saved when you leave it (Previous, Next, a number under Detail, or Back to chunk
list), so move on before closing the app. When a chunk is finished, send that file to the coordinator.
""",
    "How to rank": f"""
- **Click an edit** without a rank: it gets the best rank still free (the first click is rank 1).
- **Click a ranked edit again**: it loses its rank, and the edits ranked below it move up one place.
- Every rank has its own frame colour: {", ".join(f"{i + 1} = {c}" for i, c in enumerate(RANK_NAMES))},
  and slate for rank 11 and lower. The rank is also written above the image.
- When two edits are left, clicking one of them ranks the other one last, so a sample with two edits
  needs one click.
- **Zoom** (top right of an image, or a click on the original) shows the full-size image. Inside it, the left
  and right arrow keys move between the images of the sample, and Esc or a click closes it.
- Move with **< Previous** and **Next >**, the left and right arrow keys, or by clicking a number under
  **Detail**. Leaving a sample saves its ranking. A sample left with no complete ranking and no skipping
  reason stays unranked, and you can come back to it later.
- **Detail** shows every sample of the chunk: grey = not done yet, green = ranked, yellow = skipped.
- **Back to chunk list** saves the sample on screen and lets you open another chunk.
- Ties are not allowed: always decide which edit is better, even when they are close.
""",
    "What \"better\" means, in order of importance": """
1. **Does what was asked.** Read the request carefully; an edit that ignores it loses, even if it looks nice.
2. **Looks natural and well made.** No visible artefacts, halos, smudges, odd colours, or distorted faces.
3. **Keeps the rest of the photo.** Parts the request did not mention should stay as they were.
""",
    "Skipping a sample": """
Skip only when a sample cannot be ranked fairly: the request is unclear or is not an edit request,
the edits are jokes or unrelated, images are broken or the original is wrong, or the content is
inappropriate.

Write the reason in **Skipping reason** and move on. While a reason is written, edits cannot be
ranked; while any edit is ranked, no reason can be written (click the ranked edits again to clear them).
""",
}

CSS = """
.progress-wrap {height: 10px; background: var(--border-color-primary); border-radius: 999px; overflow: hidden;}
.progress-wrap {display: flex;}
.pg-ranked {height: 100%; background: #22c55e;}
.pg-skipped {height: 100%; background: #eab308;}
.sq-grid {display: flex; flex-wrap: wrap; gap: 6px; padding: 4px 0;}
.sq {width: 44px; height: 32px; border-radius: 8px; border: 1px solid rgba(0, 0, 0, 0.15); font-size: 0.8rem;
     font-weight: 600; cursor: pointer; color: #1f2937; display: inline-flex; align-items: center;
     justify-content: center; padding: 0 !important; margin: 0; line-height: 1; text-align: center;}
.sq.todo {background: #e5e7eb;}
.sq.ranked {background: #22c55e; color: #ffffff;}
.sq.skipped {background: #eab308; color: #1f2937;}
.sq.here {outline: 3px solid #4f46e5; outline-offset: 1px;}
.sq:hover {filter: brightness(0.92);}
.progress-text {font-size: 0.9rem; margin-top: 6px; color: var(--body-text-color-subdued);}
.sample-header {align-self: center;}
.status-line p {color: var(--body-text-color-subdued);}
.inline-row {align-items: center !important; flex-wrap: nowrap !important;}
.inline-label {flex-grow: 0 !important; min-width: max-content !important;}
.inline-label p {margin: 0 !important; font-weight: 600; white-space: nowrap;}
.guide-acc {padding: 0 !important; overflow: hidden; border: none !important; margin-bottom: 6px;}
.guide-acc > .label-wrap {background: #2563eb !important; color: #ffffff !important; padding: 12px 16px !important;
                          font-weight: 600; margin: 0 !important;}
.guide-acc > .label-wrap * {color: #ffffff !important;}
.guide-acc > .label-wrap + div {background: #1e293b; padding: 6px 18px 10px;}
.guide-acc > .label-wrap + div * {color: #e2e8f0 !important; background: transparent !important; border-color: transparent !important;}
.guide-acc > .label-wrap + div code {background: #334155 !important; padding: 1px 5px; border-radius: 4px;}

.gal {position: relative;}
.gal-hint {font-size: 0.9rem; min-height: 1.4em; margin: 0 0 8px 2px; color: var(--body-text-color-subdued);}
.gal-hint.warn {color: #dc2626; font-weight: 600;}
.gal-grid {display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 14px;}
@media (max-width: 900px) {.gal-grid {grid-template-columns: repeat(2, minmax(0, 1fr));}}
.gal.loading .gal-grid {opacity: 0.25; pointer-events: none;}
.gal.loading::after {content: "Loading the next sample..."; position: absolute; top: 40px; left: 0; right: 0;
                     text-align: center; font-weight: 600; color: var(--body-text-color);}
.card {cursor: pointer; user-select: none; min-width: 0;}
.card.original {cursor: zoom-in;}
.card-label {font-size: 0.85rem; font-weight: 600; margin: 0 0 4px 2px; color: var(--body-text-color-subdued);
             white-space: nowrap; overflow: hidden; text-overflow: ellipsis;}
.frame {position: relative; height: 320px; border: 5px solid var(--border-color-primary); border-radius: 12px;
        overflow: hidden; background: var(--background-fill-secondary); display: flex; align-items: center;
        justify-content: center;}
.frame::before {content: "Loading..."; position: absolute; font-size: 0.85rem; color: var(--body-text-color-subdued);}
.frame.broken::before {content: "This image could not be loaded";}
.frame img {position: relative; width: 100%; height: 100%; object-fit: contain; margin: 0 !important;
            border-radius: 0 !important; opacity: 0; transition: opacity 0.12s;}
.frame img.ready {opacity: 1;}
.card.original .frame {border-color: #dc2626;}
.card.unranked:hover .frame {border-color: #94a3b8;}
.zoom {position: absolute; top: 8px; right: 8px; z-index: 2; padding: 3px 9px !important; border-radius: 6px;
       border: none; background: rgba(15, 23, 42, 0.72); color: #ffffff; font-size: 0.75rem; cursor: zoom-in;}
.zoom:hover {background: rgba(15, 23, 42, 0.92);}
.preload {display: none;}
""" + "".join(
    f".card.rank-{k} .frame {{border-color: {c};}}\n"
    for k, c in zip([*map(str, range(1, 11)), "x"], RANK_COLORS))


# Everything that runs in the browser: ranking clicks, the full-size viewer, and the arrow keys.
HEAD = """
<style>
.lightbox {position: fixed; inset: 0; z-index: 10000; background: rgba(0, 0, 0, 0.9); display: none;
           flex-direction: column; align-items: center; justify-content: center; cursor: zoom-out;}
.lightbox.open {display: flex;}
.lightbox img {max-width: 96vw; max-height: 88vh; object-fit: contain;}
.lightbox .lb-cap {color: #e5e7eb; margin-top: 12px; font: 600 0.95rem system-ui, sans-serif;}
</style>
<script>
(() => {
  const ER = window.editRank = {};
  const gal = () => document.querySelector('#gallery .gal');
  const reasonBox = () => document.querySelector('#reason-box textarea');
  const orderOf = (g) => JSON.parse(g.dataset.order || '[]');

  function hint(g, text, warn) {
    const h = g.querySelector('.gal-hint');
    if (h) { h.textContent = text; h.classList.toggle('warn', !!warn); }
  }

  function paint(g) {
    const o = orderOf(g), n = +g.dataset.n;
    g.querySelectorAll('.card.edit').forEach((c) => {
      const r = o.indexOf(+c.dataset.i) + 1;
      c.className = 'card edit ' + (r ? 'ranked rank-' + (r <= 10 ? r : 'x') : 'unranked');
      c.querySelector('.card-label').textContent = r ? 'Rank ' + r : 'Not ranked (click to rank)';
    });
    const box = reasonBox();
    if (box) box.disabled = o.length > 0;
    hint(g, o.length === n ? 'Ranking complete. Move to another sample to save it.'
         : o.length ? 'Ranked ' + o.length + ' of ' + n + '. Click the next best edit.'
         : 'Click the edits from best to worst.');
  }

  // A new sample arrives as new HTML: set it up once.
  function init() {
    document.querySelectorAll('.gal:not([data-ready])').forEach((g) => {
      g.dataset.ready = '1';
      g.querySelectorAll('img').forEach((im) => { if (im.complete && im.naturalWidth) im.classList.add('ready'); });
      paint(g);
    });
  }
  new MutationObserver(init).observe(document.documentElement, {childList: true, subtree: true});
  document.addEventListener('load', (e) => {
    if (e.target.tagName === 'IMG') e.target.classList.add('ready');
  }, true);
  document.addEventListener('error', (e) => {
    const f = e.target.tagName === 'IMG' && e.target.closest('.frame');
    if (f) f.classList.add('broken');
  }, true);

  // Full-size viewer.
  let lb = null, items = [], at = 0;
  function show() {
    if (!lb) {
      lb = document.createElement('div');
      lb.className = 'lightbox';
      lb.innerHTML = '<img alt=""><div class="lb-cap"></div>';
      lb.addEventListener('click', () => lb.classList.remove('open'));
      document.body.appendChild(lb);
    }
    const c = items[at], img = lb.querySelector('img');
    img.src = c.querySelector('.frame img').src;           // the smaller copy shows at once
    if (c.dataset.full && c.dataset.full !== img.src) {
      const full = new Image();
      full.onload = () => { if (items[at] === c && lb.classList.contains('open')) img.src = full.src; };
      full.src = c.dataset.full;
    }
    lb.querySelector('.lb-cap').textContent = c.querySelector('.card-label').textContent + '   (' + (at + 1)
      + ' of ' + items.length + ', left and right arrows to browse, Esc or click to close)';
    lb.classList.add('open');
  }
  function openViewer(card) {
    items = [...card.closest('.gal').querySelectorAll('.card')];
    at = items.indexOf(card);
    show();
  }

  document.addEventListener('click', (e) => {
    const card = e.target.closest && e.target.closest('#gallery .card');
    if (!card) return;
    const g = card.closest('.gal');
    if (e.target.closest('.zoom') || card.classList.contains('original')) { openViewer(card); return; }
    if (g.classList.contains('loading')) return;
    const box = reasonBox();
    if (box && box.value.trim()) { hint(g, 'Clear the skipping reason before ranking edits.', true); return; }
    const o = orderOf(g), i = +card.dataset.i, n = +g.dataset.n, k = o.indexOf(i);
    if (k >= 0) {
      o.splice(k, 1);                                       // lower ranks move up by one
    } else {
      o.push(i);
      if (o.length === n - 1) for (let j = 0; j < n; j++) if (!o.includes(j)) o.push(j);  // the last one left
    }
    g.dataset.order = JSON.stringify(o);
    paint(g);
  });

  // Called before every move to another sample: dim the current one and hand the ranking to the app.
  ER.leave = () => {
    const g = gal();
    if (!g) return '[]';
    g.classList.add('loading');
    return g.dataset.order || '[]';
  };
  ER.nav = (...args) => [args[0], args[1], ER.leave()];

  document.addEventListener('keydown', (e) => {
    if (lb && lb.classList.contains('open')) {
      const step = {ArrowLeft: -1, ArrowRight: 1}[e.key];
      if (e.key === 'Escape') lb.classList.remove('open');
      else if (step) { at = (at + step + items.length) % items.length; show(); }
      else return;
      e.preventDefault();
      return;
    }
    const t = e.target;
    if (t && (t.tagName === 'INPUT' || t.tagName === 'TEXTAREA' || t.isContentEditable)) return;
    if (e.metaKey || e.ctrlKey || e.altKey) return;
    const id = {ArrowLeft: 'prev-btn', ArrowRight: 'next-btn'}[e.key];
    if (!id) return;
    let el = document.getElementById(id);
    if (el && el.tagName !== 'BUTTON') el = el.querySelector('button') || el;
    if (el && el.offsetParent !== null) { e.preventDefault(); el.click(); }
  });
})();
</script>
"""

# Moves (buttons, Enter in the reason box) send the ranking kept in the browser as the third input.
NAV_JS = "(...args) => window.editRank.nav(...args)"

# Clicking a numbered square in Detail opens that sample. The listener sits on the component, so it
# keeps working after the squares are re-rendered.
GRID_JS = """
element.addEventListener('click', (e) => {
  const b = e.target.closest('[data-idx]');
  if (b) trigger('click', {idx: parseInt(b.dataset.idx, 10), order: window.editRank.leave()});
});
"""


# --- chunks and results on disk ----------------------------------------------------------------
def find_chunks() -> dict[str, Path]:
    """Chunk folders in chunks/; zip files are extracted the first time they are seen."""
    CHUNKS.mkdir(exist_ok=True)
    for archive in sorted(CHUNKS.glob("*.zip")):
        if not (CHUNKS / archive.stem / "chunk.json").exists():
            print(f"Extracting {archive.name} ...", flush=True)
            with zipfile.ZipFile(archive) as zf:
                zf.extractall(CHUNKS)
    return {p.parent.name: p.parent for p in sorted(CHUNKS.glob("*/chunk.json"))}


def safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", name.strip()).strip("_")


def result_path(chunk: str, annotator: str) -> Path:
    return RESULTS / f"{chunk}__{safe_name(annotator)}.json"


def read_records(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))["records"] if path.exists() else {}


def write_record(sess: dict, sample_id: str, record: dict | None) -> None:
    """Merge one record (None removes it) into the result file and replace the file atomically."""
    path = result_path(sess["chunk"], sess["annotator"])
    records = read_records(path)
    if record is None:
        records.pop(sample_id, None)
    else:
        records[sample_id] = record
    sess["records"] = records
    data = {"app_version": APP_VERSION, "version": sess["version"], "chunk": sess["chunk"],
            "annotator": sess["annotator"], "updated_at": now(), "n_samples": len(sess["samples"]),
            "n_ranked": sum(not r["skipped"] for r in records.values()),
            "n_skipped": sum(r["skipped"] for r in records.values()), "records": records}
    RESULTS.mkdir(exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, path)


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# --- display copies ----------------------------------------------------------------------------
# Three pools so that urgent work never waits behind work for later: the sample on screen, the next few
# samples, and the rest of the chunk.
_now = ThreadPoolExecutor(max_workers=4)
_soon = ThreadPoolExecutor(max_workers=2)
_later = ThreadPoolExecutor(max_workers=1)   # leaves CPU for the browser on slow laptops
_jobs: dict[Path, tuple[Future, int]] = {}
_jobs_lock = threading.Lock()


def make_display(src: Path, out_base: Path) -> Path:
    """A copy of `src` scaled to DISPLAY_SIDE (JPEG, or PNG when the image has transparency). Images that
    are already small, GIFs, and files Pillow cannot read are shown as they are."""
    for ext in (".jpg", ".png"):
        if out_base.with_suffix(ext).exists():
            return out_base.with_suffix(ext)
    try:
        with Image.open(src) as im:
            if src.suffix.lower() == ".gif" or (max(im.size) <= DISPLAY_SIDE and src.stat().st_size <= KEEP_BYTES):
                return src
            scale = DISPLAY_SIDE / max(im.size)
            im.draft("RGB", (round(im.width * scale), round(im.height * scale)))   # JPEG: decode at 1/2, 1/4 or 1/8
            img = ImageOps.exif_transpose(im)
            alpha = img.mode in ("RGBA", "LA", "PA") or (img.mode == "P" and "transparency" in img.info)
            img = img.convert("RGBA" if alpha else "RGB")
            img.thumbnail((DISPLAY_SIDE, DISPLAY_SIDE), Image.Resampling.LANCZOS)
            out = out_base.with_suffix(".png" if alpha else ".jpg")
            out.parent.mkdir(parents=True, exist_ok=True)
            tmp = out.with_name(out.name + ".tmp")
            img.save(tmp, "PNG" if alpha else "JPEG", quality=85)
            os.replace(tmp, out)
            return out
    except Exception:
        return src


def queue_display(base: Path, rel: str, level: int) -> Future:
    """The job making the display copy of base/rel in pool `level` (0 = now, 1 = soon, 2 = later).
    A job still waiting in a slower pool moves to the faster one."""
    src = base / rel
    with _jobs_lock:
        fut, old = _jobs.get(src, (None, None))
        if fut is None or (level < old and fut.cancel()):
            fut = (_now, _soon, _later)[level].submit(make_display, src, base / "_display" / rel)
            _jobs[src] = (fut, level)
        return fut


def display_files(base: Path, rels: list[str], wait: float = 0.6) -> list[Path]:
    """Display copies for the sample on screen, made in parallel if they are not ready yet. Copies not made
    within `wait` seconds are replaced by the full-size file, so opening a sample never waits long."""
    jobs = [queue_display(base, rel, 0) for rel in rels]
    futures_wait(jobs, timeout=wait)
    return [f.result() if f.done() else base / rel for f, rel in zip(jobs, rels)]


def ready_display_file(base: Path, rel: str) -> Path | None:
    """The display copy if it is already made, else None (no waiting)."""
    fut = _jobs.get(base / rel, (None, None))[0]
    return fut.result() if fut is not None and fut.done() and not fut.cancelled() else None


def prepare(sess: dict) -> None:
    """Make the display copies of the next samples (and the previous one) in the background."""
    base, n = Path(sess["dir"]), len(sess["samples"])
    for step in [*range(1, AHEAD + 1), -1]:
        for rel in images(sess["samples"][(sess["idx"] + step) % n]):
            queue_display(base, rel, 1)


def prepare_chunk(sess: dict) -> None:
    """Make the display copies of the whole chunk in the background, starting at the sample on screen."""
    base, n = Path(sess["dir"]), len(sess["samples"])
    with _jobs_lock:   # drop waiting jobs of a chunk opened before
        for src, (fut, level) in list(_jobs.items()):
            if level == 2 and fut.cancel():
                del _jobs[src]
    for step in range(n):
        for rel in images(sess["samples"][(sess["idx"] + step) % n]):
            queue_display(base, rel, 2)


def images(s: dict) -> list[str]:
    return [s["original"], *(e["image"] for e in s["edits"])]


def file_url(path: Path) -> str:
    return "/gradio_api/file=" + quote(path.resolve().as_posix(), safe="/:")


# --- rendering ---------------------------------------------------------------------------------
def current(sess: dict) -> dict:
    return sess["samples"][sess["idx"]]


def counts(sess: dict) -> tuple[int, int, int]:
    total, recs = len(sess["samples"]), sess["records"]
    skipped = sum(r["skipped"] for r in recs.values())
    return len(recs) - skipped, skipped, total - len(recs)


def progress_html(sess: dict) -> str:
    ranked, skipped, left = counts(sess)
    total = ranked + skipped + left
    pct = [100 * x / total if total else 0 for x in (ranked, skipped, left)]
    return (f"<div class='progress-wrap'><div class='pg-ranked' style='width:{pct[0]:.2f}%'></div>"
            f"<div class='pg-skipped' style='width:{pct[1]:.2f}%'></div></div>"
            f"<div class='progress-text'>Ranked <b>{ranked}</b> ({pct[0]:.0f}%), skipped <b>{skipped}</b> "
            f"({pct[1]:.0f}%), unranked <b>{left}</b> ({pct[2]:.0f}%) of {total} | chunk <b>{sess['chunk']}</b> "
            f"| annotator <b>{sess['annotator']}</b></div>")


def grid_html(sess: dict) -> str:
    """One numbered square per sample; clicking a square opens that sample (see GRID_JS)."""
    cells = []
    for i, s in enumerate(sess["samples"]):
        rec = sess["records"].get(s["id"])
        state = "skipped" if rec and rec["skipped"] else "ranked" if rec else "todo"
        here = " here" if i == sess["idx"] else ""
        cells.append(f"<button class='sq {state}{here}' data-idx='{i}'>{i + 1}</button>")
    return f"<div class='sq-grid'>{''.join(cells)}</div>"


def card_html(kind: str, label: str, shown: Path, full: Path, i: int | None = None) -> str:
    data = f" data-i='{i}'" if i is not None else ""
    return (f"<div class='card {kind}'{data} data-full='{escape(file_url(full))}'>"
            f"<div class='card-label'>{label}</div><div class='frame'>"
            f"<img src='{escape(file_url(shown))}' alt='' draggable='false'>"
            f"<button class='zoom' type='button'>Zoom</button></div></div>")


def gallery_html(sess: dict) -> str:
    """The original and the edits of the sample on screen. Every sample gets new image elements, so an
    image of the previous sample is never shown while the new one loads. The next sample's images are
    loaded in advance by hidden image elements."""
    s, base = current(sess), Path(sess["dir"])
    shown = display_files(base, images(s))
    cards = [card_html("original", "Original", shown[0], base / s["original"])]
    for i, e in enumerate(s["edits"]):
        cards.append(card_html("edit unranked", "Not ranked (click to rank)", shown[i + 1], base / e["image"], i))
    nxt = sess["samples"][(sess["idx"] + 1) % len(sess["samples"])]
    early = [p for p in (ready_display_file(base, rel) for rel in images(nxt)) if p]
    preload = "".join(f"<img src='{escape(file_url(p))}' alt=''>" for p in early)
    return (f"<div class='gal' data-n='{len(s['edits'])}' data-order='{json.dumps(sess['order'])}' "
            f"data-t='{time.monotonic():.3f}'><div class='gal-hint'></div>"
            f"<div class='gal-grid'>{''.join(cards)}</div><div class='preload'>{preload}</div></div>")


def full_outputs(sess: dict, status: str = "") -> list:
    s = current(sess)
    rec = sess["records"].get(s["id"])
    if not status and rec:
        status = ("Skipped earlier. Clear the reason and rank the edits to replace the skip." if rec["skipped"]
                  else "Already ranked. Changes are saved when you move to another sample.")
    header = f"#### Sample {sess['idx'] + 1} of {len(sess['samples'])} (ID {s['id']})"
    request = TAG.sub("", s["request"]).strip() or "(empty request)"
    reason = rec["skip_reason"] if rec and rec["skipped"] else ""
    gallery = gallery_html(sess)
    prepare(sess)
    return [gr.update(visible=False), gr.update(visible=True), progress_html(sess), grid_html(sess), header,
            request, gallery, reason, status]


def open_sample(sess: dict, idx: int) -> None:
    sess["idx"] = max(0, min(idx, len(sess["samples"]) - 1))
    s = current(sess)
    rec = sess["records"].get(s["id"])
    keys = [e["key"] for e in s["edits"]]
    sess["order"] = [keys.index(k) for k in rec["ranking"]] if rec and not rec["skipped"] else []
    sess["t0"] = time.monotonic()


def next_unfinished(sess: dict, after: int) -> int | None:
    n = len(sess["samples"])
    for step in range(1, n + 1):
        i = (after + step) % n
        if sess["samples"][i]["id"] not in sess["records"]:
            return i
    return None


def parse_order(raw, n: int) -> list[int] | None:
    """The ranking sent by the browser (edit positions, best first), or None if it is not valid."""
    try:
        order = json.loads(raw) if isinstance(raw, str) else raw
    except ValueError:
        return None
    if not isinstance(order, list) or not all(isinstance(i, int) and 0 <= i < n for i in order):
        return None
    return order if len(set(order)) == len(order) else None


# --- event handlers ----------------------------------------------------------------------------
def chunk_choices(annotator: str) -> list[tuple[str, str]]:
    """Chunk names, labelled with this annotator's progress when a name is given."""
    out = []
    for name, folder in find_chunks().items():
        label = name
        if safe_name(annotator or ""):
            done = len(read_records(result_path(name, annotator)))
            total = len(json.loads((folder / "chunk.json").read_text(encoding="utf-8"))["samples"])
            label = f"{name} ({done}/{total} done)"
        out.append((label, name))
    return out


def refresh_chunks(annotator: str, selected: str | None = None):
    choices = chunk_choices(annotator)
    names = [v for _, v in choices]
    return gr.update(choices=choices, value=selected if selected in names else (names[0] if names else None))


def back_to_chunks(sess: dict | None, reason: str, order: str):
    """Save the sample on screen, then show the chunk list again."""
    if not sess:
        return [None, gr.update(visible=True), gr.update(visible=False), gr.update()]
    leave_current(sess, reason, order)
    return [None, gr.update(visible=True), gr.update(visible=False), refresh_chunks(sess["annotator"], sess["chunk"])]


def start(annotator: str, chunk: str):
    annotator = (annotator or "").strip()
    chunks = find_chunks()
    if not safe_name(annotator):
        raise gr.Error("Enter your name first (use the same name every time).")
    if chunk not in chunks:
        raise gr.Error("Choose a chunk. Put the chunk .zip files in the chunks/ folder and press Refresh.")
    meta = json.loads((chunks[chunk] / "chunk.json").read_text(encoding="utf-8"))
    sess = {"chunk": chunk, "dir": str(chunks[chunk]), "version": meta["version"], "annotator": annotator,
            "samples": meta["samples"], "records": read_records(result_path(chunk, annotator)), "order": []}
    first = next_unfinished(sess, -1)
    open_sample(sess, 0 if first is None else first)
    out = full_outputs(sess)
    prepare_chunk(sess)
    status = (f"Welcome back: {len(sess['records'])} samples already done." if sess["records"]
              else "Started. Your work is saved every time you move to another sample.")
    return [sess, *out[:-1], status]


def leave_current(sess: dict, reason: str, order) -> str:
    """Save the sample being left: a complete ranking or a skip. Anything else leaves it unranked."""
    s, num = current(sess), sess["idx"] + 1
    n, reason = len(s["edits"]), (reason or "").strip()
    parsed = parse_order(order, n)
    if parsed is not None:
        sess["order"] = parsed
    if reason:
        record, msg = {"ranking": None, "skipped": True, "skip_reason": reason}, f"Sample {num} skipped."
    elif len(sess["order"]) == n:
        record = {"ranking": [s["edits"][i]["key"] for i in sess["order"]], "skipped": False, "skip_reason": None}
        msg = f"Sample {num} saved."
    elif s["id"] in sess["records"]:
        write_record(sess, s["id"], None)
        return f"Sample {num} left unranked (its earlier result was removed)."
    else:
        return f"Sample {num} left unranked."
    old = sess["records"].get(s["id"])
    if not old or any(old[k] != v for k, v in record.items()):   # unchanged results keep their time stamps
        write_record(sess, s["id"], {**record, "seconds": round(time.monotonic() - sess["t0"], 1), "saved_at": now()})
    return msg


def finished_note(sess: dict) -> str:
    if len(sess["records"]) < len(sess["samples"]):
        return ""
    return (f" All samples in this chunk are done: send results/"
            f"{result_path(sess['chunk'], sess['annotator']).name} to the coordinator.")


def move(delta: int, sess: dict, reason: str, order: str):
    if sess is None:
        return [sess, *[gr.update()] * len(FULL)]
    msg = leave_current(sess, reason, order)
    target = sess["idx"] + delta
    if not 0 <= target < len(sess["samples"]):
        msg += " This is the first sample." if delta < 0 else " This is the last sample."
    open_sample(sess, target)
    return [sess, *full_outputs(sess, msg + finished_note(sess))]


def goto(sess: dict, reason: str, evt: gr.EventData):
    idx = getattr(evt, "idx", None)
    if sess is None or idx is None:
        return [sess, *[gr.update()] * len(FULL)]
    msg = leave_current(sess, reason, getattr(evt, "order", None))
    open_sample(sess, int(idx))
    return [sess, *full_outputs(sess, msg + finished_note(sess))]


# --- layout ------------------------------------------------------------------------------------
FULL: list = []


def build() -> gr.Blocks:
    with gr.Blocks(title="Edit ranking") as demo:
        sess = gr.State(None)
        order = gr.Textbox(visible=False)   # filled by NAV_JS with the ranking kept in the browser

        with gr.Column(visible=True) as setup_col:
            with gr.Row(elem_classes="inline-row"):
                gr.Markdown("Your name", elem_classes="inline-label", scale=0, min_width=80)
                name = gr.Textbox(show_label=False, container=False, placeholder="e.g. Hai", scale=2)
                gr.Markdown("Chunk", elem_classes="inline-label", scale=0, min_width=50)
                names = list(find_chunks())
                chunk = gr.Dropdown(names, value=names[0] if names else None, show_label=False,
                                    container=False, scale=2)
                refresh = gr.Button("Refresh", scale=0, min_width=90)
                start_btn = gr.Button("Start", variant="primary", scale=0, min_width=150)

        with gr.Column(visible=False) as work_col:
            progress = gr.HTML()
            with gr.Accordion("Detail", open=False, elem_classes="detail-acc"):
                grid = gr.HTML(js_on_load=GRID_JS)
            with gr.Row(elem_classes="inline-row"):
                header = gr.Markdown(elem_classes="sample-header")
                back_btn = gr.Button("Back to chunk list", size="sm", scale=0, min_width=170)
            with gr.Row(elem_classes="inline-row"):
                gr.Markdown("Edit request", elem_classes="inline-label", scale=0, min_width=100)
                request = gr.Textbox(show_label=False, container=False, interactive=False, lines=1, max_lines=8,
                                     scale=1)
            gallery = gr.HTML(elem_id="gallery")
            with gr.Row(elem_classes="inline-row"):
                gr.Markdown("Skipping reason", elem_classes="inline-label", scale=0, min_width=120)
                reason = gr.Textbox(show_label=False, container=False, elem_id="reason-box",
                                    placeholder="Only if this sample cannot be ranked", scale=4)
                prev_btn = gr.Button("< Previous", elem_id="prev-btn", scale=0, min_width=120)
                next_btn = gr.Button("Next >", variant="primary", elem_id="next-btn", scale=0, min_width=120)
            status = gr.Markdown(elem_classes="status-line")

        for title, text in GUIDE.items():
            with gr.Accordion(title, open=False, elem_classes="guide-acc"):
                gr.Markdown(text)

        FULL[:] = [setup_col, work_col, progress, grid, header, request, gallery, reason, status]
        nav = {"js": NAV_JS, "show_progress": "hidden"}

        refresh.click(refresh_chunks, [name, chunk], chunk)
        name.blur(refresh_chunks, [name, chunk], chunk)
        back_btn.click(back_to_chunks, [sess, reason, order], [sess, setup_col, work_col, chunk], js=NAV_JS)
        start_btn.click(start, [name, chunk], [sess, *FULL])
        name.submit(start, [name, chunk], [sess, *FULL])
        prev_btn.click(partial(move, -1), [sess, reason, order], [sess, *FULL], **nav)
        next_btn.click(partial(move, 1), [sess, reason, order], [sess, *FULL], **nav)
        reason.submit(partial(move, 1), [sess, reason, order], [sess, *FULL], **nav)
        grid.click(goto, [sess, reason], [sess, *FULL], show_progress="hidden")
    return demo


if __name__ == "__main__":
    theme = gr.themes.Soft(primary_hue="indigo", neutral_hue="slate", radius_size="lg",
                           font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"])
    build().launch(theme=theme, css=CSS, head=HEAD, allowed_paths=[str(CHUNKS)], inbrowser=True)
