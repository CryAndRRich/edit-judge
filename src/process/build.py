"""Filter candidates from every adapter and add them to the unified dataset (data/samples, data/samples.jsonl).

Rules (decided 2026-10-02):
- the request text is not empty;
- exactly one original image, present locally and readable;
- every kept image (original and edits) has a shorter side of at least MIN_SIDE pixels;
- edits identical to the original, and duplicate edits within a sample, are dropped (SHA-256);
- at least MIN_EDITS edits remain;
- the same original appearing in several sources is kept once.

Incremental: every decision (kept / rejected, with the reason) is appended to the ledger
(data/index/ledger.jsonl). Later builds skip decided candidates, so raw files of decided
candidates can be deleted (`prune`) without losing anything: data/samples holds hard links, which
keep the kept images alive. Candidates still waiting for downloads are left undecided.

The samples hold no source information. The mapping from sample IDs back to sources (licences,
takedown requests, de-duplication, the VOZ no-training rule) is data/index/provenance.jsonl.
"""

import hashlib
import json
import logging
import os
import shutil
import sqlite3
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from ..paths import DATA_DIR, INDEX_DIR, LEDGER, PROVENANCE, RAW_DIR, SAMPLES_DIR, SAMPLES_JSONL

from .adapters import ADAPTERS
from .imagecheck import ImageChecker
from .model import Candidate

log = logging.getLogger(__name__)

MIN_EDITS = 2
MIN_SIDE = 256


def sample_id(c: Candidate) -> str:
    return hashlib.sha1(f"{c.source}:{c.key}".encode("utf-8")).hexdigest()[:12]


def _ext(path: Path, fmt: str | None) -> str:
    """File extension from the detected format, falling back to the raw file's own suffix."""
    if fmt:
        return {"JPEG": ".jpg", "MPO": ".jpg", "TIFF": ".tif"}.get(fmt, f".{fmt.lower()}")
    return path.suffix.lower() or ".bin"


def _link(src: Path, dst: Path) -> None:
    """Hard link (no extra disk space); copy only if linking is impossible (different file system)."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _rel(p: Path) -> str:
    return str(p.relative_to(DATA_DIR))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


class Builder:
    def __init__(self, min_edits: int = MIN_EDITS, min_side: int = MIN_SIDE, regenerate_mineo_pairing: bool = False):
        self.min_edits, self.min_side = min_edits, min_side
        self.regenerate_mineo_pairing = regenerate_mineo_pairing
        self.images = ImageChecker()
        self.rejected: dict[str, Counter] = defaultdict(Counter)      # source -> reason -> candidates (this run)
        self.edit_drops: dict[str, Counter] = defaultdict(Counter)    # source -> reason -> edits (this run)
        self.seen: Counter = Counter()                                # candidates seen this run
        self.pending: Counter = Counter()                             # incomplete candidates this run
        self.decisions: list[dict] = []

    # --- filtering -------------------------------------------------------------------------
    def _usable(self, path: Path | None) -> tuple[dict | None, str | None]:
        if path is None:
            return None, "not_downloaded"
        info = self.images.info(path)
        if not info.get("readable"):
            return None, info.get("reason", "unreadable")
        if min(info["width"], info["height"]) < self.min_side:
            return None, "too_small"
        return info, None

    def check(self, c: Candidate) -> tuple[dict | None, str | None]:
        """Return (cleaned sample, None) or (None, rejection reason)."""
        if not c.request.strip():
            return None, "empty_request"
        if len(c.originals) == 0:
            return None, "no_original"
        if len(c.originals) > 1:
            return None, "multiple_originals"
        orig_info, why = self._usable(c.originals[0])
        if why:
            return None, f"original_{why}"
        kept, seen = [], {orig_info["sha256"]}
        for e in c.edits:
            info, why = self._usable(e.image)
            if why:
                self.edit_drops[c.source].update([f"edit_{why}"])
                continue
            if info["sha256"] in seen:
                self.edit_drops[c.source].update(
                    ["edit_identical_to_original" if info["sha256"] == orig_info["sha256"] else "edit_duplicate"])
                continue
            seen.add(info["sha256"])
            kept.append((e, info))
        if len(kept) < self.min_edits:
            return None, f"fewer_than_{self.min_edits}_edits"
        return {"candidate": c, "original": (c.originals[0], orig_info), "edits": kept}, None

    def _decide(self, c: Candidate, decision: str, reason: str | None, sid: str | None = None,
                original_sha: str | None = None) -> None:
        if reason:
            self.rejected[c.source].update([reason])
        self.decisions.append({"source": c.source, "key": c.key, "decision": decision, "reason": reason,
                               "sample_id": sid, "original_sha256": original_sha,
                               "files": [_rel(f) for f in c.files], "decided_at": _now(), "pruned": False})

    # --- build -----------------------------------------------------------------------------
    def run(self, write: bool = True, rebuild: bool = False) -> dict:
        """Incremental build. With rebuild=True (or no ledger yet), data/ is recreated from all of raw/."""
        ledger = [] if rebuild else read_jsonl(LEDGER)
        full = rebuild or not LEDGER.exists()
        decided = {(d["source"], d["key"]) for d in ledger}
        kept_sha = {d["original_sha256"] for d in ledger if d["decision"] == "kept"}
        new_samples: list[dict] = []
        by_sha: dict[str, dict] = {}
        for name, adapter in ADAPTERS.items():
            gen = adapter(regenerate_pairing=self.regenerate_mineo_pairing) if name == "mineo" else adapter()
            for c in gen:
                if (c.source, c.key) in decided:
                    continue
                self.seen[c.source] += 1
                if not c.complete:
                    self.pending[c.source] += 1
                    continue
                s, reason = self.check(c)
                if not s:
                    self._decide(c, "rejected", reason)
                    continue
                sha = s["original"][1]["sha256"]
                if sha in kept_sha:
                    self._decide(c, "rejected", "duplicate_original", original_sha=sha)
                elif sha in by_sha:  # same original twice in this run: keep the copy with more edits
                    loser = s if len(s["edits"]) <= len(by_sha[sha]["edits"]) else by_sha[sha]
                    by_sha[sha] = s if loser is by_sha[sha] else by_sha[sha]
                    self._decide(loser["candidate"], "rejected", "duplicate_original", original_sha=sha)
                else:
                    by_sha[sha] = s
        for sha, s in by_sha.items():
            new_samples.append(s)
            self._decide(s["candidate"], "kept", None, sample_id(s["candidate"]), sha)
        self.images.save()
        if write:
            self.write(sorted(new_samples, key=lambda s: sample_id(s["candidate"])), full)
            mode = "w" if full else "a"
            with open(LEDGER, mode, encoding="utf-8") as fh:
                for d in self.decisions:
                    fh.write(json.dumps(d, ensure_ascii=False) + "\n")
        return self.run_stats(len(new_samples), full)

    def write(self, samples: list[dict], full: bool) -> None:
        out = SAMPLES_DIR
        out.mkdir(parents=True, exist_ok=True)
        INDEX_DIR.mkdir(parents=True, exist_ok=True)
        if full and out.exists():
            shutil.rmtree(out)  # full rebuild only; incremental builds never delete samples
        mode = "w" if full else "a"
        with open(SAMPLES_JSONL, mode, encoding="utf-8") as fj, \
             open(PROVENANCE, mode, encoding="utf-8") as fp:
            for s in samples:
                c, sid = s["candidate"], sample_id(s["candidate"])
                d = out / sid
                opath, oinfo = s["original"]
                orig_rel = f"original{_ext(opath, oinfo['format'])}"
                _link(opath, d / orig_rel)
                edits, prov_edits = [], []
                for n, (e, info) in enumerate(s["edits"], 1):
                    rel = f"edits/{n:02d}{_ext(e.image, info['format'])}"
                    _link(e.image, d / rel)
                    edits.append({"image": rel, "note": e.note})
                    prov_edits.append({"image": rel, "raw": _rel(e.image), "ref": e.ref})
                record = {"id": sid, "request": c.request, "original": orig_rel, "edits": edits}
                (d / "sample.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
                fj.write(json.dumps({**record, "dir": f"samples/{sid}"}, ensure_ascii=False) + "\n")
                fp.write(json.dumps({"id": sid, "source": c.source, "key": c.key, "trainable": c.trainable,
                                     "original_raw": _rel(opath), "edits": prov_edits}, ensure_ascii=False) + "\n")
        log.info("added %s samples to %s", len(samples), SAMPLES_DIR)

    def run_stats(self, added: int, full: bool) -> dict:
        return {
            "mode": "full rebuild" if full else "incremental",
            "added_samples": added,
            "this_run": {src: {"new_candidates": self.seen[src], "still_downloading": self.pending[src],
                               "rejected": dict(self.rejected[src].most_common()),
                               "edits_dropped": dict(self.edit_drops[src].most_common())}
                         for src in sorted(self.seen)},
            "dataset": dataset_stats(),
            "rules": {"min_edits": self.min_edits, "min_side_px": self.min_side},
        }


# --- dataset-wide statistics -----------------------------------------------------------------

def dataset_stats() -> dict:
    samples = read_jsonl(SAMPLES_JSONL)
    prov = read_jsonl(PROVENANCE)
    n_edits = [len(s["edits"]) for s in samples]
    exts = Counter(Path(p).suffix.lstrip(".").upper() for s in samples
                   for p in [s["original"], *(e["image"] for e in s["edits"])])
    return {"samples": len(samples), "edits": sum(n_edits),
            "edits_per_sample": dict(sorted(Counter(min(n, 6) for n in n_edits).items())),
            "image_formats": dict(exts.most_common()),
            "not_trainable_samples": sum(1 for p in prov if not p.get("trainable", True))}


# --- prune -----------------------------------------------------------------------------------

def prune(sources: list[str] | None = None) -> dict:
    """Delete raw image files of decided candidates (optionally only some sources).

    Files hard-linked into data/ stay available there; only files no longer linked anywhere
    free disk space. Crawler databases, metadata and page caches are kept, and crawler image
    rows are marked 'pruned' so the crawler does not download them again."""
    ledger = read_jsonl(LEDGER)
    freed = deleted = 0
    by_source: dict[str, list[str]] = defaultdict(list)
    for d in ledger:
        if d.get("pruned") or (sources and d["source"] not in sources):
            continue
        for rel in d["files"]:
            p = DATA_DIR / rel
            if p.exists():
                st = p.stat()
                if st.st_nlink == 1:
                    freed += st.st_size  # not linked into data/: deleting really frees space
                p.unlink()
                deleted += 1
                by_source[d["source"]].append(rel)
        d["pruned"] = True
    with open(LEDGER, "w", encoding="utf-8") as fh:
        for d in ledger:
            fh.write(json.dumps(d, ensure_ascii=False) + "\n")
    for src, rels in by_source.items():  # keep crawler state consistent
        db_path = RAW_DIR / src / "crawl.sqlite"
        if db_path.exists():
            db = sqlite3.connect(db_path)
            db.executemany("UPDATE images SET status='pruned' WHERE local_path=?", [(r,) for r in rels])
            db.commit()
            db.close()
    log.info("pruned %s raw files, freed %.2f GB", deleted, freed / 1e9)
    return {"deleted_files": deleted, "freed_gb": round(freed / 1e9, 3),
            "by_source": {k: len(v) for k, v in by_source.items()}}
