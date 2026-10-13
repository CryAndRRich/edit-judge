"""Adapter for GIER (raw/gier): records sharing one input image form one request with several edits.

The request is the original post text (`request`). GIER has no editor comments; the note is the
first expert-written description of the change (`expert_summary`), which says what the edit did.
"""

import json
from collections import defaultdict
from collections.abc import Iterator

from ...paths import RAW_DIR

from ..model import Candidate, Edit

GIER_DIR = RAW_DIR / "gier"


def gier() -> Iterator[Candidate]:
    path = GIER_DIR / "GIER.json"
    if not path.exists():
        return
    groups: dict[str, list] = defaultdict(list)
    for rec in json.loads(path.read_text(encoding="utf-8")):
        groups[rec["input"]].append(rec)
    for original, recs in groups.items():
        request = next((r["request"] for r in recs if r.get("request", "").strip()), "")
        cand = Candidate(source="gier", key=original, request=request, originals=[GIER_DIR / "images" / original])
        for r in recs:
            summary = (r.get("expert_summary") or [None])[0]
            cand.edits.append(Edit(image=GIER_DIR / "images" / r["output"], note=summary, ref=r["output"]))
        yield cand
