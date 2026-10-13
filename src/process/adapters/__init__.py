"""Adapters turn one kind of raw data into Candidates. To add a source, write an adapter
(a callable returning an iterable of Candidate) and add it to ADAPTERS.

Not adapted, by decision (2026-10-02):
- Tan et al. 2019 (raw/tan2019_imgedit): no original request text, only crowd-written edit descriptions.
- RealEdit (raw/realedit): image files are not downloaded yet (URLs only).
"""

from .crawled import crawled_sources
from .gier import gier
from .mineo import mineo

ADAPTERS = {"crawled": crawled_sources, "gier": gier, "mineo": mineo}
