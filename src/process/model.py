"""The common shape every raw source is converted to before filtering."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Edit:
    image: Path                 # local file (absolute)
    note: str | None = None     # text accompanying the edit: the editor's comment or a description of the change
    ref: str = ""               # source-side identifier, kept only in the provenance index


@dataclass
class Candidate:
    """One request with its original image(s) and candidate edits, before filtering."""
    source: str                 # adapter name; kept only in the provenance index
    key: str                    # stable identifier within the source (thread / post / image group)
    request: str                # the edit request text, as written
    originals: list[Path | None]  # request image(s); None = known image whose file is not available
    edits: list[Edit] = field(default_factory=list)
    trainable: bool = True      # False when the source forbids model training (VOZ: ai-train=no)
    complete: bool = True       # False while some of its images are still waiting to be downloaded

    @property
    def files(self) -> list[Path]:
        """Every local raw file of this candidate (used by prune)."""
        return [o for o in self.originals if o is not None] + [e.image for e in self.edits]
