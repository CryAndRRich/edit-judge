"""The structured result of analysing one request, and the fixed label sets (PROJECT.md Step 4)."""

from dataclasses import asdict, dataclass, field

# PSR's 15 action verbs, plus Restore from RealEdit
OPERATIONS = ("Adjust", "Delete", "Add", "Recoloring", "Replace", "Apply", "Transform", "Superresolution", "Merge",
              "Relighting", "Move", "Crop", "Specialized", "Zoom", "Clone", "Restore")
# RealEdit's subjects
SUBJECTS = ("Animal", "Background", "Body-Part", "Clothing/Accessories", "Drawing", "Food", "Logo",
            "Manmade-Structure", "Media", "Nature", "Person", "Text/Patterns", "Vehicle", "Other")


@dataclass
class RequestAnalysis:
    sample_id: str
    model: str                                   # client name and version
    prompt: str                                  # prompt name and version, e.g. "analyze_v1"
    is_edit_request: bool                        # False for "is this real?", meta posts, ...
    clean_instruction: str                       # short, actionable English; nothing the user did not ask
    sub_instructions: list[str] = field(default_factory=list)   # one per separate change, in order
    operations: list[str] = field(default_factory=list)         # from OPERATIONS
    subjects: list[str] = field(default_factory=list)           # from SUBJECTS
    ambiguity: int | None = None                 # 1 (crystal clear) to 5 (completely vague)
    complexity: int | None = None                # 1 to 5
    is_humorous: bool = False                    # jokes, "do something funny"
    refers_external: bool = False                # needs content not in the image ("use my other photo")
    language: str | None = None                  # ISO 639-1 code of the original request
    notes: str | None = None

    def problems(self) -> list[str]:
        """Values outside the label sets or ranges (empty when valid)."""
        out = [f"operation {o!r}" for o in self.operations if o not in OPERATIONS]
        out += [f"subject {s!r}" for s in self.subjects if s not in SUBJECTS]
        out += [f"{k} {v}" for k, v in (("ambiguity", self.ambiguity), ("complexity", self.complexity))
                if v is not None and not 1 <= v <= 5]
        return out

    def to_dict(self) -> dict:
        return asdict(self)
