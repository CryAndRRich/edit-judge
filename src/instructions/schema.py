"""The result of analysing one request, following RealEdit (Sushko et al., CVPR 2025) directly:

- the instruction is simplified with the rules of RealEdit's rewriting prompt (Fig. 19): URLs, typos, irrelevant
  details and thanks removed, the main task summarised; humorous or ambiguous requests become "humorous";
- the request is categorised with RealEdit's taxonomy prompt (Fig. 12, given the input image): one or more
  "operation subject" pairs, one per task in the request.

The label sets and their definitions are RealEdit's (Appendix A, Fig. 12).
"""

from dataclasses import asdict, dataclass, field

HUMOROUS = "humorous"

OPERATIONS = {
    "Add": "Inserting the subject into the image.",
    "Change-Color": "Color-correcting, silhouetting or otherwise changing the color of the subject, or colorizing "
                    "a black and white subject.",
    "Enhance": "Sharpen, enhance, blur/unblur, remove flash/glare/lens flares.",
    "Image-Formatting": "Change file type, vectorize, adjust dimensions, etc.: any change to image parameters that do "
                        "not affect the image content or aesthetics.",
    "Open-Ended": "The edit allows the editor to be creative, such as \"Edit this photo.\" or \"humorous\" or "
                  "\"do something funny with this photo\".",
    "Remove": "Erasing the subject from the image.",
    "Replace": "Substituting the subject with something specified in the instructions.",
    "Restore": "Fixing damages to the subject resulting from the preservation (e.g. stains, creases, faded color).",
    "Other": "Select this if you don't know what action is being performed.",
}

SUBJECTS = {
    "Animal": "One or multiple non-human animals.",
    "Background": "The background of the image.",
    "Body-Part": "The edit is not changing an entire person, but a body part.",
    "Clothing/Accessories": "Clothing items, accessories, leashes/collars/harnesses, anything wearable by humans or "
                            "animals.",
    "Drawing": "A hand-drawn drawing or handwritten note.",
    "Food": "Edible ingredients, prepared dishes, etc.",
    "Logo": "Logos or symbols.",
    "Manmade-Structure": "Buildings, furniture, other man-made structures or objects.",
    "Media": "Old photographs, screenshots, movie/game posters, memes, etc.: any form of print or digital media.",
    "Nature": "Plants, mountains, bodies of water, etc.: any naturally occurring items that are not people or animals.",
    "Person": "A person or group of people.",
    "Text/Patterns": "Text or patterns.",
    "Vehicle": "Cars, trucks, bikes, aircraft, trains, etc. any form of transportation vehicle.",
    "Other": "Select this if you don't know.",
}

# Short names RealEdit itself uses in its examples and tables ("Add Clothing", "Change Color", "Formatting")
ALIASES = {
    "change color": "Change-Color", "changecolor": "Change-Color", "color": "Change-Color",
    "formatting": "Image-Formatting", "image formatting": "Image-Formatting",
    "open ended": "Open-Ended", "openended": "Open-Ended",
    "clothing": "Clothing/Accessories", "accessories": "Clothing/Accessories", "apparel": "Clothing/Accessories",
    "body part": "Body-Part", "bodypart": "Body-Part",
    "text": "Text/Patterns", "patterns": "Text/Patterns",
    "manmade structure": "Manmade-Structure", "structure": "Manmade-Structure", "object": "Manmade-Structure",
}


def normalize(label: str, allowed: dict) -> str | None:
    """The canonical label for `label` (exact, case-insensitive, or a RealEdit short name), else None."""
    key = label.strip().strip(".").strip()
    for name in allowed:
        if key.lower() == name.lower():
            return name
    target = ALIASES.get(key.lower().replace("_", " ").replace("-", " ")) or ALIASES.get(key.lower())
    return target if target in allowed else None


@dataclass
class RequestAnalysis:
    sample_id: str
    model: str                                   # client name
    prompt: str                                  # prompt name and version, e.g. "analyze_v1"
    instruction: str                             # the simplified instruction, or "humorous"
    categories: list[dict] = field(default_factory=list)   # [{"operation", "subject"}], one per task
    raw: str = ""                                # the model's full answer, kept for checking
    problems: list[str] = field(default_factory=list)      # parsing problems; empty when valid

    @property
    def is_humorous(self) -> bool:
        return self.instruction.strip().strip(".").lower() == HUMOROUS

    def to_dict(self) -> dict:
        return asdict(self)
