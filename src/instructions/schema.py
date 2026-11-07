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
    # plain words small models use instead of the taxonomy names
    "people": "Person", "persons": "Person", "human": "Person", "humans": "Person", "face": "Body-Part",
    "hair": "Body-Part", "image": "Media", "photo": "Media", "picture": "Media",
}


# Loose words the 7B model wrote instead of taxonomy names (full run, 2026-11-07), mapped by hand to the closest
# RealEdit name following the definitions above. Used only for labels that are not valid taxonomy names
# (parse.loose_category); a word not listed here becomes "Other".
OPERATION_WORDS = {
    "Change-Color": ["colorize", "colorise", "colorization", "colour", "color correct", "color correcting",
                     "recolor", "silhouette", "darken", "lighten", "change hair", "add color"],
    "Enhance": ["fix", "clean", "clean up", "cleanup", "touch up", "retouch", "sharpen", "blur", "unblur", "brighten",
                "adjust", "balance", "match", "reduce noise", "denoise", "smooth", "focus", "define", "un distort",
                "highlight", "reduce glow", "add brightness"],
    "Image-Formatting": ["vectorize", "digitize", "trace", "resize", "crop", "scale", "save", "center", "frame",
                         "rotate", "flip", "flip image", "convert", "separate", "make"],
    "Add": ["insert", "extend"],
    "Remove": ["clear"],
    "Replace": ["change", "swap", "face swap", "alter", "translate", "change haircut", "change expression",
                "change style"],
    "Open-Ended": ["create", "draw", "drawn", "render", "cartoonify", "animate", "stylize", "mashup", "fill",
                   "modernize"],
    "Other": ["move", "combine", "merge", "blend", "duplicate", "mirror", "change size", "reduce size", "reduce",
              "change position", "change length", "lengthen", "censor", "modify", "keep"],
}
SUBJECT_WORDS = {
    "Clothing/Accessories": ["helmet", "glasses", "tie", "mask", "glove", "shoe", "shoes", "costume", "jewelry", "ring",
                             "bag", "leash", "hat", "laces", "wristband", "charm"],
    "Body-Part": ["eyes", "eye", "ears", "ear", "tongue", "body", "skin", "lips", "lipstick", "makeup", "tattoo",
                  "red eye"],
    "Nature": ["flower", "flowers", "water", "sky", "sun", "moon", "plant", "weather", "clouds", "ocean", "beach"],
    "Manmade-Structure": ["cabinets", "chair", "ceiling", "stand", "basket", "broom", "umbrella", "guitar",
                          "pickguard", "phone", "camera", "bottle", "battery", "pillow", "trophy", "stick", "ball",
                          "weapon", "gun", "lightsaber", "light saber", "cables", "rope", "ropes", "sign", "trash",
                          "lights", "lens", "money", "sand castle"],
    "Logo": ["flag", "symbol", "emoji"],
    "Animal": ["fish", "feather"],
    "Food": ["beer"],
    "Media": ["magazine", "screen", "images", "poster", "banner", "art", "animation", "border", "waveform", "date"],
    "Text/Patterns": ["pattern", "texture", "watermark", "tag", "design"],
}
# "Remove <one of these>" is RealEdit's Enhance ("remove flash/glare/lens flares"); "isolate X" and "extract X"
# mean removing the background around X
ENHANCE_REMOVALS = {"flash", "glare", "lens flare", "reflection", "reflections", "shadow", "shadows", "shadowing",
                    "red eye", "noise", "filter", "water spots"}
BACKGROUND_REMOVALS = {"isolate", "extract"}


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
