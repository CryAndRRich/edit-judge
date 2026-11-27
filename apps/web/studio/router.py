"""Maps an editing instruction (English or Vietnamese, with or without diacritics) to fine-grained intents and
parameters (assistant.py turns them into a task and the models to run).

How it works:
1. The instruction is normalised twice: lower case (with diacritics) and ASCII (diacritics stripped, "đ" -> "d").
   A keyword written with Vietnamese diacritics is matched against the first form, a keyword without them against
   the second, so "xóa phông nền" and "xoa phong nen" both match, while short unaccented words do not collide
   with unrelated Vietnamese ones.
2. Intent rules (regular expressions on word boundaries) find intents such as brighten, sharpen, remove_bg, bw,
   rotate.
3. Disambiguation: "remove the noise" is denoising, not object removal; "remove the background" is a cut-out, not
   the removal of an object; "change the colour" is a recolour, not a replacement.
4. Parameters are parsed from the instruction: target colour, rotation angle and direction, size, scale, aspect.
"""

import re
import unicodedata
from dataclasses import dataclass, field


# intent -> keyword patterns: regexes; word boundaries are added around them.
INTENTS: dict[str, tuple[str, ...]] = {
    # Enhance
    "brighten": (
        r"bright(en|er|ness)?", r"lighten", r"lighter", r"more light", r"low[- ]light", r"too dark", r"underexposed",
        r"illuminat\w*", r"exposure",
        r"sáng( hơn| lên)?", r"làm sáng", r"tăng sáng", r"độ sáng", r"thiếu sáng", r"tối quá", r"quá tối", r"bị tối",
        r"lam sang", r"tang sang", r"sang hon", r"sang len"),
    "darken": (
        r"dark(en|er)", r"dimmer", r"overexposed", r"too bright", r"tối hơn", r"làm tối", r"giảm sáng",
        r"lam toi", r"toi hon", r"giam sang"),
    "sharpen": (
        r"sharp(en|er|ness)?", r"crisp(er)?", r"clear(er)?", r"more detail(s|ed)?", r"detail(s)?",
        r"nét( hơn)?", r"làm nét", r"rõ( hơn| nét)?", r"sắc nét", r"chi tiết",
        r"lam net", r"net hon", r"sac net", r"ro net", r"ro hon"),
    "denoise": (
        r"de-?nois\w*", r"noise", r"noisy", r"grain(y)?", r"nhiễu", r"khử nhiễu", r"hạt", r"nhieu", r"khu nhieu"),
    "deblur": (
        r"de-?blur\w*", r"blur(ry|red)?", r"out of focus", r"unblur", r"motion blur",
        r"mờ", r"bị mờ", r"nhòe", r"bi mo", r"khu mo", r"het mo"),
    "upscale": (
        r"upscal\w*", r"up-?sampl\w*", r"super[- ]?resolution", r"higher resolution", r"high[- ]res(olution)?",
        r"increase (the )?resolution", r"enlarge", r"\d?x ?(upscale|zoom)", r"4k", r"8k", r"full ?hd", r"hd",
        r"độ phân giải", r"phân giải", r"phóng to", r"tăng kích thước", r"to hơn", r"siêu phân giải",
        r"do phan giai", r"phan giai", r"phong to"),
    "contrast": (
        r"contrast\w*", r"vivid", r"vibran\w*", r"pop", r"punch(y|ier)?", r"saturat\w*", r"dynamic",
        r"tương phản", r"rực rỡ", r"sống động", r"đậm( hơn)?", r"bão hòa", r"bão hoà",
        r"tuong phan", r"ruc ro", r"song dong"),
    "enhance": (
        r"enhanc\w*", r"improv\w*", r"better", r"quality", r"beautif\w*", r"retouch\w*", r"polish\w*",
        r"professional", r"fix (the )?(photo|image|picture)", r"look good",
        r"cải thiện", r"nâng cấp", r"đẹp hơn", r"làm đẹp", r"chất lượng", r"chỉnh sửa ảnh",
        r"cai thien", r"nang cap", r"dep hon", r"lam dep", r"chat luong"),
    "warm": (r"warm(er|th)?", r"ấm( hơn| áp)?", r"am ap", r"am hon"),
    "cool": (r"cool(er)?", r"colder", r"mát( hơn)?", r"lạnh hơn", r"mat hon", r"lanh hon"),
    # Restore
    "restore": (
        r"restor\w*", r"repair\w*", r"old (photo|picture|image)\w*", r"vintage photo", r"damaged", r"scratch\w*",
        r"torn", r"tear(s)?", r"faded", r"crease\w*", r"jpeg artifact\w*", r"compression artifact\w*", r"artifact\w*",
        r"phục hồi", r"khôi phục", r"ảnh cũ", r"hư hỏng", r"bị hỏng", r"vết xước", r"trầy", r"rách", r"phai màu",
        r"bạc màu", r"phuc hoi", r"khoi phuc", r"anh cu", r"vet xuoc", r"phai mau"),
    "face": (
        r"face(s)?", r"facial", r"portrait", r"skin", r"khuôn mặt", r"gương mặt", r"mặt", r"da mặt", r"chân dung",
        r"khuon mat", r"guong mat", r"chan dung"),
    # Remove
    "remove_object": (
        r"remov\w*", r"eras\w*", r"delet\w*", r"get rid of", r"take out", r"take away", r"clean up", r"erase",
        r"without (the|a|any)", r"no more", r"eliminat\w*", r"watermark\w*", r"logo(s)?",
        r"xóa", r"xoá", r"loại bỏ", r"bỏ đi", r"tẩy", r"gỡ bỏ", r"xóa bỏ", r"xoá bỏ",
        r"xoa", r"loai bo", r"bo di", r"tay xoa", r"go bo"),
    # Add / Replace
    "add": (
        r"add(ing)?", r"put", r"insert\w*", r"place", r"include", r"wear(ing)?", r"give (him|her|them|it)",
        r"thêm", r"chèn", r"đặt (vào|thêm|lên)", r"cho thêm", r"đội", r"đeo", r"them vao", r"chen vao", r"them mot"),
    "replace": (
        r"replac\w*", r"swap\w*", r"substitut\w*", r"turn (.{1,30}) into", r"transform\w*", r"change (.{1,30}) (to|into)",
        r"make (it|him|her|them) (look )?like", r"thay( thế| bằng| đổi)?", r"đổi( thành| sang)?", r"biến (.{1,20}) thành",
        r"thay the", r"thay bang", r"doi thanh", r"bien thanh"),
    "replace_bg": (
        r"(change|replace|swap|new) (the )?background", r"background (to|with|of)", r"(on|at|in) (a|the) (beach|forest|city|mountain|space|studio)",
        r"đổi (phông|nền)", r"thay (phông|nền)", r"phông nền (thành|là)", r"doi nen", r"thay nen", r"doi phong"),
    # Change color
    "bw": (
        r"black (and|&) white", r"b ?& ?w", r"b/w", r"gr[ae]y ?scale", r"monochrom\w*", r"desaturat\w*", r"sepia",
        r"đen trắng", r"trắng đen", r"den trang", r"trang den", r"đơn sắc"),
    "colorize": (
        r"colou?riz\w*", r"add colou?r(s)?", r"in colou?r", r"tô màu", r"thêm màu", r"tô màu cho", r"lên màu",
        r"to mau", r"them mau", r"len mau"),
    "recolor": (
        r"colou?r(ed|s)?", r"hue", r"paint(ed)?", r"dye\w*", r"tint\w*",
        r"màu", r"sơn", r"nhuộm", r"mau sac", r"doi mau", r"son mau", r"nhuom"),
    # Image formatting
    "remove_bg": (
        r"(remove|erase|delete|cut out|get rid of|drop|clear) (the |its )?(background|backdrop|bg)",
        r"background remov\w*", r"no background", r"transparent", r"cut-?out", r"isolat\w*", r"white background",
        r"plain background", r"(remove|erase).{0,12}background",
        r"(remove|erase|delete|xoa|bo)\b.{0,40}\b(and|va) (the |cai )?(background|backdrop|phong nen|nen|bg)",
        r"tách nền", r"xóa (phông|nền|phông nền)", r"xoá (phông|nền|phông nền)", r"bỏ (phông|nền)", r"nền trắng",
        r"trong suốt", r"tach nen", r"xoa (phong|nen|phong nen)", r"nen trang", r"trong suot"),
    "crop": (
        r"crop\w*", r"trim", r"cắt (ảnh|bớt|xén|cúp)", r"cắt xén", r"cúp ảnh", r"cat anh", r"cat xen", r"crop anh"),
    "rotate": (
        r"rotat\w*", r"turn (it |the (photo|image|picture) )?(left|right|upside down|clockwise|counter-?clockwise)",
        r"upside down", r"(\d{2,3}) ?(°|deg\w*)", r"xoay", r"quay (ảnh|trái|phải)", r"quay anh"),
    "flip": (r"flip\w*", r"mirror\w*", r"lật", r"lat anh", r"đối xứng", r"phản chiếu"),
    "resize": (
        r"resiz\w*", r"scale (down|up|to)", r"smaller", r"shrink", r"downscal\w*", r"\d{2,5} ?[x×] ?\d{2,5}",
        r"thu nhỏ", r"đổi kích thước", r"thay đổi kích thước", r"kích thước", r"nhỏ hơn", r"thu nho", r"kich thuoc"),
    "extend": (
        r"extend\w*", r"outpaint\w*", r"expand\w*", r"widen", r"zoom out", r"wider", r"uncrop",
        r"mở rộng", r"nới rộng", r"rộng hơn", r"mo rong", r"rong hon"),
    "square": (r"square", r"1:1", r"vuông", r"vuong"),
    # set by the disambiguation only ("blur the background"): the generalists decide
    "blur_bg": (),
}

# Words that, right after a removal verb, mean a degradation (Enhance/Restore) rather than an object (Remove).
DEGRADATIONS = (r"noise", r"noisy", r"grain", r"blur\w*", r"haze", r"fog", r"rain", r"scratch\w*", r"artifact\w*",
                r"compression", r"dust", r"stain\w*", r"tear\w*", r"crease\w*", r"red[- ]eye", r"glare",
                r"wrinkle\w*", r"blemish\w*", r"acne", r"pimple\w*", r"shadow\w*",
                r"nhiễu", r"mờ", r"xước", r"vết xước", r"vết bẩn", r"bụi", r"sương", r"mụn", r"nếp nhăn", r"hạt",
                r"nhieu", r"bi mo", r"vet xuoc", r"vet ban", r"bui", r"mun", r"nep nhan")
DEGRADATION_INTENT = {"noise": "denoise", "noisy": "denoise", "grain": "denoise", "nhiễu": "denoise", "nhieu": "denoise",
                      "hạt": "denoise", "blur": "deblur", "mờ": "deblur", "bi mo": "deblur", "haze": "enhance",
                      "fog": "enhance", "rain": "enhance", "sương": "enhance"}
REMOVE_VERBS = (r"remov\w*", r"eras\w*", r"delet\w*", r"get rid of", r"take out", r"eliminat\w*", r"clean up",
                r"reduce", r"fix", r"xóa", r"xoá", r"loại bỏ", r"bỏ", r"tẩy", r"khử", r"giảm", r"xoa", r"loai bo",
                r"khu", r"giam", r"sửa")

COLOR_NAMES = {
    "red": "red", "orange": "orange", "yellow": "yellow", "gold": "yellow", "golden": "yellow", "green": "green",
    "blue": "blue", "navy": "blue", "cyan": "cyan", "teal": "cyan", "turquoise": "cyan", "purple": "purple",
    "violet": "purple", "magenta": "magenta", "pink": "pink", "brown": "brown", "white": "white", "black": "black",
    "redder": "red", "bluer": "blue", "greener": "green", "yellower": "yellow", "pinker": "pink",
    "đỏ": "red", "vàng": "yellow", "xanh lá": "green", "xanh lục": "green", "xanh dương": "blue",
    "xanh da trời": "blue", "xanh biển": "blue", "xanh": "blue", "tím": "purple", "hồng": "pink", "nâu": "brown",
    "màu cam": "orange", "vang": "yellow", "xanh la": "green", "xanh duong": "blue", "mau cam": "orange",
    "mau do": "red", "mau tim": "purple", "mau hong": "pink", "nau": "brown",
}


def strip_diacritics(text: str) -> str:
    text = text.replace("đ", "d").replace("Đ", "D")
    return "".join(ch for ch in unicodedata.normalize("NFD", text) if unicodedata.category(ch) != "Mn")


def has_diacritics(pattern: str) -> bool:
    return strip_diacritics(pattern) != pattern


def compile_pattern(pattern: str) -> re.Pattern:
    return re.compile(rf"(?<![\w]){pattern}(?![\w])", re.IGNORECASE | re.UNICODE)


COMPILED = {intent: [(compile_pattern(p), has_diacritics(p)) for p in patterns] for intent, patterns in INTENTS.items()}
DEGRADATION_RE = compile_pattern("(" + "|".join(DEGRADATIONS) + ")")
REMOVE_VERB_RE = compile_pattern("(" + "|".join(REMOVE_VERBS) + ")")
BACKGROUND_RE = compile_pattern(r"(background|backdrop|bg|phông|nền|phong nen|nen)")
COLOR_RES = [(compile_pattern(re.escape(word)), color, has_diacritics(word))
             for word, color in sorted(COLOR_NAMES.items(), key=lambda item: -len(item[0]))]


@dataclass
class Route:
    instruction: str
    intents: dict[str, list[str]] = field(default_factory=dict)     # intent -> matched words
    params: dict = field(default_factory=dict)                      # target colour, rotation angle, size...


class TaskRouter:
    """route(instruction) -> intents and parameters."""

    def route(self, instruction: str) -> Route:
        text = unicodedata.normalize("NFC", (instruction or "").strip()).lower()
        text = re.sub(r"\s+", " ", text)
        ascii_text = strip_diacritics(text)
        route = Route(instruction=instruction or "")

        for intent, patterns in COMPILED.items():
            hits = []
            for regex, accented in patterns:
                for match in regex.finditer(text if accented else ascii_text):
                    hits.append(match.group(0))
            if hits:
                route.intents[intent] = sorted(set(hits))

        self._disambiguate(route, text, ascii_text)
        self._params(route, text, ascii_text)
        return route

    @staticmethod
    def _disambiguate(route: Route, text: str, ascii_text: str) -> None:
        intents = route.intents
        # "blur the background" asks for a blur (bokeh), it is not deblurring
        if re.search(r"(blur|lam mo|mo)\s+(the\s+|phan\s+)?(background|backdrop|bg|nen|phong nen|phong|hau canh)", ascii_text):
            intents.pop("deblur", None)
            intents["blur_bg"] = ["blur background"]
        # "remove the noise / scratches / blur": a degradation, not an object
        if "remove_object" in intents:
            objects_left = False
            for source in (text, ascii_text):
                for verb in REMOVE_VERB_RE.finditer(source):
                    tail = source[verb.end():verb.end() + 40]
                    tail_words = re.sub(r"^\s*(the|all|any|those|these|of|cái|các|những|mọi|nhung|cac|cai)\s+", "",
                                        tail)
                    degradation = DEGRADATION_RE.match(tail_words.strip())
                    if degradation:
                        word = strip_diacritics(degradation.group(0)) if source is ascii_text else degradation.group(0)
                        follow = DEGRADATION_INTENT.get(word) or DEGRADATION_INTENT.get(strip_diacritics(word))
                        if follow:
                            intents.setdefault(follow, []).append(degradation.group(0))
                        elif re.match(r"(scratch|xước|xuoc|tear|stain|dust|crease|bụi|bui|vết)", word):
                            intents.setdefault("restore", []).append(degradation.group(0))
                        else:
                            intents.setdefault("enhance", []).append(degradation.group(0))
                    elif not BACKGROUND_RE.match(tail_words.strip()) and not re.match(r"\s*(colou?r|màu|mau)", tail_words):
                        objects_left = True
            if not objects_left:
                intents.pop("remove_object")
        # "remove the background" is a cut-out (remove_bg), not object removal of "background"
        if "remove_bg" in intents and "remove_object" in intents:
            other = [w for w in intents["remove_object"] if not BACKGROUND_RE.search(w)]
            if not other or all(REMOVE_VERB_RE.fullmatch(w) for w in other):
                # the only removal is the background, unless another object is named ("remove the man and the bg")
                if not re.search(r"\b(and|và|va)\b", ascii_text):
                    intents.pop("remove_object")
        # "change the colour": colour intents win over the generic replace
        if "replace" in intents and ({"recolor", "bw", "colorize"} & set(intents)):
            if not re.search(r"(replace|swap|thay the|bien .* thanh)", ascii_text):
                intents.pop("replace")
        if "replace_bg" in intents:
            intents.pop("replace", None)
            if not re.search(r"(remove|transparent|tach|trong suot)", ascii_text):
                intents.pop("remove_bg", None)
        # "colourise this black-and-white photo": black and white describes the input, not the request
        if "colorize" in intents and "bw" in intents:
            intents.pop("bw")
        # "faded colours" (phai màu, bạc màu) is a restoration, not a recolour
        if "recolor" in intents and re.search(r"(phai|bac|faded|washed[- ]out) (mau|colou?r)", ascii_text):
            if not re.search(r"(doi|thay|chuyen|son|change|make|turn|paint) .{0,20}(mau|colou?r)", ascii_text):
                intents.pop("recolor")
        # "colour" in "black and white" / "colorize" is not a recolour; "remove colour" is black and white
        if "recolor" in intents and ({"bw", "colorize"} & set(intents)):
            intents.pop("recolor")
        if "recolor" in intents and re.search(r"(remove|drop|bỏ|xóa|xoá|xoa) (the |all )?(colou?r|màu|mau)", text + " " + ascii_text):
            intents.pop("recolor")
            intents.setdefault("bw", []).append("remove colour")
        # "make it darker" is not brighten; "brighten the colours" is not recolour
        if "darken" in intents and "brighten" in intents:
            darker = re.search(r"dark(en|er)|toi hon|lam toi", ascii_text)
            if darker and not re.search(r"bright(en|er)|lighten|sang hon|lam sang|tang sang", ascii_text):
                intents.pop("brighten")
        if "recolor" in intents and set(intents) & {"brighten", "contrast", "warm", "cool", "enhance"}:
            if not any(regex.search(t) for regex, _, accented in COLOR_RES for t in [text if accented else ascii_text]):
                intents.pop("recolor")
                intents.setdefault("contrast", []).append("colour")
        # adding a colour word to "make the car red": recolour
        if "recolor" not in intents and not ({"bw", "colorize"} & set(intents)):
            if re.search(r"\b(make|turn|paint|change|sơn|đổi|chuyển|lam|doi|son)\b", ascii_text):
                for regex, color, accented in COLOR_RES:
                    if regex.search(text if accented else ascii_text):
                        intents["recolor"] = [color]
                        intents.pop("replace", None)
                        break
        # "sharp" inside "sharpen the colours" etc. is fine; "clear" in "clear background" is a cut-out
        if "sharpen" in intents and "remove_bg" in intents and intents["sharpen"] == ["clear"]:
            intents.pop("sharpen")
        # face alone (no restoration verb) is an Enhance Person request: generalists decide (../docs/editors.md)
        if "face" in intents and not ({"restore", "deblur", "sharpen", "enhance", "denoise"} & set(intents)):
            intents.pop("face")
        # "add more contrast": an adjustment, not a new object
        if "add" in intents and re.search(r"(add|them|cho them) (more |some |a bit of |a little |a touch of |chut |mot chut |them )?"
                                          r"(contrast|bright\w*|sharp\w*|detail\w*|saturation|vibran\w*|warmth|light"
                                          r"|clarity|exposure|depth|do net|do sang|tuong phan|anh sang|chi tiet)", ascii_text):
            intents.pop("add")
        # "add colour" is colourise, not add
        if "add" in intents and "colorize" in intents and len(intents["add"]) == 1:
            intents.pop("add")
        for intent in list(intents):
            intents[intent] = sorted(set(intents[intent]))

    @staticmethod
    def _params(route: Route, text: str, ascii_text: str) -> None:
        params = route.params
        if "recolor" in route.intents:
            for regex, color, accented in COLOR_RES:
                if regex.search(text if accented else ascii_text):
                    params["target_color"] = color
                    break
        if "rotate" in route.intents:
            angle = re.search(r"(\d{2,3}) ?(°|deg|do\b)", ascii_text)
            degrees = int(angle.group(1)) if angle else 90
            if re.search(r"upside down|lon nguoc|180", ascii_text):
                degrees = 180
            clockwise = bool(re.search(r"(?<!counter)(?<!counter-)clockwise|right|phai|cung chieu", ascii_text))
            params["rotate_degrees"] = -degrees if clockwise else degrees      # PIL: positive = counter-clockwise
        if "flip" in route.intents:
            params["flip"] = "vertical" if re.search(r"vertical|upside|doc|tren duoi", ascii_text) else "horizontal"
        size = re.search(r"(\d{2,5}) ?[x×] ?(\d{2,5})", ascii_text)
        if size:
            params["size"] = [int(size.group(1)), int(size.group(2))]
        percent = re.search(r"(\d{1,3}) ?%", ascii_text)
        if percent and ("resize" in route.intents or "upscale" in route.intents):
            params["scale"] = int(percent.group(1)) / 100
        factor = re.search(r"\b([2-8])x\b|x([2-8])\b", ascii_text)
        if factor:
            params["scale"] = int(factor.group(1) or factor.group(2))
        if "square" in route.intents:
            params["aspect"] = [1, 1]
        ratio = re.search(r"\b(\d{1,2}):(\d{1,2})\b", ascii_text)
        if ratio:
            params["aspect"] = [int(ratio.group(1)), int(ratio.group(2))]


ROUTER = TaskRouter()
