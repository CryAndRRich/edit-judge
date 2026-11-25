"""Scores and ranks the candidate edits of one request.

Every candidate gets a **gain**: its score minus the score of the unedited input (the "do nothing" baseline), as in
the Enhance benchmark (reports/enhance/stats.md). The winner is the candidate with the highest gain.

Backends (env EJ_WEB_SCORER = auto | editreward | imagereward | proxy):
- editreward: the repository's EditReward wrapper (src/scorer/rewards/editreward.py, Qwen2.5-VL-7B, ICLR 2026),
  scoring (original, instruction, edit). It needs about 16 GB of GPU memory, so "auto" picks it only on CUDA with
  enough memory and its package installed (requirements/scorer.txt);
- imagereward: the repository's ImageReward wrapper (BLIP, NeurIPS 2023; it reads the edit and the text only). It
  runs on a laptop (about 2 GB of weights) once `transformers<5` and `timm` are installed;
- proxy: a transparent heuristic on numpy, used when no reward model can run here. It is NOT EditReward: it
  combines instruction alignment (did the asked change happen: brighter, sharper, background gone, rotated...),
  image quality change (exposure, contrast, sharpness, colourfulness, noise, clipping) and fidelity (content
  kept). The UI says which backend scored the edits.

A reward backend that fails to load falls back to the proxy, and the reason is reported. The proxy breakdown is
computed for every candidate in every mode, as a diagnostic.
"""

import logging
import math
import os
import sys
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter, ImageOps

from .catalog import REPO_ROOT
from .router import Route

log = logging.getLogger("web_app.scorer")

PROXY_LABEL = "Proxy score (heuristic)"
GEOMETRY = {"crop", "rotate", "flip", "resize", "extend", "square"}
LOOSE_FIDELITY = {"add", "replace", "replace_bg", "remove_bg", "colorize", "bw", "recolor", "blur_bg"}
HUES = {"red": 0, "orange": 30, "yellow": 55, "green": 120, "cyan": 180, "blue": 220, "purple": 275,
        "magenta": 300, "pink": 330, "brown": 25}


# ---------------------------------------------------------------- image measurements

def flatten(image: Image.Image) -> Image.Image:
    """RGB, with transparency composited on white (what a viewer sees)."""
    if image.mode in ("RGBA", "LA", "P"):
        rgba = image.convert("RGBA")
        white = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        return Image.alpha_composite(white, rgba).convert("RGB")
    return image.convert("RGB")


def shrink(image: Image.Image, side: int) -> Image.Image:
    if max(image.size) <= side:
        return image
    scale = side / max(image.size)
    return image.resize((max(1, round(image.width * scale)), max(1, round(image.height * scale))), Image.LANCZOS)


@dataclass
class Features:
    width: int
    height: int
    luminance: float
    contrast: float
    sharpness: float
    noise: float
    colorfulness: float
    clipped: float
    warmth: float
    hue_hist: np.ndarray          # 36 bins of 10 degrees, saturated pixels only (fractions of all pixels)
    border_std: float
    border_alpha: float           # fraction of transparent pixels on the border (cut-outs)
    border_sharpness: float
    center_sharpness: float
    quality: float


def laplacian(gray: np.ndarray) -> np.ndarray:
    padded = np.pad(gray, 1, mode="edge")
    return np.abs(4 * gray - padded[:-2, 1:-1] - padded[2:, 1:-1] - padded[1:-1, :-2] - padded[1:-1, 2:])


def measure(image: Image.Image) -> Features:
    alpha = None
    if image.mode == "RGBA":
        alpha = np.asarray(shrink(image, 512).getchannel("A"), np.float32) / 255
    flat = shrink(flatten(image), 512)
    array = np.asarray(flat, np.float32) / 255
    r, g, b = array[..., 0], array[..., 1], array[..., 2]
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    lap = laplacian(lum)
    median = np.asarray(Image.fromarray((lum * 255).astype(np.uint8)).filter(ImageFilter.MedianFilter(3)),
                        np.float32) / 255
    rg, yb = r - g, 0.5 * (r + g) - b
    colorful = math.sqrt(rg.std() ** 2 + yb.std() ** 2) + 0.3 * math.sqrt(rg.mean() ** 2 + yb.mean() ** 2)
    maxc, minc = array.max(-1), array.min(-1)
    sat = np.where(maxc > 1e-3, (maxc - minc) / np.maximum(maxc, 1e-3), 0)
    delta = np.maximum(maxc - minc, 1e-6)
    hue = np.select([maxc == r, maxc == g], [((g - b) / delta) % 6, (b - r) / delta + 2], (r - g) / delta + 4) * 60
    vivid = (sat > 0.3) & (maxc > 0.15)
    hue_hist = np.histogram(hue[vivid], bins=36, range=(0, 360))[0] / max(1, lum.size)
    h, w = lum.shape
    band = max(2, min(h, w) // 12)
    border = np.zeros_like(lum, bool)
    border[:band], border[-band:], border[:, :band], border[:, -band:] = True, True, True, True
    center = np.zeros_like(lum, bool)
    center[h // 4:3 * h // 4, w // 4:3 * w // 4] = True
    features = Features(
        width=image.width, height=image.height, luminance=float(lum.mean()), contrast=float(lum.std()),
        sharpness=float(lap.mean()), noise=float(np.abs(lum - median).mean()), colorfulness=float(colorful),
        clipped=float(((lum < 0.02) | (lum > 0.98)).mean()), warmth=float((r - b).mean()), hue_hist=hue_hist,
        border_std=float(array[border].std(0).mean()),
        border_alpha=float((alpha[border] < 0.5).mean()) if alpha is not None else 0.0,
        border_sharpness=float(lap[border].mean()), center_sharpness=float(lap[center].mean()), quality=0.0)
    features.quality = quality(features)
    return features


QUALITY_WEIGHTS = {"exposure": 0.25, "contrast": 0.2, "sharp": 0.3, "color": 0.15, "noise": -0.15, "clip": -0.15}
# quality terms that the request itself asks to change (or makes meaningless) are left out of the quality change:
# "brighter" moves the exposure on purpose, "black and white" removes the colour, a cut-out flattens everything
SKIP_QUALITY = {"brighten": {"exposure", "clip"}, "darken": {"exposure", "clip"}, "bw": {"color"},
                "colorize": {"color"}, "recolor": {"color"}, "denoise": {"sharp"}, "blur_bg": {"sharp"},
                "contrast": {"clip"}, "remove_bg": {"exposure", "contrast", "sharp", "color", "clip"},
                "replace_bg": {"exposure", "contrast", "color"}, "extend": {"sharp"}}


def quality_terms(f: Features) -> dict[str, float]:
    return {"exposure": 1 - min(1.0, abs(f.luminance - 0.5) / 0.35),
            "contrast": min(1.0, f.contrast / 0.22),
            "sharp": float(np.clip(math.log(max(f.sharpness, 1e-4) / 0.005) / math.log(20), 0, 1)),
            "color": min(1.0, f.colorfulness / 0.25),
            "noise": float(np.clip(f.noise / 0.03, 0, 1)),
            "clip": float(np.clip(f.clipped * 5, 0, 1))}


def quality(f: Features, skip: set[str] = frozenset()) -> float:
    """No-reference quality in about [-0.3, 0.9]: exposure, contrast, sharpness, colour; noise and clipping cost."""
    terms = quality_terms(f)
    return sum(QUALITY_WEIGHTS[name] * value for name, value in terms.items() if name not in skip)


def gray_thumb(image: Image.Image, size: tuple[int, int] = (96, 96)) -> np.ndarray:
    return np.asarray(flatten(image).convert("L").resize(size, Image.BILINEAR), np.float32) / 255


def subject_mask(edit: Image.Image, size: tuple[int, int] = (96, 96)) -> np.ndarray:
    """Where a cut-out still shows the subject: opaque pixels of an RGBA image, or else pixels that differ from the
    (plain) colour of the edit's border."""
    if edit.mode == "RGBA":
        return np.asarray(edit.getchannel("A").resize(size, Image.BILINEAR), np.float32) / 255 > 0.5
    array = np.asarray(flatten(edit).resize(size, Image.BILINEAR), np.float32) / 255
    border = np.concatenate([array[:4].reshape(-1, 3), array[-4:].reshape(-1, 3), array[:, :4].reshape(-1, 3),
                             array[:, -4:].reshape(-1, 3)])
    return np.linalg.norm(array - np.median(border, 0), axis=-1) > 0.1


def correlation(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a - a.mean(), b - b.mean()
    denominator = math.sqrt(float((a * a).sum() * (b * b).sum()))
    return float((a * b).sum() / denominator) if denominator > 1e-9 else (1.0 if np.allclose(a, b) else 0.0)


def tanh(x: float) -> float:
    return float(math.tanh(x))


# ---------------------------------------------------------------- the proxy

class ProxyScorer:
    name = "proxy"
    label = PROXY_LABEL
    kind = "proxy"

    def breakdown(self, original: Image.Image, edit: Image.Image, route: Route,
                  original_features: Features | None = None) -> dict:
        fo = original_features or measure(original)
        fe = measure(edit)
        intents, params = route.intents, route.params
        same_size = (fo.width, fo.height) == (fe.width, fe.height)
        thumb_o, thumb_e = gray_thumb(original), gray_thumb(edit)
        color_o = np.asarray(flatten(original).resize((96, 96), Image.BILINEAR), np.float32) / 255
        color_e = np.asarray(flatten(edit).resize((96, 96), Image.BILINEAR), np.float32) / 255
        change = float(np.abs(color_o - color_e).mean())          # in colour: greyscale is a change
        unchanged = same_size and change < 0.004 and edit.mode == original.mode

        per_intent = {}
        for intent in intents or {"_generic": []}:
            per_intent[intent] = 0.0 if unchanged else round(self.align(intent, fo, fe, original, edit, params,
                                                                        thumb_o, thumb_e), 4)
        alignment = float(np.mean(list(per_intent.values()))) if per_intent else 0.0
        skip = set().union(*(SKIP_QUALITY.get(intent, set()) for intent in intents)) if intents else set()
        quality_delta = tanh((quality(fe, skip) - quality(fo, skip)) / 0.15)

        if set(intents) & GEOMETRY:
            fidelity, corr = 0.0, None             # the geometry is checked by the alignment itself
        elif "remove_bg" in intents:
            # a cut-out keeps the subject only: compare the photos where the edit still shows something
            subject = subject_mask(edit)
            if subject.mean() < 0.03:
                corr, fidelity = 0.0, -1.0                      # nothing left of the subject
            else:
                corr = correlation(thumb_o[subject], thumb_e[subject])
                fidelity = float(np.clip((corr - 0.5) / 0.5, -1, 0))
        elif set(intents) & LOOSE_FIDELITY:
            # the subject (centre) must survive a cut-out, a new background or a recolour
            h, w = thumb_o.shape
            centre = (slice(h // 4, 3 * h // 4), slice(w // 4, 3 * w // 4))
            corr = correlation(thumb_o[centre], thumb_e[centre])
            fidelity = float(np.clip((corr - 0.5) / 0.5, -1, 0))
        else:
            corr = correlation(thumb_o, thumb_e)
            fidelity = float(np.clip((corr - 0.85) / 0.15, -1, 0))

        score = 0.55 * alignment + 0.25 * quality_delta + 0.2 * fidelity
        return {"score": round(float(np.clip(score, -1, 1)), 4), "alignment": round(alignment, 4),
                "quality": round(quality_delta, 4), "fidelity": round(fidelity, 4),
                "structure_correlation": None if corr is None else round(corr, 4), "change": round(change, 4),
                "unchanged": unchanged, "intents": per_intent,
                "metrics": {"luminance": [round(fo.luminance, 3), round(fe.luminance, 3)],
                            "sharpness": [round(fo.sharpness, 4), round(fe.sharpness, 4)],
                            "colorfulness": [round(fo.colorfulness, 3), round(fe.colorfulness, 3)],
                            "noise": [round(fo.noise, 4), round(fe.noise, 4)],
                            "size": [[fo.width, fo.height], [fe.width, fe.height]]}}

    @staticmethod
    def align(intent, fo: Features, fe: Features, original, edit, params, thumb_o, thumb_e) -> float:
        ratio = lambda a, b: math.log(max(a, 1e-5) / max(b, 1e-5))
        sharpen = tanh(ratio(fe.sharpness, fo.sharpness) / 0.25)
        if intent == "brighten":
            return tanh((fe.luminance - fo.luminance) / 0.06) - float(np.clip((fe.clipped - fo.clipped) * 4, 0, 0.6))
        if intent == "darken":
            return tanh((fo.luminance - fe.luminance) / 0.06)
        if intent == "contrast":
            return tanh(((fe.contrast - fo.contrast) / 0.03 + (fe.colorfulness - fo.colorfulness) / 0.04) / 2)
        if intent in ("sharpen", "deblur", "face"):
            return sharpen
        if intent == "denoise":
            return tanh((fo.noise - fe.noise) / max(fo.noise, 1e-4) / 0.3)
        if intent == "upscale":
            pixels = math.log2(max(fe.width * fe.height, 1) / max(fo.width * fo.height, 1)) / 2   # x2 per side = 1
            return 0.7 * float(np.clip(pixels, -1, 1)) + 0.3 * sharpen
        if intent in ("enhance", "_generic"):
            return tanh((fe.quality - fo.quality) / 0.05)
        if intent == "restore":
            denoise = tanh((fo.noise - fe.noise) / max(fo.noise, 1e-4) / 0.3)
            return (denoise + sharpen + tanh((fe.quality - fo.quality) / 0.05)) / 3
        if intent == "warm":
            return tanh((fe.warmth - fo.warmth) / 0.03)
        if intent == "cool":
            return tanh((fo.warmth - fe.warmth) / 0.03)
        if intent == "bw":
            return 1 - 2 * min(1.0, fe.colorfulness / max(fo.colorfulness, 0.02))
        if intent == "colorize":
            return tanh((fe.colorfulness - fo.colorfulness) / 0.05)
        if intent == "recolor":
            target = HUES.get(params.get("target_color", ""), None)
            if target is None:
                return tanh(float(np.abs(fe.hue_hist - fo.hue_hist).sum()) / 0.1)
            bins = [((target + d) % 360) // 10 for d in range(-20, 21, 10)]
            gained = float(fe.hue_hist[bins].sum() - fo.hue_hist[bins].sum())
            return tanh(gained / 0.06)
        if intent == "remove_bg":
            if edit.mode == "RGBA":
                return 2 * fe.border_alpha - 1
            uniform = tanh((fo.border_std - fe.border_std) / 0.04)
            return uniform
        if intent == "replace_bg":
            h, w = thumb_o.shape
            border = np.ones_like(thumb_o, bool)
            border[h // 5:4 * h // 5, w // 5:4 * w // 5] = False
            diff = np.abs(thumb_o - thumb_e)
            return tanh((diff[border].mean() - diff[~border].mean()) / 0.04)
        if intent == "blur_bg":
            drop = (fo.border_sharpness - fe.border_sharpness) / max(fo.border_sharpness, 1e-5)
            kept = fe.center_sharpness / max(fo.center_sharpness, 1e-5)
            return tanh(drop / 0.3) * min(1.0, kept)
        if intent == "remove_object":
            if thumb_o.shape != thumb_e.shape:
                return -0.5
            diff = np.abs(thumb_o - thumb_e)
            changed = float((diff > 0.06).mean())            # the share of the photo that changed visibly
            if changed < 0.003:
                return -0.2                                  # nothing removed
            local = 1 - abs(math.log(changed / 0.06)) / math.log(10)   # best around 6% of the photo
            return float(np.clip(local, -1, 1))
        if intent in ("add", "replace"):
            change = float(np.abs(thumb_o - thumb_e).mean())
            return float(np.clip(1 - abs(math.log(max(change, 1e-4) / 0.05)) / math.log(5), -1, 1))
        if intent in ("crop", "square"):
            # scale-invariant: the edit must show a part of the photo (match a centre crop better than the whole)
            aspect = params.get("aspect")
            wanted = aspect[0] / aspect[1] if aspect else (fo.width / fo.height)
            crop_w, crop_h = (fo.width, fo.height)
            if aspect:
                crop_w, crop_h = ((round(fo.height * wanted), fo.height) if fo.width / fo.height > wanted
                                  else (fo.width, round(fo.width / wanted)))
            else:
                crop_w, crop_h = round(fo.width * 0.85), round(fo.height * 0.85)
            left, top = (fo.width - crop_w) // 2, (fo.height - crop_h) // 2
            cropped = gray_thumb(flatten(original).crop((left, top, left + crop_w, top + crop_h)))
            zoomed_in = correlation(cropped, thumb_e) - correlation(thumb_o, thumb_e)
            fits = abs(math.log((fe.width / fe.height) / wanted)) < 0.05
            aspect_changed = abs(math.log((fe.width / fe.height) / (fo.width / fo.height))) > 0.05
            if aspect:
                return (0.5 if fits else -0.5) + (0.5 if zoomed_in > -0.05 and aspect_changed else -0.3)
            return tanh(zoomed_in / 0.05)
        if intent == "rotate":
            degrees = params.get("rotate_degrees", 90) % 360
            reference = original.rotate(degrees, expand=True)
            matches = correlation(gray_thumb(reference), thumb_e)
            stays = correlation(thumb_o, thumb_e) if (fo.width, fo.height) == (fe.width, fe.height) else -1
            return tanh((matches - max(stays, 0.3)) / 0.15)
        if intent == "flip":
            mirrored = ImageOps.flip(original) if params.get("flip") == "vertical" else ImageOps.mirror(original)
            return tanh((correlation(gray_thumb(mirrored), thumb_e) - correlation(thumb_o, thumb_e)) / 0.1)
        if intent == "resize":
            if params.get("size"):
                wanted_w, wanted_h = params["size"]
                return 1.0 if (fe.width, fe.height) == (wanted_w, wanted_h) else \
                    float(np.clip(1 - abs(math.log(fe.width / wanted_w)) - abs(math.log(fe.height / wanted_h)), -1, 1))
            scale = params.get("scale")
            actual = fe.width / fo.width
            if scale:
                return float(np.clip(1 - abs(math.log(actual / scale)) / math.log(2), -1, 1))
            return 1.0 if abs(math.log(actual)) > 0.1 else -0.5
        if intent == "extend":
            # scale-invariant: the whole original must sit centred inside a larger canvas
            flat, best_corr, best_fraction = flatten(edit), -1.0, 1.0
            for fraction in np.linspace(0.6, 1.0, 9):
                inner_w = round(fe.width * fraction)
                inner_h = round(inner_w * fo.height / fo.width)
                if inner_h > fe.height or inner_w < 8 or inner_h < 8:
                    continue
                left, top = (fe.width - inner_w) // 2, (fe.height - inner_h) // 2
                value = correlation(gray_thumb(flat.crop((left, top, left + inner_w, top + inner_h))), thumb_o)
                if value > best_corr:
                    best_corr, best_fraction = value, float(fraction)
            if best_fraction > 0.97 or best_corr < 0.5:
                return -0.6                                  # not extended (or the content changed)
            return 0.4 + 0.6 * best_corr
        return 0.0


# ---------------------------------------------------------------- reward models of the repository

class RewardModelScorer:
    """EditReward or ImageReward through src/scorer/rewards (the same `score_batch` as the benchmark)."""
    kind = "reward_model"

    def __init__(self, name: str):
        if str(REPO_ROOT) not in sys.path:
            sys.path.insert(0, str(REPO_ROOT))
        from src.scorer.rewards import get_reward_model
        self.name = name
        self.model = get_reward_model(name)
        self.label = {"editreward": "EditReward (Qwen2.5-VL-7B)", "imagereward": "ImageReward (BLIP)"}[name]
        self.device = None
        self._lock = threading.Lock()

    def load(self) -> None:
        import torch
        if self.device:
            return
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        if self.name == "editreward" and device != "cuda":
            raise RuntimeError("EditReward needs CUDA GPUs with about 16 GB of memory (Qwen2.5-VL-7B in fp16)")
        self.model.load(device)
        self.device = device
        log.info("reward model %s loaded on %s", self.name, device)

    def score(self, original: Image.Image, text: str, edits: list[Image.Image]) -> list[float | None]:
        from src.scorer.rewards.base import EditItem, score_batch
        original = flatten(original)
        scores: list[float | None] = []
        with self._lock:
            self.load()
            items = [(EditItem(f"web/{index}", Path("original"), Path(f"edit-{index}"), text or "improve the photo"),
                      original, flatten(edit)) for index, edit in enumerate(edits)]
            for start in range(0, len(items), self.model.batch_size):
                results, _ = score_batch(self.model, items[start:start + self.model.batch_size])
                scores.extend(result.get("score") for result in results)
        return [score if score is not None and math.isfinite(score) else None for score in scores]


def cuda_memory_gb() -> float:
    try:
        import torch
        if torch.cuda.is_available():
            return sum(torch.cuda.get_device_properties(i).total_memory for i in range(torch.cuda.device_count())) / 1e9
    except Exception:
        pass
    return 0.0


# ---------------------------------------------------------------- the scorer used by the app

class ImageScorer:
    def __init__(self, mode: str | None = None):
        self.mode = (mode or os.environ.get("EJ_WEB_SCORER", "auto")).lower()
        self.proxy = ProxyScorer()
        self.backend = None                    # a RewardModelScorer, or None for the proxy
        self.fallback_reason = None
        wanted = self.mode
        if wanted == "auto":
            wanted = "editreward" if cuda_memory_gb() >= 15 else "proxy"
            if wanted == "proxy":
                self.fallback_reason = ("EditReward (Qwen2.5-VL-7B) needs about 16 GB of CUDA memory; this machine "
                                        "has none, so the proxy scores the edits. Set EJ_WEB_SCORER=imagereward "
                                        "to use ImageReward here.")
        if wanted in ("editreward", "imagereward"):
            try:
                self.backend = RewardModelScorer(wanted)
            except BaseException as error:       # SystemExit from the registry, ImportError, ...
                self.fallback_reason = f"{wanted} unavailable ({type(error).__name__}: {error}); proxy used"
                log.warning(self.fallback_reason)
        elif wanted != "proxy":
            self.fallback_reason = f"unknown scorer {wanted!r}; proxy used"

    def warmup(self) -> None:
        """Load the reward model now (server start, in a thread), so the first request does not wait for it; a
        model that cannot load is replaced by the proxy at once, with the reason."""
        backend = self.backend
        if backend is None:
            return
        try:
            with backend._lock:
                backend.load()
        except BaseException as error:
            if isinstance(error, KeyboardInterrupt):
                raise
            self.fallback_reason = f"{backend.name} could not load ({type(error).__name__}: {error}); proxy used"
            log.warning(self.fallback_reason)
            self.backend = None

    def info(self) -> dict:
        active = self.backend or self.proxy
        return {"mode": self.mode, "backend": active.name, "label": active.label, "kind": active.kind,
                "fallback_reason": self.fallback_reason}

    def score_candidates(self, original: Image.Image, route: Route, candidates: list[dict]) -> list[dict]:
        """Adds `score` (gain over the unedited input, raw score, baseline) and `proxy` (breakdown) to every
        candidate with an `image`; returns the candidates."""
        original_features = measure(original)
        scored = [c for c in candidates if c.get("image") is not None]
        for candidate in scored:
            candidate["proxy"] = self.proxy.breakdown(original, candidate["image"], route, original_features)

        if self.backend is not None:
            try:
                values = self.backend.score(original, route.instruction, [original] + [c["image"] for c in scored])
                baseline, raw = values[0], values[1:]
                for candidate, value in zip(scored, raw):
                    gain = None if value is None or baseline is None else round(value - baseline, 5)
                    candidate["score"] = {"gain": gain, "raw": value, "baseline": baseline,
                                          "scorer": self.backend.name}
                return candidates
            except BaseException as error:        # a load failure (SystemExit included) must not lose the run
                if isinstance(error, KeyboardInterrupt):
                    raise
                self.fallback_reason = f"{self.backend.name} failed ({type(error).__name__}: {error}); proxy used"
                log.exception("reward model failed, falling back to the proxy")
                self.backend = None

        for candidate in scored:
            value = candidate["proxy"]["score"]
            candidate["score"] = {"gain": value, "raw": value, "baseline": 0.0, "scorer": "proxy"}
        return candidates


def rank_candidates(candidates: list[dict]) -> tuple[dict | None, list[dict]]:
    """(best candidate, all candidates sorted best first). Ties: the faster model first. Failed or unscored
    candidates come last, unranked."""
    def key(candidate):
        gain = (candidate.get("score") or {}).get("gain")
        valid = gain is not None and math.isfinite(gain)
        return (0 if valid else 1, -(gain if valid else 0.0), candidate.get("elapsed_ms") or 0.0)

    ranked = sorted(candidates, key=key)
    position = 0
    for candidate in ranked:
        gain = (candidate.get("score") or {}).get("gain")
        if gain is not None and math.isfinite(gain):
            position += 1
            candidate["rank"] = position
        else:
            candidate["rank"] = None
    best = ranked[0] if ranked and ranked[0]["rank"] == 1 else None
    return best, ranked
