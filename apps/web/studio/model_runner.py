"""Runs the candidate editing models on one image.

Two kinds of handler share one interface (`EditingModel.edit(image, route) -> PIL image`):
- real models: Real-ESRGAN x4plus (spandrel, on Apple MPS, CUDA or the CPU; weights downloaded once) and plain PIL
  code (crop, resize, rotate, flip, greyscale);
- mock models: a Pillow/numpy simulation of what each model does (../docs/editors.md), with a per-model profile (edit
  strength, softness, colour drift), so the pipeline, the scorer and the UI can be built before the weights run.
  To wire in a real model, subclass EditingModel, implement `edit`, and register it in `HANDLERS`.

`ModelRunner.run_one` runs one model and measures it; a failure is returned, never raised. The assistant runs the
models one after another (pipeline.py). A mock waits about as long as the real model takes on the server
(`estimate`, from the T4 benchmark), so the progress bars behave as they will there; EJ_WEB_MOCK_DELAY=0 turns the
wait off.
"""

import hashlib
import logging
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageOps

from .catalog import MODELS_BY_ID, WEB_DIR
from .router import Route

log = logging.getLogger("studio.models")

WORK_MAX_SIDE = 1024                      # mocks work at most at this size
CACHE_DIR = Path(os.environ.get("EJ_WEB_CACHE") or WEB_DIR / "studio" / ".cache")
ESRGAN_URL = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
ESRGAN_MAX_SIDE = int(os.environ.get("EJ_WEB_ESRGAN_MAX_SIDE", "512"))   # x4 output: 2048 px at most
REAL_MODELS_ENABLED = os.environ.get("EJ_WEB_REAL_MODELS", "1") != "0"
MOCK_DELAY = os.environ.get("EJ_WEB_MOCK_DELAY", "1") != "0"

HUES = {"red": 0, "orange": 30, "yellow": 55, "green": 120, "cyan": 180, "blue": 220, "purple": 275,
        "magenta": 300, "pink": 330, "brown": 25}


# ---------------------------------------------------------------- image operations (numpy / PIL)

def to_array(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0


def to_image(array: np.ndarray) -> Image.Image:
    return Image.fromarray((np.clip(array, 0, 1) * 255 + 0.5).astype(np.uint8), "RGB")


def blend(a: Image.Image, b: Image.Image, amount: float) -> Image.Image:
    if b.size != a.size:
        b = b.resize(a.size, Image.LANCZOS)
    return Image.blend(a.convert("RGB"), b.convert("RGB"), float(np.clip(amount, 0, 1)))


def limit_size(image: Image.Image, max_side: int) -> Image.Image:
    if max(image.size) <= max_side:
        return image
    scale = max_side / max(image.size)
    return image.resize((max(1, round(image.width * scale)), max(1, round(image.height * scale))), Image.LANCZOS)


def rgb_to_hsv(array: np.ndarray) -> np.ndarray:
    r, g, b = array[..., 0], array[..., 1], array[..., 2]
    maxc, minc = array.max(-1), array.min(-1)
    v, delta = maxc, maxc - minc
    s = np.where(maxc > 1e-6, delta / np.maximum(maxc, 1e-6), 0)
    safe = np.maximum(delta, 1e-6)
    h = np.select([maxc == r, maxc == g], [((g - b) / safe) % 6, (b - r) / safe + 2], (r - g) / safe + 4)
    h = np.where(delta > 1e-6, h / 6.0, 0)
    return np.stack([h, s, v], -1)


def hsv_to_rgb(hsv: np.ndarray) -> np.ndarray:
    h, s, v = hsv[..., 0] * 6, hsv[..., 1], hsv[..., 2]
    i = np.floor(h).astype(int) % 6
    f = h - np.floor(h)
    p, q, t = v * (1 - s), v * (1 - s * f), v * (1 - s * (1 - f))
    choices = [np.stack(c, -1) for c in ((v, t, p), (q, v, p), (p, v, t), (p, q, v), (t, p, v), (v, p, q))]
    out = np.zeros_like(hsv)
    for index, choice in enumerate(choices):
        out = np.where((i == index)[..., None], choice, out)
    return out


def otsu(values: np.ndarray) -> float:
    hist, edges = np.histogram(values, bins=64)
    centers = (edges[:-1] + edges[1:]) / 2
    weight = np.cumsum(hist)
    total = weight[-1]
    mean = np.cumsum(hist * centers)
    between = (mean[-1] * weight - mean * total) ** 2 / np.maximum(weight * (total - weight), 1e-9)
    return float(centers[int(np.argmax(between))])


def foreground_mask(image: Image.Image) -> np.ndarray:
    """A soft foreground mask (H x W, 0..1): colour distance from the border statistics, with a centre prior."""
    small = limit_size(image.convert("RGB"), 256)
    array = to_array(small.filter(ImageFilter.GaussianBlur(1.5)))
    h, w, _ = array.shape
    band = max(3, min(h, w) // 16)
    border = np.concatenate([array[:band].reshape(-1, 3), array[-band:].reshape(-1, 3),
                             array[:, :band].reshape(-1, 3), array[:, -band:].reshape(-1, 3)])
    mean, std = border.mean(0), border.std(0)
    distance = np.linalg.norm((array - mean) / (std + 0.06), axis=-1)
    ys, xs = np.mgrid[0:h, 0:w]
    prior = np.exp(-(((xs / w - 0.5) / 0.35) ** 2 + ((ys / h - 0.5) / 0.4) ** 2))
    saliency = distance * (0.35 + 0.65 * prior)
    mask = (saliency > otsu(saliency)).astype(np.uint8) * 255
    mask_image = Image.fromarray(mask, "L").filter(ImageFilter.MaxFilter(5)).filter(ImageFilter.MinFilter(5))
    mask_image = fill_holes(mask_image)
    mask_image = mask_image.filter(ImageFilter.GaussianBlur(2)).resize(image.size, Image.BILINEAR)
    return np.asarray(mask_image, dtype=np.float32) / 255.0


def fill_holes(mask: Image.Image) -> Image.Image:
    """Background regions not connected to the border become foreground (a dark face inside a light blanket)."""
    padded = ImageOps.expand(mask.point(lambda v: 255 if v > 127 else 0), border=1, fill=0)
    ImageDraw.floodfill(padded, (0, 0), 128)
    array = np.asarray(padded)[1:-1, 1:-1]
    return Image.fromarray(np.where(array == 128, 0, 255).astype(np.uint8), "L")


def detail_region(image: Image.Image, fraction: float = 0.2) -> tuple[int, int, int, int]:
    """The box (x0, y0, x1, y1) of the window that differs most in colour from its surroundings (centre-surround
    contrast) and is detailed: where a small, distinct object most likely is. A stand-in for Grounding DINO."""
    small = limit_size(image.convert("RGB"), 192)
    array = to_array(small)
    h, w, _ = array.shape
    win_h, win_w = max(4, int(h * fraction)), max(4, int(w * fraction))

    def box_mean(values: np.ndarray, kh: int, kw: int) -> np.ndarray:
        integral = np.pad(values.cumsum(0).cumsum(1), ((1, 0), (1, 0)) + ((0, 0),) * (values.ndim - 2))
        sums = integral[kh:, kw:] - integral[:-kh, kw:] - integral[kh:, :-kw] + integral[:-kh, :-kw]
        return sums / (kh * kw)

    inner = box_mean(array, win_h, win_w)                         # windows of the object size
    outer = box_mean(array, 2 * win_h, 2 * win_w)                 # their surroundings (twice the size)
    oy, ox = win_h // 2, win_w // 2
    rows, cols = outer.shape[0], outer.shape[1]
    inner = inner[oy:oy + rows, ox:ox + cols]
    contrast = np.linalg.norm(inner - outer, axis=-1)
    gray = array.mean(-1)
    gradient = np.abs(np.diff(gray, axis=0, prepend=gray[:1])) + np.abs(np.diff(gray, axis=1, prepend=gray[:, :1]))
    detail = box_mean(gradient, win_h, win_w)[oy:oy + rows, ox:ox + cols]
    score = contrast * (0.5 + detail / (detail.max() + 1e-6))
    y, x = np.unravel_index(int(np.argmax(score)), score.shape)
    y, x = y + oy, x + ox                                         # back to inner-window coordinates
    sx, sy = image.width / w, image.height / h
    return int(x * sx), int(y * sy), int((x + win_w) * sx), int((y + win_h) * sy)


def inpaint(image: Image.Image, mask: np.ndarray, radius: int = 24) -> Image.Image:
    """Fills the mask from its surroundings (normalised convolution): a stand-in for LaMa-style inpainting."""
    array = to_array(image)
    keep = 1 - mask
    scale = max(1, max(image.size) // 512)
    blur = lambda a: np.asarray(Image.fromarray((a * 255).astype(np.uint8)).filter(
        ImageFilter.GaussianBlur(radius * scale)), dtype=np.float32) / 255
    numerator = np.stack([blur(np.clip(array[..., c] * keep, 0, 1)) for c in range(3)], -1)
    denominator = blur(keep)[..., None]
    fill = numerator / np.maximum(denominator, 1e-3)
    grain = (np.random.default_rng(0).normal(0, 0.012, array.shape)).astype(np.float32)
    return to_image(array * keep[..., None] + (fill + grain) * mask[..., None])


def ellipse_mask(size: tuple[int, int], box: tuple[int, int, int, int], feather: float = 0.25) -> np.ndarray:
    w, h = size
    x0, y0, x1, y1 = box
    ys, xs = np.mgrid[0:h, 0:w]
    cx, cy, rx, ry = (x0 + x1) / 2, (y0 + y1) / 2, max(2, (x1 - x0) / 2), max(2, (y1 - y0) / 2)
    d = np.sqrt(((xs - cx) / rx) ** 2 + ((ys - cy) / ry) ** 2)
    return np.clip((1 + feather - d) / feather, 0, 1).astype(np.float32)


def composite(foreground: Image.Image, mask: np.ndarray, background: Image.Image) -> Image.Image:
    fg, bg = to_array(foreground), to_array(background.resize(foreground.size))
    return to_image(fg * mask[..., None] + bg * (1 - mask[..., None]))


def cutout(image: Image.Image, mask: np.ndarray, transparent: bool) -> Image.Image:
    if transparent:
        rgba = image.convert("RGBA")
        rgba.putalpha(Image.fromarray((mask * 255).astype(np.uint8), "L"))
        return rgba
    return composite(image, mask, Image.new("RGB", image.size, (255, 255, 255)))


def recolor(image: Image.Image, color: str, mask: np.ndarray | None, strength: float = 1.0) -> Image.Image:
    array = to_array(image)
    if color in ("white", "black"):
        target = np.full_like(array, 0.95 if color == "white" else 0.08)
        lum = array.mean(-1, keepdims=True)
        out = target * 0.6 + lum * 0.4
    else:
        hsv = rgb_to_hsv(array)
        hsv[..., 0] = HUES.get(color, 0) / 360
        hsv[..., 1] = np.clip(np.maximum(hsv[..., 1], 0.55), 0, 1)
        out = hsv_to_rgb(hsv)
    weight = (mask if mask is not None else np.ones(array.shape[:2], np.float32))[..., None] * strength
    return to_image(array * (1 - weight) + out * weight)


def colorize(image: Image.Image, strength: float = 1.0) -> Image.Image:
    """Luminance-mapped toning (cool shadows, warm mid-tones and highlights): a stand-in for DDColor."""
    gray = np.asarray(image.convert("L"), dtype=np.float32) / 255
    stops = np.array([0.0, 0.35, 0.7, 1.0])
    palette = np.array([[0.05, 0.07, 0.12], [0.30, 0.33, 0.38], [0.78, 0.64, 0.50], [1.0, 0.97, 0.92]])
    color = np.stack([np.interp(gray, stops, palette[:, c]) for c in range(3)], -1)
    original = to_array(image)
    return to_image(original * (1 - strength) + color * strength)


def outpaint(image: Image.Image, amount: float = 0.25) -> Image.Image:
    """Extends the canvas with a blurred, stretched copy of the photo: a stand-in for SDXL outpainting."""
    w, h = image.size
    pad_x = int(w * amount / 2)
    pad_y = int(h * amount / 2) if h > w else int(h * amount / 4)
    size = (w + 2 * pad_x, h + 2 * pad_y)
    stretched = image.resize(size, Image.BILINEAR).filter(ImageFilter.GaussianBlur(14))
    canvas = stretched.copy()
    canvas.paste(image, (pad_x, pad_y))
    margin = max(4, min(w, h) // 40)                    # feather inside the photo, so no seam shows
    mask = Image.new("L", size, 0)
    ImageDraw.Draw(mask).rectangle((pad_x + margin, pad_y + margin, pad_x + w - margin, pad_y + h - margin), fill=255)
    mask = mask.filter(ImageFilter.GaussianBlur(margin / 1.5))
    return Image.composite(canvas, stretched, mask)


def center_crop(image: Image.Image, aspect: tuple[int, int] | None, keep: float = 0.85) -> Image.Image:
    w, h = image.size
    if aspect:
        ratio = aspect[0] / aspect[1]
        new_w, new_h = (int(h * ratio), h) if w / h > ratio else (w, int(w / ratio))
    else:
        new_w, new_h = int(w * keep), int(h * keep)
    left, top = (w - new_w) // 2, (h - new_h) // 2
    return image.crop((left, top, left + new_w, top + new_h))


def channel_gain(image: Image.Image, gains: tuple[float, float, float]) -> Image.Image:
    return to_image(to_array(image) * np.array(gains, np.float32))


def gamma(image: Image.Image, value: float) -> Image.Image:
    return to_image(np.power(to_array(image), value))


def painterly(image: Image.Image) -> Image.Image:
    return ImageEnhance.Color(image.filter(ImageFilter.ModeFilter(7)).filter(ImageFilter.EDGE_ENHANCE)).enhance(1.35)


# ---------------------------------------------------------------- the edit "recipes" shared by the mocks

def apply_intents(image: Image.Image, route: Route, strength: float = 1.0, transparent: bool = False,
                  upscale: float = 1.0) -> Image.Image:
    """What an instruction-following editor would do, intent by intent, at `strength` (0..1.5)."""
    intents, params = route.intents, route.params
    out = image.convert("RGB")
    s = strength
    if "remove_bg" in intents:
        out = cutout(out, foreground_mask(out), transparent)
        if transparent:
            return out
    if "replace_bg" in intents:
        mask = foreground_mask(out)
        background = ImageEnhance.Color(out.filter(ImageFilter.GaussianBlur(18))).enhance(1.6)
        background = channel_gain(background, (0.85, 1.0, 1.25))
        out = composite(out, mask, background)
    if "blur_bg" in intents:
        out = composite(out, foreground_mask(out), out.filter(ImageFilter.GaussianBlur(6 + 6 * s)))
    if "remove_object" in intents:
        out = inpaint(out, ellipse_mask(out.size, detail_region(out)))
    if "add" in intents or "replace" in intents:
        style = route.instruction.lower()
        if any(word in style for word in ("paint", "van gogh", "cartoon", "anime", "sketch", "tranh", "hoạt hình")):
            out = blend(out, painterly(out), 0.75 * s)
        else:
            mask = foreground_mask(out) * ellipse_mask(out.size, (out.width // 4, 0, 3 * out.width // 4,
                                                                  out.height // 2))
            hsv = rgb_to_hsv(to_array(out))
            hsv[..., 0] = (hsv[..., 0] + 0.12 * s) % 1
            hsv[..., 1] = np.clip(hsv[..., 1] * 1.3, 0, 1)
            edited = hsv_to_rgb(hsv)
            out = to_image(to_array(out) * (1 - mask[..., None]) + edited * mask[..., None])
    if "recolor" in intents:
        out = recolor(out, params.get("target_color", "red"), foreground_mask(out), min(1.0, 0.85 * s))
    if "bw" in intents:
        gray = ImageOps.grayscale(out).convert("RGB")
        if "sepia" in route.instruction.lower():
            gray = channel_gain(gray, (1.07, 0.95, 0.78))
        out = gray
    if "colorize" in intents:
        out = colorize(out, min(1.0, 0.9 * s))
    if "brighten" in intents:
        out = ImageEnhance.Brightness(gamma(out, 1 - 0.28 * s)).enhance(1 + 0.08 * s)
    if "darken" in intents:
        out = ImageEnhance.Brightness(gamma(out, 1 + 0.3 * s)).enhance(1 - 0.05 * s)
    if "contrast" in intents:
        out = ImageEnhance.Color(ImageEnhance.Contrast(out).enhance(1 + 0.22 * s)).enhance(1 + 0.3 * s)
    if "warm" in intents:
        out = channel_gain(out, (1 + 0.07 * s, 1 + 0.02 * s, 1 - 0.07 * s))
    if "cool" in intents:
        out = channel_gain(out, (1 - 0.06 * s, 1.0, 1 + 0.07 * s))
    if "restore" in intents:
        out = out.filter(ImageFilter.MedianFilter(3))
        out = ImageOps.autocontrast(out, cutoff=1)
        out = ImageEnhance.Color(out).enhance(1 + 0.2 * s)
        out = out.filter(ImageFilter.UnsharpMask(radius=2, percent=int(90 * s), threshold=3))
    if "denoise" in intents:
        out = blend(out, out.filter(ImageFilter.MedianFilter(3)).filter(ImageFilter.SMOOTH_MORE), 0.8 * min(s, 1))
    if "deblur" in intents or "face" in intents:
        out = out.filter(ImageFilter.UnsharpMask(radius=3, percent=int(130 * s), threshold=2))
    if "sharpen" in intents:
        out = out.filter(ImageFilter.UnsharpMask(radius=2, percent=int(150 * s), threshold=2))
    if "enhance" in intents:
        out = ImageOps.autocontrast(out, cutoff=0.5)
        out = ImageEnhance.Color(out).enhance(1 + 0.12 * s)
        out = out.filter(ImageFilter.UnsharpMask(radius=1.5, percent=int(70 * s), threshold=3))
    if "upscale" in intents and upscale > 1:
        out = out.resize((round(out.width * upscale), round(out.height * upscale)), Image.LANCZOS)
        out = out.filter(ImageFilter.UnsharpMask(radius=2, percent=int(110 * s), threshold=2))
    out = geometry(out, route)
    if not intents:                       # no recognised intent: a mild generalist "improvement"
        out = ImageOps.autocontrast(out, cutoff=0.5)
        out = ImageEnhance.Color(out).enhance(1 + 0.1 * s)
    return out


def geometry(image: Image.Image, route: Route) -> Image.Image:
    intents, params = route.intents, route.params
    out = image
    if "crop" in intents or "square" in intents or ("aspect" in params and "extend" not in intents):
        out = center_crop(out, tuple(params["aspect"]) if params.get("aspect") else None)
    if "rotate" in intents:
        out = out.rotate(params.get("rotate_degrees", 90), expand=True, resample=Image.BICUBIC,
                         fillcolor=(255, 255, 255) if out.mode == "RGB" else None)
    if "flip" in intents:
        out = ImageOps.flip(out) if params.get("flip") == "vertical" else ImageOps.mirror(out)
    if "resize" in intents:
        if params.get("size"):
            out = out.resize(tuple(params["size"]), Image.LANCZOS)
        else:
            scale = params.get("scale") or (0.5 if any(w in route.instruction.lower() for w in
                                                       ("smaller", "shrink", "thu nhỏ", "nhỏ hơn", "down")) else 2)
            out = out.resize((max(1, round(out.width * scale)), max(1, round(out.height * scale))), Image.LANCZOS)
    if "extend" in intents:
        out = outpaint(out)
    return out


# ---------------------------------------------------------------- models

class EditingModel:
    """One candidate. `edit` gets an RGB image (at most WORK_MAX_SIDE for mocks) and the route."""
    backend = "mock"
    max_side = WORK_MAX_SIDE

    def __init__(self, model_id: str):
        self.model_id = model_id
        self.info = MODELS_BY_ID[model_id]

    def edit(self, image: Image.Image, route: Route) -> Image.Image:
        raise NotImplementedError



@dataclass(frozen=True)
class Profile:
    strength: float = 1.0       # how far the edit goes
    softness: float = 0.0       # 0..1: detail lost (diffusion VAE round trip, low working resolution)
    drift: float = 0.0          # unasked global colour change (classic diffusion editors)
    work_side: int | None = None


GENERALIST_PROFILES = {
    "realedit": Profile(strength=0.85, softness=0.3, drift=0.05, work_side=512),
    "magicbrush": Profile(strength=0.7, softness=0.35, drift=0.1, work_side=512),
}


class GeneralistMock(EditingModel):
    def edit(self, image, route):
        profile = GENERALIST_PROFILES[self.model_id]
        size = image.size
        work = limit_size(image, profile.work_side) if profile.work_side else image
        out = apply_intents(work, route, profile.strength)
        geometric = bool(set(route.intents) & {"crop", "rotate", "flip", "resize", "extend", "square"})
        if profile.work_side and not geometric:
            out = out.resize(size, Image.BICUBIC)          # back to the input size, as the real pipelines do
        if profile.softness:
            out = blend(out, out.filter(ImageFilter.GaussianBlur(1.2)), profile.softness)
        if profile.drift:
            seed = int(hashlib.md5((self.model_id + route.instruction).encode()).hexdigest()[:8], 16)
            direction = 1 if seed % 2 else -1
            hsv = rgb_to_hsv(to_array(out))
            hsv[..., 0] = (hsv[..., 0] + direction * profile.drift * 0.25) % 1
            hsv[..., 1] = np.clip(hsv[..., 1] * (1 + profile.drift), 0, 1)
            out = to_image(hsv_to_rgb(hsv))
        return out


REMOVAL_PROFILES = {"objectclear": (0.2, 22), "osor": (0.19, 20), "rorem": (0.18, 26), "lama": (0.16, 18),
                    "migan": (0.15, 16)}      # (mask size as a share of the image, inpainting radius)


class RemovalMock(EditingModel):
    """ObjectClear, OSOR, RORem, LaMa, MI-GAN: inpaint the region a mask would select (here: the most detailed
    window)."""
    def edit(self, image, route):
        fraction, radius = REMOVAL_PROFILES[self.model_id]
        out = inpaint(image, ellipse_mask(image.size, detail_region(image, fraction)), radius=radius)
        if self.model_id in ("rorem", "osor"):
            out = blend(out, out.filter(ImageFilter.GaussianBlur(0.8)), 0.25)       # SDXL VAE round trip at 512
        return out



class CutoutMock(EditingModel):
    """BEN2, BiRefNet: foreground matting; RGBA output (transparent background)."""
    def edit(self, image, route):
        mask = foreground_mask(image)
        if self.model_id == "ben2":
            mask = np.clip((mask - 0.2) / 0.6, 0, 1)                  # crisper edges
        if "replace_bg" in route.intents:
            return composite(image, mask, Image.new("RGB", image.size, (235, 238, 245)))
        return cutout(image, mask, transparent=True)


class SDXLInpaintMock(EditingModel):
    """SDXL Inpainting (add, replace, recolour one region) and SDXL Outpainting (extend)."""
    def edit(self, image, route):
        work = limit_size(image, 1024)
        if self.model_id == "sdxl-outpaint" or "extend" in route.intents:
            return outpaint(work)
        if "recolor" in route.intents:
            return recolor(work, route.params.get("target_color", "red"), foreground_mask(work), 0.9)
        return apply_intents(work, Route(route.instruction, {k: v for k, v in route.intents.items()
                                                             if k in ("add", "replace", "replace_bg")}
                                         or {"replace": ["edit"]}, {}, route.params), 1.0)



class EnhanceMock(EditingModel):
    def edit(self, image, route):
        intents = set(route.intents)
        mid = self.model_id
        if mid == "vosr":                       # generative SR: sharpen (x1) or x2 for upscale requests
            out = image.filter(ImageFilter.UnsharpMask(radius=2, percent=140, threshold=2))
            out = blend(out, out.filter(ImageFilter.DETAIL), 0.5)
            if "upscale" in intents:
                out = out.resize((image.width * 2, image.height * 2), Image.LANCZOS).filter(
                    ImageFilter.UnsharpMask(radius=2, percent=80, threshold=2))
            return out
        if mid == "hvi-cidnet":                  # low-light: lifts the shadows strongly
            out = gamma(image, 0.62)
            return ImageEnhance.Contrast(out).enhance(1.08)
        if mid == "instructir":                  # the degradation named in the instruction
            out = image
            if intents & {"denoise"}:
                out = out.filter(ImageFilter.MedianFilter(3))
            if intents & {"deblur", "sharpen", "upscale"}:
                out = out.filter(ImageFilter.UnsharpMask(radius=2, percent=90, threshold=3))
            if intents & {"brighten"}:
                out = gamma(out, 0.8)
            if intents & {"enhance", "contrast"} or out is image:
                out = ImageEnhance.Contrast(ImageOps.autocontrast(out, cutoff=0.3)).enhance(1.05)
            return out
        raise KeyError(mid)


class RestoreMock(EditingModel):
    def edit(self, image, route):
        mid = self.model_id
        # CodeFormer, GFPGAN: face restorers; mostly sharpen + smooth skin (whole image, no detector here)
        smooth = image.filter(ImageFilter.SMOOTH_MORE)
        sharp = smooth.filter(ImageFilter.UnsharpMask(radius=2, percent=110 if mid == "gfpgan" else 90, threshold=2))
        return blend(image, sharp, 0.85 if mid == "codeformer" else 1.0)



class AddMock(EditingModel):
    """EraseDraw, Paint by Inpaint: mask-free addition, simulated like a generalist that only adds."""
    def edit(self, image, route):
        work = limit_size(image, 512)
        add = {k: v for k, v in route.intents.items() if k == "add"} or {"add": ["object"]}
        out = apply_intents(work, Route(route.instruction, add, {}, route.params), 0.9)
        return out.resize(image.size, Image.BICUBIC)


class DDColorMock(EditingModel):
    def edit(self, image, route):
        return colorize(image, 0.85)


class PlainCode(EditingModel):
    """List 7's plain code: exact geometry and greyscale with Pillow. Does nothing for other requests."""
    backend = "real"
    max_side = 4096

    def edit(self, image, route):
        out = image
        if "bw" in route.intents:
            out = ImageOps.grayscale(out).convert("RGB")
        return geometry(out, route)


class RealESRGAN(EditingModel):
    """Real-ESRGAN x4plus, loaded with spandrel (../docs/editors.md). The input is limited to ESRGAN_MAX_SIDE."""
    backend = "real"
    max_side = ESRGAN_MAX_SIDE
    _model = None
    _device = None
    _lock = threading.Lock()

    @classmethod
    def weights(cls) -> Path:
        path = CACHE_DIR / "models" / "RealESRGAN_x4plus.pth"
        if not path.exists():
            import requests
            path.parent.mkdir(parents=True, exist_ok=True)
            log.info("downloading Real-ESRGAN x4plus weights (64 MB) from %s", ESRGAN_URL)
            partial = path.with_suffix(".part")
            with requests.get(ESRGAN_URL, stream=True, timeout=60) as response:
                response.raise_for_status()
                with open(partial, "wb") as out:
                    for chunk in response.iter_content(1 << 20):
                        out.write(chunk)
            partial.rename(path)
        return path

    @classmethod
    def load(cls):
        if cls._model is None:
            import torch
            from spandrel import ModelLoader
            device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
            model = ModelLoader().load_from_file(str(cls.weights())).eval()
            cls._model, cls._device = model.to(device), device
            log.info("Real-ESRGAN x4plus loaded on %s", device)
        return cls._model

    def edit(self, image, route):
        import torch
        with self._lock:
            model = self.load()
            tensor = torch.from_numpy(to_array(image)).permute(2, 0, 1).unsqueeze(0)
            try:
                with torch.inference_mode():
                    output = model(tensor.to(self._device))
            except RuntimeError as error:                    # an op missing on MPS: fall back to the CPU
                if self._device == "cpu":
                    raise
                log.warning("Real-ESRGAN on %s failed (%s); falling back to the CPU", self._device, error)
                RealESRGAN._model, RealESRGAN._device = model.to("cpu"), "cpu"
                with torch.inference_mode():
                    output = RealESRGAN._model(tensor)
            array = output.squeeze(0).permute(1, 2, 0).float().cpu().numpy()
            del output, tensor
            # free the allocator cache: on an 8 GB Mac, MPS keeps about 3 GB after each run and the next runs swap
            # (measured: 6 s for the first run, 35-50 s for the next ones; 4 s each with the cache emptied)
            if RealESRGAN._device == "mps":
                torch.mps.empty_cache()
            elif RealESRGAN._device == "cuda":
                torch.cuda.empty_cache()
        return to_image(array)



HANDLERS: dict[str, type[EditingModel]] = {
    **{model_id: GeneralistMock for model_id in GENERALIST_PROFILES},
    **{model_id: RemovalMock for model_id in REMOVAL_PROFILES},
    "sdxl-inpaint": SDXLInpaintMock, "sdxl-outpaint": SDXLInpaintMock,
    "erasedraw": AddMock, "paint-by-inpaint": AddMock,
    "vosr": EnhanceMock, "hvi-cidnet": EnhanceMock, "instructir": EnhanceMock,
    "realesrgan": RealESRGAN,
    "codeformer": RestoreMock, "gfpgan": RestoreMock,
    "ddcolor": DDColorMock,
    "ben2": CutoutMock, "birefnet": CutoutMock,
    "plain-code": PlainCode,
}


def make_model(model_id: str) -> EditingModel:
    handler = HANDLERS[model_id]
    if handler is RealESRGAN and not REAL_MODELS_ENABLED:
        return UpscaleFallback(model_id)
    return handler(model_id)


class UpscaleFallback(EditingModel):
    """Real-ESRGAN replaced by a Lanczos x4 mock when real models are disabled (EJ_WEB_REAL_MODELS=0)."""
    max_side = ESRGAN_MAX_SIDE

    def edit(self, image, route):
        return image.resize((image.width * 4, image.height * 4), Image.LANCZOS).filter(
            ImageFilter.UnsharpMask(radius=2, percent=60, threshold=2))


# ---------------------------------------------------------------- runner

@dataclass
class RunResult:
    model_id: str
    status: str                       # "ok" or "failed"
    backend: str                      # "real" or "mock"
    elapsed_ms: float
    image: Image.Image | None = None
    error: str | None = None


class ModelRunner:
    """Runs one candidate: the real model on the GPU server (`live`, live.py) or a mock here."""

    def __init__(self, live=None):
        self.live = live

    def run_one(self, model_id: str, image: Image.Image, route: Route, estimate: float = 0.0,
                context=None) -> RunResult:
        """`estimate`: seconds the real model takes on the server; a mock waits about that long (MOCK_DELAY).
        `context`: the request as the real models need it (live.EditContext), on the GPU server."""
        started = time.perf_counter()
        if self.live is not None and context is not None:
            try:
                output = self.live.edit(model_id, image.convert("RGB"), context)
                return RunResult(model_id, "ok", "real", (time.perf_counter() - started) * 1000, output)
            except Exception as error:      # one model failing must not stop the others
                log.exception("model %s failed", model_id)
                return RunResult(model_id, "failed", "real", (time.perf_counter() - started) * 1000,
                                 error=f"{type(error).__name__}: {error}"[:300])
        try:
            model = make_model(model_id)
        except KeyError:
            return RunResult(model_id, "failed", "mock", 0.0, error=f"unknown model {model_id!r}")
        try:
            work = limit_size(image.convert("RGB"), model.max_side)
            output = model.edit(work, route)
            if model.backend == "mock" and MOCK_DELAY and estimate:
                seed = int(hashlib.md5((model_id + route.instruction).encode()).hexdigest()[:8], 16)
                wait = estimate * (0.6 + 0.7 * (seed % 1000) / 1000) - (time.perf_counter() - started)
                if wait > 0:
                    time.sleep(wait)
            if output is None or output.width < 1 or output.height < 1:
                raise ValueError("the model returned an empty image")
            if output.mode not in ("RGB", "RGBA"):
                output = output.convert("RGB")
            return RunResult(model_id, "ok", model.backend, (time.perf_counter() - started) * 1000, output)
        except Exception as error:      # one model failing must not stop the others
            log.exception("model %s failed", model_id)
            return RunResult(model_id, "failed", model.backend, (time.perf_counter() - started) * 1000,
                             error=f"{type(error).__name__}: {error}"[:300])
