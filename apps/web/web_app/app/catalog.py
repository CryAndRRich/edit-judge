"""The editing models the web app knows: the 22 models of MODEL_LIST.md (7 lists) and the plain-code (PIL) option of
list 7. Grounding DINO and SAM 2.1 only make masks, so they are not candidates.

`backend` says how the web app runs a model on this machine:
- "real": the model itself runs locally (Real-ESRGAN x4plus through spandrel, plain PIL code);
- "mock": a Pillow/numpy simulation of what the model does, until its weights are wired in (most need a T4-class
  GPU, see MODEL_LIST.md). The API and the UI label every mock output as such.

Enhance models with Kaggle T4 benchmark results (reports/enhance/stats.json) use the same IDs as that file.
"""

import json
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]      # edit-judge/
WEB_DIR = Path(__file__).resolve().parents[2]        # edit-judge/apps/web/
# Steven's Enhance benchmark report: removed from the repository on 2026-11-24 (its results are in MODEL_LIST.md and
# the editors' runs replace it, work/editors/summary.md); without the file no benchmark numbers are shown
ENHANCE_STATS = REPO_ROOT / "reports" / "enhance" / "stats.json"

CATEGORIES: dict[str, dict] = {
    "shared": {"label": "Shared", "list": 1, "description": "Generalists that follow any instruction"},
    "remove": {"label": "Remove", "list": 2, "description": "Remove objects, people, text, logos"},
    "add_replace": {"label": "Add / Replace", "list": 3, "description": "Add or replace objects, clothes, backgrounds"},
    "enhance": {"label": "Enhance", "list": 4, "description": "Quality, sharpness, lighting, colour tone"},
    "restore": {"label": "Restore", "list": 5, "description": "Old, damaged, blurred or faded photos"},
    "color": {"label": "Change Color", "list": 6, "description": "Recolour a region, colourise black and white"},
    "formatting": {"label": "Image Formatting", "list": 7, "description": "Cut-out, extend, resize, crop, rotate"},
}


@dataclass(frozen=True)
class ModelInfo:
    id: str
    name: str
    date: str
    categories: tuple[str, ...]
    weights_gb: float | None          # fp16 weights from MODEL_LIST.md: the VRAM estimate (activations not counted)
    license: str
    source: str
    description: str
    classic: bool = False
    backend: str = "mock"
    tags: tuple[str, ...] = field(default=())

    def to_dict(self) -> dict:
        data = asdict(self)
        data["benchmark"] = enhance_benchmark().get(self.id)
        return data


MODELS: tuple[ModelInfo, ...] = (
    # 1. Shared (generalists)
    ModelInfo("flux3_klein", "FLUX 3 Image Klein", "10/2026", ("shared",), 20.3, "FLUX Kommunity (gated)",
              "hf:black-forest-labs/flux-3-image-klein", "Newest generalist; split over 2 T4s"),
    ModelInfo("varedit_2b", "VAREdit 2B", "ICLR 2026", ("shared",), 11.7, "MIT", "hf:HiDream-ai/VAREdit",
              "Visual autoregressive editor, 512 x 512 only, fast"),
    ModelInfo("flux2_klein_4b", "FLUX.2 [klein] 4B", "01/2026", ("shared",), 16.0, "Apache 2.0",
              "hf:black-forest-labs/FLUX.2-klein-4B", "4-step distilled FLUX.2 editor"),
    ModelInfo("vibe", "VIBE", "12/2025", ("shared",), 9.3, "Apache 2.0 / SANA", "hf:iitolstykh/VIBE-Image-Edit",
              "Sana1.5 1.6B with a Qwen3-VL-2B encoder"),
    ModelInfo("omnigen2", "OmniGen2", "06/2025", ("shared",), 15.1, "Apache 2.0", "hf:OmniGen2/OmniGen2",
              "Unified generation and editing, Qwen2.5-VL conditioning"),
    ModelInfo("realedit", "RealEdit", "CVPR 2025", ("shared",), 2.2, "CC / MIT", "hf:peter-sushko/RealEdit",
              "InstructPix2Pix fine-tuned on real user edit requests", classic=True),
    ModelInfo("magicbrush", "MagicBrush", "NeurIPS 2023", ("shared",), 2.2, "CreativeML OpenRAIL-M",
              "hf:vinesmsuic/magicbrush-jul7", "InstructPix2Pix fine-tuned on manual edits", classic=True),
    ModelInfo("instructpix2pix", "InstructPix2Pix", "CVPR 2023", ("shared",), 2.7, "MIT",
              "hf:timbrooks/instruct-pix2pix", "The original instruction-following diffusion editor", classic=True),
    # 2. Remove
    ModelInfo("objectclear", "ObjectClear", "CVPR 2026", ("remove",), 7.8, "S-Lab (non-commercial)",
              "hf:jixin0101/ObjectClear", "Removes objects with their shadows and reflections"),
    ModelInfo("rorem", "RORem", "CVPR 2025", ("remove",), 6.9, "Apache 2.0", "hf:LetsThink/RORem",
              "SDXL inpainting trained for reliable object removal"),
    ModelInfo("lama", "LaMa (big-lama)", "WACV 2022", ("remove",), 0.2, "Apache 2.0",
              "gh:enesmsahin/simple-lama-inpainting", "Fast Fourier-convolution inpainting", classic=True),
    # 3. Add / Replace, 6. Color (one region), 7. Formatting (outpainting)
    ModelInfo("sdxl_inpaint", "SDXL Inpainting 0.1", "09/2023", ("add_replace", "color", "formatting"), 6.9,
              "OpenRAIL++", "hf:diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
              "Masked inpainting: add, replace, recolour one region, outpaint"),
    # 4. Enhance (and 5. Restore)
    ModelInfo("vosr", "VOSR 2.0", "CVPR 2026", ("enhance", "restore"), 7.0, "Apache 2.0", "hf:CSWRY/VOSR",
              "One-step generative super-resolution and sharpening"),
    ModelInfo("hvi_cidnet", "HVI-CIDNet", "CVPR 2025", ("enhance",), 0.008, "MIT",
              "hf:Fediory/HVI-CIDNet-Generalization", "Low-light enhancement (run on lighting requests only)",
              tags=("lighting",)),
    ModelInfo("instructir", "InstructIR", "ECCV 2024", ("enhance",), 0.06, "MIT", "hf:marcosv/InstructIR",
              "Instruction-driven restoration: noise, blur, haze, low light"),
    ModelInfo("realesrgan", "Real-ESRGAN x4plus", "ICCV 2021 (workshop)", ("enhance", "formatting"), 0.07, "BSD-3",
              "gh:xinntao/Real-ESRGAN", "x4 super-resolution (runs locally through spandrel)", classic=True,
              backend="real"),
    # 5. Restore
    ModelInfo("instantir", "InstantIR", "10/2024", ("restore",), 11.5, "Apache 2.0", "hf:InstantX/InstantIR",
              "SDXL-based blind image restoration"),
    ModelInfo("codeformer", "CodeFormer", "NeurIPS 2022", ("restore",), 0.4, "S-Lab (non-commercial)",
              "gh:sczhou/CodeFormer", "Codebook face restoration", classic=True, tags=("face",)),
    ModelInfo("gfpgan", "GFPGAN v1.4", "CVPR 2021", ("restore",), 0.35, "Apache 2.0", "gh:TencentARC/GFPGAN",
              "GAN-prior face restoration with background upscaling", classic=True, tags=("face",)),
    # 6. Change color
    ModelInfo("ddcolor", "DDColor", "ICCV 2023", ("color",), 0.9, "Apache 2.0", "hf:piddnad/ddcolor_modelscope",
              "Dual-decoder colourisation of the whole image"),
    # 7. Image formatting
    ModelInfo("ben2", "BEN2", "01/2025", ("formatting",), 0.38, "MIT", "hf:PramaLLC/BEN2",
              "Background erase network: foreground cut-out"),
    ModelInfo("birefnet", "BiRefNet", "07/2024", ("formatting",), 0.44, "MIT", "hf:ZhengPeng7/BiRefNet",
              "Bilateral-reference dichotomous segmentation cut-out"),
    ModelInfo("pil_plain", "Plain code (PIL)", "-", ("formatting",), 0.0, "-", "Pillow",
              "Crop, resize, rotate, flip, grayscale: sometimes the right answer", classic=True, backend="real"),
)

MODELS_BY_ID: dict[str, ModelInfo] = {model.id: model for model in MODELS}


def models_in(category: str) -> list[ModelInfo]:
    return [model for model in MODELS if category in model.categories]


@lru_cache(maxsize=1)
def enhance_benchmark() -> dict:
    """Kaggle 2 x T4 results of the Enhance smoke benchmark (30 samples), by model ID; {} without the report."""
    try:
        stats = json.loads(ENHANCE_STATS.read_text())
    except (OSError, ValueError):
        return {}
    keep = {}
    for model_id, row in stats.items():
        keep[model_id] = {"samples": row.get("samples"), "ok": row.get("ok"),
                          "seconds_per_image_t4": row.get("seconds_mean"), "peak_gpu_gb": row.get("peak_gpu_gb"),
                          "editreward_gain": (row.get("editreward") or {}).get("gain"),
                          "editreward_win_rate": (row.get("editreward") or {}).get("win_rate"),
                          "imagereward_gain": (row.get("imagereward") or {}).get("gain")}
    return keep
