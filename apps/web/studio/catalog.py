"""The editing models of the live assistant and their benchmark (../docs/editors.md, "Scored run" and "Live demo
selection", 2026-11-26).

- data/rankings.json: every benchmarked model of each task, best mean score first (the scorer mlp-siglip, 80 dev
  samples per task) with its mean seconds per image on a T4.
- Live models: all of them except the slow generalists left out of the demo (FLUX.2 klein 4B, InstructPix2Pix).
  Model IDs are the names of the src/editors registry, so a mock can be replaced by the real wrapper one by one.
- `backend` says how the web app runs a model on this machine: "real" (plain PIL code, Real-ESRGAN through spandrel)
  or "mock" (a Pillow/numpy simulation, until the GPU server runs the real models).
"""

import json
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]      # edit-judge/
WEB_DIR = Path(__file__).resolve().parents[1]        # edit-judge/apps/web/
DATA_DIR = Path(__file__).resolve().parent / "data"

TASKS: dict[str, str] = {"remove": "Remove", "add_replace": "Add / Replace", "enhance": "Enhance", "restore": "Restore",
                         "change_color": "Change color", "formatting": "Formatting", "open_ended": "Open-ended"}
NOT_LIVE = {"flux2-klein-4b", "instruct-pix2pix"}     # too slow for a live edit (../docs/editors.md)


@dataclass(frozen=True)
class ModelInfo:
    id: str
    name: str
    date: str
    license: str
    source: str
    backend: str = "mock"

    def to_dict(self) -> dict:
        return asdict(self)


MODELS: tuple[ModelInfo, ...] = (
    ModelInfo("objectclear", "ObjectClear", "CVPR 2026", "S-Lab (non-commercial)", "hf:jixin0101/ObjectClear"),
    ModelInfo("osor", "OSOR", "ECCV 2026", "MIT code, OpenRAIL++-M weights", "hf:QinmingZhou/OSOR"),
    ModelInfo("rorem", "RORem", "CVPR 2025", "Apache 2.0", "hf:LetsThink/RORem"),
    ModelInfo("lama", "LaMa", "WACV 2022", "Apache 2.0", "gh:advimman/lama"),
    ModelInfo("migan", "MI-GAN", "ICCV 2023", "MIT", "gh:Sanster/models (migan_traced.pt)"),
    ModelInfo("sdxl-inpaint", "SDXL Inpainting", "09/2023", "OpenRAIL++",
              "hf:diffusers/stable-diffusion-xl-1.0-inpainting-0.1"),
    ModelInfo("sdxl-outpaint", "SDXL Outpainting", "09/2023", "OpenRAIL++",
              "hf:diffusers/stable-diffusion-xl-1.0-inpainting-0.1"),
    ModelInfo("erasedraw", "EraseDraw", "ECCV 2024", "MIT", "hf:alpercanberk/erasedraw-v0.2"),
    ModelInfo("paint-by-inpaint", "Paint by Inpaint", "CVPR 2025", "MIT", "hf:paint-by-inpaint/add-finetuned-mb"),
    ModelInfo("realedit", "RealEdit", "CVPR 2025", "CC / MIT", "hf:peter-sushko/RealEdit"),
    ModelInfo("magicbrush", "MagicBrush", "NeurIPS 2023", "CreativeML OpenRAIL-M", "hf:vinesmsuic/magicbrush-jul7"),
    ModelInfo("vosr", "VOSR 2.0", "CVPR 2026", "Apache 2.0", "hf:CSWRY/VOSR"),
    ModelInfo("hvi-cidnet", "HVI-CIDNet", "CVPR 2025", "MIT", "hf:Fediory/HVI-CIDNet-Generalization"),
    ModelInfo("instructir", "InstructIR", "ECCV 2024", "MIT", "hf:marcosv/InstructIR"),
    ModelInfo("realesrgan", "Real-ESRGAN", "ICCV 2021 (workshop)", "BSD-3", "gh:xinntao/Real-ESRGAN", backend="real"),
    ModelInfo("gfpgan", "GFPGAN", "CVPR 2021", "Apache 2.0", "gh:TencentARC/GFPGAN"),
    ModelInfo("codeformer", "CodeFormer", "NeurIPS 2022", "S-Lab (non-commercial)", "gh:sczhou/CodeFormer"),
    ModelInfo("ddcolor", "DDColor", "ICCV 2023", "Apache 2.0", "hf:piddnad/ddcolor_modelscope"),
    ModelInfo("ben2", "BEN2", "01/2025", "MIT", "hf:PramaLLC/BEN2"),
    ModelInfo("birefnet", "BiRefNet", "07/2024", "MIT", "hf:ZhengPeng7/BiRefNet"),
    ModelInfo("plain-code", "Plain code", "-", "-", "Pillow", backend="real"),
)
MODELS_BY_ID: dict[str, ModelInfo] = {model.id: model for model in MODELS}
# display names of the benchmarked models that do not run live
OTHER_NAMES = {"flux2-klein-4b": "FLUX.2 klein 4B", "instruct-pix2pix": "InstructPix2Pix"}


def model_name(model_id: str) -> str:
    info = MODELS_BY_ID.get(model_id)
    return info.name if info else OTHER_NAMES.get(model_id, model_id)


@lru_cache(maxsize=1)
def rankings() -> dict:
    """data/rankings.json: task -> [{model, score, seconds, samples}], best first."""
    return json.loads((DATA_DIR / "rankings.json").read_text())


@lru_cache(maxsize=1)
def dataset_stats() -> dict:
    """data/dataset.json (the figures of the Dataset tab) and the number of benchmarked editing models."""
    data = json.loads((DATA_DIR / "dataset.json").read_text())
    data["stats"]["models"] = len({row["model"] for rows in rankings()["tasks"].values() for row in rows})
    return data
