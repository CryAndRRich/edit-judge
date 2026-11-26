"""Editing models and their benchmark (PROJECT.md Part II Step 7, MODEL_LIST.md, all 7 lists).

Each task has a small fixed set of dev samples; every model of its list runs on them, and the generalists of
list 1 run on every task. A run has these steps, each resumable and each writing to work/editors/:

1. samples (samples.py): the dev samples of each task, chosen by their request categories, each with an op
   (remove, add, replace, background; light, enhance; restore; colorize, recolor; cutout, format; open);
2. plan (plan.py; remove, add_replace, change_color, formatting): Qwen2.5-VL-7B reads each request and names what
   to find, with its own box, and what to paint (plan_v2, v3), or the colour change and the format action (plan_v4);
3. masks (grounding.py; ops remove, add, replace, background, recolor): Grounding DINO's boxes inside Qwen's box
   (Qwen's box when DINO finds none), SAM 2.1 masks; every mask is saved, so a bad mask can be told apart from a
   bad edit;
4. run (base.py): every model of the task on every sample, each model in its own process (and its own Python
   environment when it needs one); mask models keep only the masked region (pasted back on the original);
   formatting edits that change the size keep it;
5. score.py, report.py: the scorer judges the outputs next to the human edits; a report per task and a summary of
   every task and model.

Register each editor as name -> "module:Class" (module relative to this package); modules are imported only when an
editor is used, so torch loads only then. To add one, subclass base.Editor in its own file in models/ and register
it here.

Dropped on 2026-11-24 for latency (the assistant edits live; MODEL_LIST.md "Left out"): InstantIR, OmniGen2, VIBE,
FLUX 3 Image Klein; VAREdit 2B on 2026-11-26.
"""

import importlib

TASKS = ("remove", "add_replace", "enhance", "restore", "change_color", "formatting", "open_ended")

EDITORS: dict[str, str] = {
    "lama": ".models.lama:Lama",                        # list 2, classic (WACV 2022)
    "migan": ".models.migan:MIGAN",                     # list 2, light GAN (ICCV 2023)
    "rorem": ".models.rorem:RORem",                     # list 2 (CVPR 2025)
    "objectclear": ".models.objectclear:ObjectClear",   # list 2 (CVPR 2026)
    "osor": ".models.osor:OSOR",                        # list 2, one step (ECCV 2026)
    "sdxl-inpaint": ".models.sdxl_inpaint:SDXLInpaint",  # list 3 (and list 6, 7 later)
    "paint-by-inpaint": ".models.instruct_add:PaintByInpaint",   # list 3, mask-free addition (CVPR 2025)
    "erasedraw": ".models.instruct_add:EraseDraw",      # list 3, mask-free addition (ECCV 2024)
    # list 4 Enhance (ported from Steven's run of 2026-11-22)
    "realesrgan": ".models.realesrgan:RealESRGAN",      # lists 4 and 7, classic (ICCV 2021 workshop)
    "hvi-cidnet": ".models.hvi_cidnet:HVICIDNet",       # list 4, low light only (CVPR 2025)
    "instructir": ".models.instructir:InstructIR",      # list 4 (ECCV 2024)
    "vosr": ".models.vosr:VOSR",                        # lists 4 and 5 (CVPR 2026)
    # list 5 Restore
    "codeformer": ".models.codeformer:CodeFormer",      # list 5, classic (NeurIPS 2022)
    "gfpgan": ".models.gfpgan:GFPGAN",                  # list 5, classic (CVPR 2021)
    # list 6 Change-Color (and sdxl-inpaint above, op recolor)
    "ddcolor": ".models.ddcolor:DDColor",               # list 6, colourise (ICCV 2023)
    # list 7 Image-Formatting (and realesrgan above, resize up)
    "ben2": ".models.ben2:BEN2",                        # list 7, cut-out (01/2025)
    "birefnet": ".models.birefnet:BiRefNet",            # list 7, cut-out (07/2024)
    "sdxl-outpaint": ".models.sdxl_inpaint:SDXLOutpaint",   # list 7, extend
    "plain-code": ".models.plain_code:PlainCode",       # list 7, crop / rotate / flip / resize / grayscale (PIL)
    # list 1 Shared: every task
    "instruct-pix2pix": ".models.instruct_add:InstructPix2Pix",  # classic (CVPR 2023)
    "magicbrush": ".models.instruct_add:MagicBrush",    # classic (NeurIPS 2023)
    "realedit": ".models.instruct_add:RealEdit",        # classic (CVPR 2025)
    "flux2-klein-4b": ".models.flux2_klein:Flux2Klein4B",   # 01/2026
}


def get_editor(name: str):
    target = EDITORS.get(name)
    if target is None:
        raise SystemExit(f"no editor {name!r}; registered: {sorted(EDITORS)}")
    module, class_name = target.split(":")
    return getattr(importlib.import_module(module, __name__), class_name)()


def editors_of(task: str) -> list[str]:
    """Registered editors that run on `task`, in registry order."""
    return [name for name in EDITORS if task in get_editor(name).tasks]
