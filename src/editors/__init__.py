"""Editing models and their benchmark (PROJECT.md Part II Step 7, MODEL_LIST.md).

This package covers the inpainting lists of MODEL_LIST.md so far: list 2 (Remove) and list 3 (Add/Replace). Every
model there edits (image, mask), so a run has four steps, each resumable and each writing to work/editors/:

1. samples (samples.py): a small fixed set of dev samples per task, chosen by their request categories;
2. plan (plan.py): Qwen2.5-VL-7B reads each request and names what to find, with its own box (remove, replace;
   background: the subjects to keep), where to put the new thing (add) and what to paint;
3. masks (grounding.py): Grounding DINO's boxes inside Qwen's box (Qwen's box when DINO finds none), SAM 2.1
   masks (add: the plan's box; background: inverted); every mask is saved, so a bad mask can be told apart from a
   bad edit;
4. run (base.py): every model of the task on every sample, each model in its own process (and its own Python
   environment when it needs one); only the masked region of an output is kept (pasted back on the original);
   then score.py and report.py judge the outputs with the scorer.

Register each editor as name -> "module:Class" (module relative to this package); modules are imported only when an
editor is used, so torch loads only then. To add one, subclass base.Editor in its own file in models/ and register
it here.
"""

import importlib

TASKS = ("remove", "add_replace")

EDITORS: dict[str, str] = {
    "lama": ".models.lama:Lama",                        # list 2, classic (WACV 2022)
    "rorem": ".models.rorem:RORem",                     # list 2 (CVPR 2025)
    "objectclear": ".models.objectclear:ObjectClear",   # list 2 (CVPR 2026)
    "sdxl-inpaint": ".models.sdxl_inpaint:SDXLInpaint",  # list 3 (and list 6, 7 later)
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
