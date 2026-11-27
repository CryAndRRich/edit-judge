"""Reads a request and picks the models that edit it (../docs/editors.md, "Live demo selection").

understand(instruction) -> the task (one of the 7 benchmark tasks), the operations the request needs, the English
instruction sent to the models and the target to find, from the keyword router (router.py, English and Vietnamese),
so the page shows the task and the models at once. On the GPU server Qwen2.5-VL-7B then rewrites the instruction in
English and plans the region (live.py); on a laptop a Vietnamese request is sent unchanged.

pick_models(task, operations) -> the live models of the task, best mean score first, that apply to the request (a
model limited to one operation, such as EraseDraw to Add, runs only when the request has it): 6 when 6 apply, else
3, else all of them (Open-ended has 2). They run one after another, fastest first.
"""

import itertools
import os
import re
from dataclasses import asdict, dataclass, field
from functools import lru_cache

from .catalog import MODELS_BY_ID, NOT_LIVE, TASKS, model_name, rankings
from .router import ROUTER, strip_diacritics

SPEEDUP = float(os.environ.get("EJ_WEB_SPEEDUP", "0.45"))   # T4 seconds -> this GPU (A30 about 0.45)

# models that take only some requests of their task: the operation they need
SCOPE = {"erasedraw": "add", "paint-by-inpaint": "add", "hvi-cidnet": "lowlight", "ddcolor": "colorize",
         "plain-code": "geometry", "ben2": "cutout", "birefnet": "cutout", "sdxl-outpaint": "extend"}
TASK_SCOPE = {("sdxl-inpaint", "change_color"): "recolor", ("realesrgan", "formatting"): "upscale"}
NO_SCOPE = {("realesrgan", "enhance"), ("sdxl-inpaint", "add_replace")}

GEOMETRY = {"crop", "rotate", "flip", "resize", "square", "bw"}
ENHANCE = {"brighten", "darken", "sharpen", "denoise", "deblur", "upscale", "contrast", "enhance", "warm", "cool"}
MASK_TASKS = {"remove", "add_replace", "change_color"}
TARGET_RE = re.compile(r"\b(?:remove|erase|delete|get rid of|add|put|insert|replace|recolou?r|make|change|turn|paint)\s+"
                       r"(?:the |a |an |all |this |that |his |her |their |my )?([a-z][a-z \-]{1,40}?)"
                       r"(?=\s+(?:with|into|to|in|on|from|at|for|by|and|red|orange|yellow|green|blue|purple|pink|"
                       r"black|white|brown|grey|gray)\b|[.,!?]|$)")


@dataclass
class Understanding:
    instruction: str
    task: str
    operations: list[str]
    english: str
    target: str | None
    vietnamese: bool
    params: dict = field(default_factory=dict)
    intents: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["task_label"] = TASKS[self.task]
        return data


WHOLE_IMAGE_RE = re.compile(r"^(make (it|him|her|them) (look )?like|transform\w*)$")


def task_of(intents: dict[str, list[str]]) -> str:
    """The benchmark task of the router's intents; "make it look like ..." restyles the whole photo: Open-ended."""
    if set(intents) == {"replace"} and all(WHOLE_IMAGE_RE.match(word) for word in intents["replace"]):
        return "open_ended"
    intents = set(intents)
    if intents & {"restore", "face"}:
        return "restore"
    if intents & (GEOMETRY | {"remove_bg", "extend"}):
        return "formatting"
    if "remove_object" in intents:
        return "remove"
    if intents & {"add", "replace", "replace_bg"}:
        return "add_replace"
    if intents & {"colorize", "recolor"}:
        return "change_color"
    if intents & ENHANCE:
        return "enhance"
    return "open_ended"


def operations_of(intents: set[str], task: str) -> list[str]:
    ops = set()
    if "add" in intents:
        ops.add("add")
    if "brighten" in intents:
        ops.add("lowlight")
    if "colorize" in intents:
        ops.add("colorize")
    elif task == "change_color":
        ops.add("recolor")
    if "upscale" in intents:
        ops.add("upscale")
    if intents & GEOMETRY:
        ops.add("geometry")
    if intents & {"remove_bg", "replace_bg"}:
        ops.add("cutout")
    if "extend" in intents:
        ops.add("extend")
    return sorted(ops)


def understand(instruction: str, task: str | None = None) -> Understanding:
    """`task` overrides the task read from the request (the user changed it in the UI)."""
    text = " ".join(instruction.split())
    route = ROUTER.route(text)
    intents = set(route.intents)
    chosen = task if task in TASKS else task_of(route.intents)
    vietnamese = strip_diacritics(text) != text or bool(re.search(r"\b(xoa|lam|anh|mau|thanh|cho|nen|them)\b",
                                                                    text.lower()))
    english = text if vietnamese else (text[:1].upper() + text[1:]).rstrip(".!?") + "."
    match = TARGET_RE.search(english.lower()) if chosen in MASK_TASKS and not vietnamese else None
    target = match.group(1).strip() if match else None
    if target and target.split()[0] in ("it", "him", "her", "them", "this", "that", "everything"):
        target = None                                   # a pronoun: the whole photo, nothing to find
    return Understanding(instruction=text, task=chosen, operations=operations_of(intents, chosen), english=english,
                         target=target, vietnamese=vietnamese,
                         params=route.params, intents=route.intents)


def scope_of(model: str, task: str) -> str | None:
    if (model, task) in NO_SCOPE:
        return None
    return TASK_SCOPE.get((model, task)) or SCOPE.get(model)


def live_rows(task: str) -> list[dict]:
    return [row for row in rankings()["tasks"][task] if row["model"] not in NOT_LIVE and row["model"] in MODELS_BY_ID]


def count_for(available: int) -> int:
    return 6 if available >= 6 else min(3, available)


def pick_models(task: str, operations: list[str] | set[str]) -> list[dict]:
    """[{model, name, score, seconds_t4, estimate}] in benchmark order; the caller runs them fastest first."""
    ops = set(operations)
    fits = [row for row in live_rows(task) if (scope_of(row["model"], task) or "") in ops | {""}]
    return [{"model": row["model"], "name": model_name(row["model"]), "score": row["score"],
             "seconds_t4": row["seconds"], "estimate": round(max(0.25, row["seconds"] * SPEEDUP), 2)}
            for row in fits[:count_for(len(fits))]]


@lru_cache(maxsize=None)
def live_models(task: str) -> frozenset[str]:
    """Models some request of the task can reach: every combination of the task's special operations."""
    scopes = sorted({scope_of(row["model"], task) for row in live_rows(task)} - {None})
    live = set()
    for size in range(len(scopes) + 1):
        for ops in itertools.combinations(scopes, size):
            live.update(row["model"] for row in pick_models(task, ops))
    return frozenset(live)


def downloadable_models() -> list[str]:
    """Every editor the assistant can run (the GPU server downloads only these: apps/web/download.py)."""
    return sorted(set().union(*(live_models(task) for task in TASKS)))


def rankings_view(measured: dict[str, float] | None = None) -> dict:
    """The Rankings tab: every benchmarked model per task, with its place, whether the assistant runs it, its mean
    seconds per image on the benchmark GPU (`seconds`) and on the live GPU (`estimate`: `measured` there, the GPU
    server's own wait for one edit now, its load included, else the benchmark's times SPEEDUP)."""
    measured = measured or {}
    tasks = []
    for task, label in TASKS.items():
        live = live_models(task)
        rows = [{"place": index + 1, "model": row["model"], "name": model_name(row["model"]), "score": row["score"],
                 "seconds": row["seconds"], "measured": row["model"] in measured,
                 "estimate": measured.get(row["model"]) or round(row["seconds"] * SPEEDUP, 2),
                 "live": row["model"] in live}
                for index, row in enumerate(rankings()["tasks"][task])]
        tasks.append({"task": task, "label": label, "rows": rows})
    return {"tasks": tasks, "samples_per_task": rankings()["samples_per_task"], "gpu": rankings()["gpu"],
            "speedup": SPEEDUP}
