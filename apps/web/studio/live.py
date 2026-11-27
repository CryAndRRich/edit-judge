"""The real models of the GPU server: Qwen2.5-VL-7B reads the request, Grounding DINO and SAM 2.1 find the region,
the src/editors wrappers edit, all on one CUDA GPU in this process. On a laptop (no CUDA) none of this loads and
the mocks of model_runner.py run instead.

EJ_WEB_BACKEND = auto (default: real when CUDA is available) | real | mock.

One request (`LiveModels.prepare`, then `LiveModels.edit` per model):
1. Qwen2.5-VL-7B rewrites the request (English or Vietnamese) as one short English instruction (prompt
   src/editors/prompts/request_v1.txt). The task and the models were already chosen from the request by the
   keyword router (assistant.py), so the page can show them at once; if Qwen's answer does not parse, the
   request itself is sent.
2. The ops that need a plan (PLANNED_OPS: remove, add, replace, background, recolor, format): Qwen writes the edit
   plan of the benchmark (plan.py, `plan_one`, the task's prompt version) and, for the ops with a region (all but
   format), Grounding DINO and SAM 2.1 turn it into a mask (grounding.py, `final_mask`), exactly as on Kaggle.
3. Each editor edits with `edit_one` (base.py): the same resize, black check and paste-back as the benchmark.

GPU memory. Qwen (about 16 GB in bfloat16), DINO and SAM (about 2 GB) and the scorer's SigLIP stay loaded. The
editors are kept in an LRU cache: their own memory (measured with torch.cuda.memory_allocated around load) stays
under EJ_WEB_EDITORS_GB (default 24), and before an editor loads, the least recently used ones are dropped until the
GPU has room for it (torch.cuda.mem_get_info, which also sees the other users of a shared GPU). A dropped editor
loads again from disk when it is needed.
"""

import gc
import json
import logging
import os
import re
import sys
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from string import Template

from .catalog import REPO_ROOT

log = logging.getLogger("studio.live")

BACKEND = os.environ.get("EJ_WEB_BACKEND", "auto").lower()
EDITORS_GB = float(os.environ.get("EJ_WEB_EDITORS_GB", "24"))   # GPU memory the cached editors may hold
FREE_MARGIN_GB = 2.0                     # kept free on the GPU besides the editor being loaded (activations)
RAM_GB = float(os.environ.get("EJ_WEB_RAM_GB", "40"))   # RAM the editors moved off the GPU may hold
DEFAULT_EDITOR_GB = 6.0                  # first guess of an editor's size before it was measured
EDIT_WEIGHT = 0.3                        # weight of the newest edit in the running mean of seconds per edit
STATS_FILE = REPO_ROOT / "work" / "web" / "editors.json"     # what the server measured (EditorStats)
RESULTS_DIR = Path(os.environ.get("EJ_WEB_RESULTS") or REPO_ROOT / "work" / "web" / "results")   # ResultCache
CACHE_RESULTS = os.environ.get("EJ_WEB_CACHE_RESULTS", "1") != "0"
REQUEST_PROMPT = "request_v1"
REQUEST_TOKENS = 120
# the ops whose editors need a plan (a region, or the format action): colorize and cutout need none
PLANNED_OPS = {"remove": "remove", "add": "add_replace", "replace": "add_replace", "background": "add_replace",
               "recolor": "change_color", "format": "formatting"}


def add_repo_path() -> None:
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))


def cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def use_real() -> bool:
    """Whether this process runs the real models (EJ_WEB_BACKEND, else CUDA)."""
    if BACKEND == "mock":
        return False
    if BACKEND == "real":
        return True
    return cuda_available()


def prefer_ipv4() -> None:
    """Connections try the IPv4 addresses of a host first. Server A has IPv6 addresses without a working IPv6
    route (2026-11-28: curl -6 to huggingface.co times out, -4 answers in 0.3 s); Python tries the addresses in
    the order the resolver gives, IPv6 first, and waits on each. Without sudo the system preference
    (/etc/gai.conf) cannot change, so the order is changed here for this process."""
    import socket
    if getattr(socket.getaddrinfo, "prefers_ipv4", False):
        return
    resolve = socket.getaddrinfo

    def ipv4_first(*args, **kwargs):
        return sorted(resolve(*args, **kwargs), key=lambda address: address[0] != socket.AF_INET)

    ipv4_first.prefers_ipv4 = True
    socket.getaddrinfo = ipv4_first


QUIET_WARNINGS = (               # messages of the libraries that say nothing about our run (log of 2026-11-28)
    "torch.meshgrid", "xFormers is not available", "The parameter 'pretrained' is deprecated",
    "Arguments other than a weight enum", "Defaulting to unsafe serialization",
)


def quiet_libraries() -> None:
    """Keeps the server's log to what matters: no deprecation notices of timm, torch.jit, PEFT, torchvision
    (FutureWarning), no info-level chatter of transformers and diffusers (a .bin VAE found instead of safetensors,
    fp16 pipelines moved to RAM, unused config keys), no download or loading progress bars. Errors still show."""
    import warnings
    warnings.filterwarnings("ignore", category=FutureWarning)
    for message in QUIET_WARNINGS:
        warnings.filterwarnings("ignore", message=f".*{message}")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("DIFFUSERS_VERBOSITY", "error")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    logging.getLogger("accelerate").setLevel(logging.WARNING)    # "We will use 90% of the memory on device 0"
    try:
        from diffusers.utils import logging as diffusers_logging
        from transformers.utils import logging as transformers_logging
        diffusers_logging.set_verbosity_error()
        diffusers_logging.disable_progress_bar()          # "Loading pipeline components"
        transformers_logging.set_verbosity_error()
        transformers_logging.disable_progress_bar()       # "Loading checkpoint shards"
    except ImportError:
        pass


def server_defaults() -> None:
    """On the GPU server: IPv4 first (prefer_ipv4), Hugging Face downloads over plain HTTPS (the Xet client has
    its own network code, which the IPv4 order does not reach), and the Hugging Face and torch caches in the
    repository's git-ignored work/ (with everything else of the project: models, outputs) unless HF_HOME /
    TORCH_HOME say otherwise; then a quiet log (quiet_libraries). Call before any Hugging Face import: the
    Hugging Face libraries read these variables when they are imported."""
    if use_real():
        prefer_ipv4()
        os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
        cache = REPO_ROOT / "work" / "hf_cache"
        os.environ.setdefault("HF_HOME", str(cache))
        os.environ.setdefault("TORCH_HOME", str(cache / "torch"))
        quiet_libraries()                     # last: it imports diffusers


def free_gb() -> float:
    import torch
    free, _ = torch.cuda.mem_get_info()
    return free / 1e9


def sample_op(task: str, operations) -> str:
    """The op of the benchmark samples (samples.py) that matches the request's task and operations."""
    ops = set(operations)
    if task == "remove":
        return "remove"
    if task == "add_replace":
        return "add" if "add" in ops else "background" if "cutout" in ops else "replace"
    if task == "change_color":
        return "colorize" if "colorize" in ops else "recolor"
    if task == "formatting":
        return "cutout" if "cutout" in ops else "format"
    if task == "enhance":
        return "light" if "lowlight" in ops else "enhance"
    if task == "restore":
        return "restore"
    return "open"


def parse_instruction(answer: str) -> str | None:
    """The English instruction of Qwen's answer, or None."""
    match = re.search(r"\{.*\}", answer, re.S)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    text = " ".join(str(data.get("instruction") or "").split()) if isinstance(data, dict) else ""
    return text or None


@dataclass
class EditContext:
    """What every editor of one request gets."""
    instruction: str                      # the English instruction (Qwen's, or the request)
    op: str                               # the benchmark op (sample_op)
    plan: dict | None = None              # plan.py's plan, for the planned tasks
    mask: object = None                   # PIL L (255 = repaint) or None
    boxes: list = field(default_factory=list)
    seconds: dict = field(default_factory=dict)   # time spent reading, planning, masking

    def target(self) -> str | None:
        """What the plan names, for the page ("the man on the left", "the sky")."""
        if not self.plan:
            return None
        things = self.plan.get("targets") or ([self.plan["anchor"]] if self.plan.get("anchor") else [])
        if self.op == "background":
            things = self.plan.get("keep") or self.plan.get("targets") or []
        return ", ".join(things) or None


class ResultCache:
    """Requests already answered, kept on disk: the same photo with the same request is read, planned and edited
    once; again, no model runs and pipeline.Jobs replays each step as long as it took the first time, then shows
    the saved output and score (this survives restarts). The key is the photo's
    bytes at the working size, the request and the task; an output is reused only for the same editor settings
    (`settings_version`) and the same scorer. EJ_WEB_CACHE_RESULTS=0 turns it off."""

    def __init__(self, folder: Path | None = None):
        self.folder = folder or RESULTS_DIR

    @staticmethod
    def key(image_path: Path, request: str, task: str) -> str:
        import hashlib
        digest = hashlib.sha1(image_path.read_bytes())
        digest.update(f"|{' '.join(request.lower().split())}|{task}".encode())
        return digest.hexdigest()[:24]

    def load_context(self, key: str) -> EditContext | None:
        from PIL import Image
        folder = self.folder / key
        try:
            data = json.loads((folder / "context.json").read_text())
            mask = None
            if data.pop("has_mask"):
                with Image.open(folder / "mask.png") as image:
                    mask = image.convert("L")
            return EditContext(**data, mask=mask)
        except (OSError, ValueError, TypeError, KeyError):
            return None

    def save_context(self, key: str, context: EditContext) -> None:
        folder = self.folder / key
        folder.mkdir(parents=True, exist_ok=True)
        if context.mask is not None:
            context.mask.save(folder / "mask.png")
        data = {"instruction": context.instruction, "op": context.op, "plan": context.plan, "boxes": context.boxes,
                "seconds": context.seconds, "has_mask": context.mask is not None}
        (folder / "context.json").write_text(json.dumps(data, ensure_ascii=False))

    def load_output(self, key: str, model: str, tag: str):
        """(image, {"seconds", "score"}) or None."""
        from PIL import Image
        folder = self.folder / key
        try:
            meta = json.loads((folder / f"{model}.{tag}.json").read_text())
            with Image.open(folder / f"{model}.{tag}.png") as image:
                image.load()
                return image.copy(), meta
        except (OSError, ValueError):
            return None

    def save_output(self, key: str, model: str, tag: str, image, meta: dict) -> None:
        folder = self.folder / key
        folder.mkdir(parents=True, exist_ok=True)
        image.save(folder / f"{model}.{tag}.png")
        (folder / f"{model}.{tag}.json").write_text(json.dumps(meta))      # last: it marks the output complete


def output_tag(model: str, scorer: str) -> str:
    """What a cached output depends on besides the request: the editor's settings and the scorer."""
    from src.editors import get_editor
    return f"v{get_editor(model).settings_version}-{scorer}"


class EditorStats:
    """What the server measured per editor, kept in STATS_FILE across restarts: GPU memory, seconds per edit (a
    running mean), seconds to load from disk and from RAM, and whether moving it to RAM and back gives the same
    edit (`offload`, checked by run.py --check; an editor not checked is never moved to RAM, only dropped)."""

    def __init__(self, path: Path | None = None):
        self.path = path or STATS_FILE
        try:
            self.rows = json.loads(self.path.read_text())
        except (OSError, ValueError):
            self.rows = {}
        self.lock = threading.Lock()

    def get(self, name: str, key: str, default=None):
        return self.rows.get(name, {}).get(key, default)

    def record(self, name: str, **values) -> None:
        with self.lock:
            row = self.rows.setdefault(name, {})
            for key, value in values.items():
                if key == "edit_s" and row.get("edit_s") is not None:       # a running mean of recent edits
                    value = round(row["edit_s"] * (1 - EDIT_WEIGHT) + value * EDIT_WEIGHT, 2)
                row[key] = round(value, 2) if isinstance(value, float) else value
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                self.path.write_text(json.dumps(self.rows, indent=1, sort_keys=True))
            except OSError:
                log.warning("could not write %s", self.path)


STAYS_ON_CPU = "_stays_on_cpu"           # editor attribute: the parts move_to found on the CPU already


def move_to(editor, device: str) -> None:
    """Moves every torch module, tensor and diffusers pipeline an editor holds (its attributes, inside plain
    lists, tuples and dicts, and the attributes of its other objects two levels down: GFPGAN's face helper, ...)
    to `device`, keeping their dtype. Moving to the CPU records the parts that were on the CPU already (InstructIR's
    text encoder runs there, as upstream); moving back leaves them there. Containers are rebuilt only when
    something inside moved, and only plain ones (a diffusers FrozenDict config stays as it is)."""
    import itertools
    import types

    import torch

    to_cpu = device == "cpu"
    stays = set() if to_cpu else set(getattr(editor, STAYS_ON_CPU, ()))
    seen = set()

    def current_device(value) -> str | None:
        if isinstance(value, torch.Tensor):
            return value.device.type
        if isinstance(value, torch.nn.Module):
            first = next(itertools.chain(value.parameters(), value.buffers()), None)
            return None if first is None else first.device.type
        return getattr(getattr(value, "device", None), "type", None)

    def visit(value, path: str, depth: int):
        if id(value) in seen:
            return value
        seen.add(id(value))
        movable = isinstance(value, (torch.nn.Module, torch.Tensor)) or (
            hasattr(value, "components") and hasattr(value, "to"))          # a diffusers pipeline
        if movable:
            if to_cpu and current_device(value) == "cpu":
                stays.add(path)
                return value
            if path in stays:
                return value
            return value.to(device)
        if type(value) in (list, tuple):
            items = [visit(item, f"{path}[{index}]", depth) for index, item in enumerate(value)]
            changed = any(new is not old for new, old in zip(items, value))
            return type(value)(items) if changed else value
        if type(value) is dict:
            items = {key: visit(item, f"{path}[{key!r}]", depth) for key, item in value.items()}
            changed = any(items[key] is not item for key, item in value.items())
            return items if changed else value
        if depth < 2 and hasattr(value, "__dict__") and not isinstance(value, (type, types.ModuleType)):
            for key, item in list(vars(value).items()):
                moved = visit(item, f"{path}.{key}", depth + 1)
                if moved is not item:
                    setattr(value, key, moved)
        return value

    for key, item in list(vars(editor).items()):
        if key == STAYS_ON_CPU:
            continue
        moved = visit(item, key, 1)
        if moved is not item:
            setattr(editor, key, moved)
    if to_cpu:
        setattr(editor, STAYS_ON_CPU, stays)


class GpuEditors:
    """The editors on the GPU (least recently used first, within `budget_gb`), and the ones moved to RAM (within
    `ram_gb`) when they leave the GPU: back on the GPU in a few seconds instead of a load from disk."""

    def __init__(self, budget_gb: float = EDITORS_GB, ram_gb: float = RAM_GB, stats: EditorStats | None = None):
        self.budget_gb, self.ram_gb = budget_gb, ram_gb
        self.on_gpu: OrderedDict[str, tuple[object, float]] = OrderedDict()
        self.in_ram: OrderedDict[str, tuple[object, float]] = OrderedDict()
        self.stats = stats or EditorStats()

    def used_gb(self) -> float:
        return sum(size for _, size in self.on_gpu.values())

    def ram_used_gb(self) -> float:
        return sum(size for _, size in self.in_ram.values())

    def where(self, name: str) -> str:
        return "gpu" if name in self.on_gpu else "ram" if name in self.in_ram else "disk"

    def evict(self, name: str) -> None:
        """Takes `name` off the GPU: into RAM when it was checked and RAM has room, else dropped."""
        import torch
        editor, size = self.on_gpu.pop(name)
        if self.stats.get(name, "offload") and size <= self.ram_gb:
            while self.in_ram and self.ram_used_gb() + size > self.ram_gb:
                self.in_ram.popitem(last=False)
            move_to(editor, "cpu")
            self.in_ram[name] = (editor, size)
            where = "moved to RAM"
        else:
            where = "dropped"
        del editor
        gc.collect()
        torch.cuda.empty_cache()
        log.info("%s %s (%.1f GB)", name, where, size)

    def make_room(self, need_gb: float) -> None:
        while self.on_gpu and (self.used_gb() + need_gb > self.budget_gb or free_gb() < need_gb + FREE_MARGIN_GB):
            self.evict(next(iter(self.on_gpu)))

    def get(self, name: str):
        """The editor `name`, on the GPU."""
        import torch

        from src.editors import get_editor

        if name in self.on_gpu:
            self.on_gpu.move_to_end(name)
            return self.on_gpu[name][0]
        need = self.stats.get(name, "gb") or max(DEFAULT_EDITOR_GB, get_editor(name).download_gb * 1.2)
        self.make_room(need)
        started, before = time.monotonic(), torch.cuda.memory_allocated()
        if name in self.in_ram:
            editor, size = self.in_ram.pop(name)
            move_to(editor, "cuda")
            self.stats.record(name, ram_load_s=time.monotonic() - started)
        else:
            editor = get_editor(name)
            editor.load("cuda")
            editor.loaded_on = "cuda"
            size = max(0.0, (torch.cuda.memory_allocated() - before) / 1e9)
            self.stats.record(name, gb=size, disk_load_s=time.monotonic() - started)
        self.on_gpu[name] = (editor, size)
        log.info("%s on the GPU in %.1f s (%.1f GB; editors hold %.1f GB, %.1f GB free)", name,
                 time.monotonic() - started, size, self.used_gb(), free_gb())
        return editor

    def estimate(self, name: str) -> float | None:
        """Seconds the next run of `name` takes here (its load included), or None before it was measured."""
        edit = self.stats.get(name, "edit_s")
        if edit is None:
            return None
        load = {"gpu": 0.0, "ram": self.stats.get(name, "ram_load_s", 0.0),
                "disk": self.stats.get(name, "disk_load_s", 0.0)}[self.where(name)]
        return round(edit + (load or 0.0), 2)

    def drop_all(self, keep: str | None = None) -> None:
        for name in [name for name in self.on_gpu if name != keep]:
            self.evict(name)

    def info(self) -> dict:
        return {"on_gpu": {name: round(size, 1) for name, size in self.on_gpu.items()},
                "in_ram": {name: round(size, 1) for name, size in self.in_ram.items()},
                "editors_gb": round(self.used_gb(), 1), "budget_gb": self.budget_gb,
                "ram_gb": round(self.ram_used_gb(), 1), "ram_budget_gb": self.ram_gb}


class LiveModels:
    """Qwen2.5-VL-7B, Grounding DINO + SAM 2.1 and the editors, on CUDA."""

    def __init__(self):
        add_repo_path()
        self.client = None
        self.grounder = None
        self.editors = GpuEditors()
        self.lock = threading.Lock()          # one request on the models at a time (pipeline.Jobs also queues)
        self.error = None
        self.warming = None                   # what the start-up is loading (for /api/health)
        try:                                  # TF32 matmuls and convolutions: faster fp32 models (LaMa, CodeFormer)
            import torch
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        except ImportError:
            pass

    # ------------------------------------------------------------ loading
    def load_reader(self) -> None:
        if self.client is None:
            from src.instructions.clients import get_client
            from src.editors.plan import DEFAULT_CLIENT
            started = time.monotonic()
            client = get_client(DEFAULT_CLIENT)
            client.load("cuda")
            self.client = client
            log.info("loaded %s in %.0f s", DEFAULT_CLIENT, time.monotonic() - started)

    def load_grounder(self) -> None:
        if self.grounder is None:
            from src.editors.grounding import Grounder
            started = time.monotonic()
            grounder = Grounder()
            grounder.load("cuda")
            self.grounder = grounder
            log.info("loaded Grounding DINO and SAM 2.1 in %.0f s", time.monotonic() - started)

    def warm(self) -> None:
        """At start-up: Qwen and the detectors, so the first request does not wait for them."""
        try:
            self.warming = "Qwen2.5-VL-7B"
            with self.lock:
                self.load_reader()
            self.warming = "Grounding DINO and SAM 2.1"
            with self.lock:
                self.load_grounder()
        except Exception as error:                  # reported by /api/health; requests fall back per step
            self.error = f"{type(error).__name__}: {error}"
            log.exception("loading the live models failed")
        self.warming = None

    def warm_editors(self, editors: list[str]) -> None:
        """At start-up, after the scorer: `editors` (the ones run.py --check found safe to move to RAM) onto the
        GPU and, when it is full, into RAM, so no request waits for a load from disk. The lock is taken per model,
        so a request that comes in meanwhile waits for one load at most."""
        for name in editors:
            if not self.editors.stats.get(name, "offload"):
                continue
            self.warming = name
            try:
                with self.lock:
                    if self.editors.where(name) == "disk":
                        self.editors.get(name)
            except Exception:
                log.exception("preloading %s failed", name)
        self.warming = None
        log.info("start-up loading done: %s", self.editors.info())

    def estimate(self, name: str) -> float | None:
        return self.editors.estimate(name)

    # ------------------------------------------------------------ one request
    def english(self, image_path: Path, request: str) -> str | None:
        from src.editors.plan import PROMPTS_DIR
        template = Template((PROMPTS_DIR / f"{REQUEST_PROMPT}.txt").read_text(encoding="utf-8"))
        prompt = template.substitute(request=" ".join(request.split())[:600].replace('"', "'"))
        answer = self.client.generate(prompt, image_path, REQUEST_TOKENS)
        instruction = parse_instruction(answer)
        if instruction is None:
            log.warning("request not understood by Qwen (%r); the request is sent as written", answer[:200])
        return instruction

    def prepare(self, image_path: Path, image, understood) -> EditContext:
        """Reads the request and, for the planned tasks, plans the edit and finds its region."""
        from src.editors.grounding import MASK_OPS, final_mask, needs_detectors
        from src.editors.plan import plan_one, plan_version

        with self.lock:
            context = EditContext(instruction=understood.english, op=sample_op(understood.task,
                                                                                  understood.operations))
            started = time.monotonic()
            self.load_reader()
            context.instruction = self.english(image_path, understood.instruction) or understood.english
            context.seconds["read"] = round(time.monotonic() - started, 2)
            if context.op in PLANNED_OPS:
                started = time.monotonic()
                sample = {"id": image_path.parent.name, "instruction": context.instruction,
                          "request": understood.instruction, "op": context.op}
                context.plan = plan_one(self.client, sample, image_path, plan_version(PLANNED_OPS[context.op]))
                context.seconds["plan"] = round(time.monotonic() - started, 2)
                if context.plan["problems"]:
                    log.info("plan problems: %s", context.plan["problems"])
                if context.op in MASK_OPS:
                    started = time.monotonic()
                    if needs_detectors(context.plan, context.instruction):
                        self.load_grounder()
                    context.mask, context.boxes = final_mask(self.grounder, image, context.plan,
                                                             context.instruction)
                    context.seconds["mask"] = round(time.monotonic() - started, 2)
            return context

    def edit(self, name: str, image, context: EditContext):
        """The output of editor `name` (PIL RGB), or an exception with a short reason."""
        import torch

        from src.editors.base import edit_one

        from src.editors import get_editor

        probe = get_editor(name)                    # the class's settings, without loading anything
        if not probe.applies({"op": context.op}, context.plan):
            raise ValueError(f"{name} does not apply to this request ({context.op})")
        if probe.uses_mask and context.mask is None:
            raise ValueError("the region to edit was not found in the photo")
        with self.lock:
            editor = self.editors.get(name)
            if editor.uses_mask:
                prompt = context.plan["prompt"] if context.plan else context.instruction
            else:
                prompt = context.instruction
            editor.op, editor.plan = context.op, context.plan
            started = time.monotonic()
            try:
                edited, status = edit_one(editor, image, context.mask, prompt)
            except torch.cuda.OutOfMemoryError:
                log.warning("%s ran out of GPU memory; dropping the other editors and trying again", name)
                self.editors.drop_all(keep=name)
                started = time.monotonic()
                edited, status = edit_one(editor, image, context.mask, prompt)
            self.editors.stats.record(name, edit_s=time.monotonic() - started)
            if status == "black":
                raise ValueError("the model returned a black image")
            return edited

    def info(self) -> dict:
        data = {"reader": self.client is not None, "grounder": self.grounder is not None, **self.editors.info(),
                "loading": self.warming, "error": self.error}
        try:
            data["gpu_free_gb"] = round(free_gb(), 1)
        except Exception:
            pass
        return data


# ---------------------------------------------------------------- server check (run.py --check)

CHECK_PROMPTS = {                 # what each kind of editor is asked in the check, on a sample photo
    "remove": "",
    "replace": "a red armchair",
    "add": "Add a red balloon in the sky.",
    "light": "Brighten this dark photo.",
    "enhance": "Make the photo sharper and clearer.",
    "restore": "Restore this old photo.",
    "colorize": "Colorize this black and white photo.",
    "cutout": "Remove the background.",
    "open": "Make it look like a winter day.",
}
CHECK_PLANS = {                   # the plan of the format editors (plan.py, plan_v4)
    "plain-code": {"action": "flip", "direction": "horizontal"},
    "sdxl-outpaint": {"action": "extend", "direction": "right", "scale": 1.3,
                      "prompt": "the same scene, continued to the right"},
}


def check_op(editor) -> str:
    """The op an editor is checked with: its first own op, else its first task's."""
    if editor.ops:
        return editor.ops[0]
    return {"remove": "remove", "add_replace": "replace", "enhance": "enhance", "restore": "restore",
            "change_color": "recolor", "formatting": "format"}.get(editor.tasks[0], "open")


def same_image(first, second, tolerance: float = 3.0) -> bool:
    """Whether two outputs match: the same size and a mean difference of at most `tolerance` (of 255)."""
    import numpy as np
    if first.size != second.size:
        return False
    difference = np.abs(np.asarray(first, dtype=np.float32) - np.asarray(second, dtype=np.float32))
    return float(difference.mean()) <= tolerance


def check_editors(names: list[str], image) -> list[dict]:
    """Every editor once: load from disk (downloading what is missing), edit `image` (a centred box as the mask of
    the mask models), then move it to RAM and back and edit again. The second edit must match the first for the
    editor to be kept in RAM by the app (`offload`); its time (CUDA warmed up) is the estimate of an edit. Records
    everything in EditorStats (work/web/editors.json), then drops the editor."""
    import torch
    from PIL import Image, ImageDraw

    add_repo_path()
    from src.editors.base import edit_one

    editors = GpuEditors(budget_gb=1e9)
    width, height = image.size
    mask = Image.new("L", image.size, 0)
    ImageDraw.Draw(mask).rectangle([width * 0.35, height * 0.35, width * 0.65, height * 0.65], fill=255)
    results = []
    for name in names:
        row = {"model": name, "status": "ok", "offload": False, "error": None}
        started = time.monotonic()
        try:
            torch.cuda.reset_peak_memory_stats()
            editor = editors.get(name)
            row["load_s"] = round(time.monotonic() - started, 1)
            op = check_op(editor)
            plan = {"op": op, "prompt": CHECK_PROMPTS.get(op, "a natural scene"), **CHECK_PLANS.get(name, {})}
            editor.op, editor.plan = op, plan
            prompt = (plan["prompt"] if op != "remove" else "") if editor.uses_mask else CHECK_PROMPTS.get(op, "")
            edit_mask = mask if editor.uses_mask else None
            began = time.monotonic()
            first, status = edit_one(editor, image, edit_mask, prompt)
            row.update(status=status, first_edit_s=round(time.monotonic() - began, 1), size=list(first.size))
            try:                                    # to RAM and back: does the editor still edit the same?
                began = time.monotonic()
                move_to(editor, "cpu")
                torch.cuda.empty_cache()
                move_to(editor, "cuda")
                row["ram_load_s"] = round(time.monotonic() - began, 1)
                began = time.monotonic()
                second, _ = edit_one(editor, image, edit_mask, prompt)
                row["edit_s"] = round(time.monotonic() - began, 1)
                row["offload"] = same_image(first, second)
            except Exception as error:
                row["offload_error"] = f"{type(error).__name__}: {error}"[:200]
            editors.stats.record(name, offload=row["offload"], edit_s=row.get("edit_s", row["first_edit_s"]),
                                 ram_load_s=row.get("ram_load_s", 0.0))
        except Exception as error:
            row.update(status="failed", error=f"{type(error).__name__}: {error}"[:300])
            log.exception("check of %s failed", name)
        row["peak_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 1)
        if name in editors.on_gpu:
            editors.on_gpu.pop(name)
        editor = None
        gc.collect()
        torch.cuda.empty_cache()
        results.append(row)
        print(f"{name:16} {row['status']:7} load {row.get('load_s', '-'):>5} s  edit {row.get('edit_s', '-'):>5} s  "
              f"RAM->GPU {row.get('ram_load_s', '-'):>4} s  offload {'yes' if row['offload'] else 'no':3}  "
              f"peak {row['peak_gb']:5.1f} GB  {row['error'] or row.get('offload_error') or ''}", flush=True)
    return results
