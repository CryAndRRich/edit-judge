"""The small benchmark set of each task: dev samples whose request categories match the task.

- remove: a category "Remove <subject>", except "Remove Background" (it asks for a plain or new background, which
  the cut-out models of list 7 handle; masking "the background" gives the erasers nothing sensible to fill);
- add_replace: a category "Add <subject>" or "Replace <subject>", half of each.

A request that also has a category of the other task (e.g. "Remove Person" and "Add Person") is left out, so a
sample belongs to one task only and its plan and mask are unambiguous. So are requests an inpainting step cannot
answer, found by words of the simplified instruction (the smoke run of 2026-11-20 showed each kind):
- humorous requests (RealEdit's "humorous", with or without a full stop);
- remove: cut-outs and crops ("cut out the bear ... transparent background", "crop so only the cat's head
  remains", "keeping only the bunny", "place this tree onto a blue background": the subject stays, list 7's job),
  retouching ("discoloration", "sweat", "acne": no object to segment, and SAM then masks the whole shirt or face)
  and restoration;
- add_replace: changes of shape or size ("make my cat longer"), which no mask expresses, and whole-image or
  identity changes ("swap the heads", "transform the cat into Solaire", "turn this building into Minions",
  "Star Wars themed", "blend the pug and the helmet"): the generalists of list 1 do those. "Remove the watermark"
  (filed under Add Logo) is a removal.
Add and Replace requests whose subject is Background get the op "background": the subject is kept and everything
else is repainted (grounding.py inverts its mask).

Requests with a single category come first (a request that also asks for other changes cannot be done by one
inpainting step), then the rest. Within that, samples are dealt round-robin over the subjects (Person,
Text/Patterns, ...), each subject in the fixed order of scorer/data.py:sample_order, so a small set still covers
many kinds of target and the first 5 (the smoke run) already differ. Only dev samples are used: the scorer chose
its run on dev but never trained on it, and test stays untouched.

The other tasks (added 2026-11-24 for lists 1 and 4 to 7; `GENERIC`) use the same order, rules and output, with
categories by their first word ("Enhance Person" -> enhance) and one op each:
- enhance: "light" when the instruction asks for more light (LOW_LIGHT, the keyword test of the Enhance run of
  2026-11-22; HVI-CIDNet runs on these only), else "enhance";
- restore: "restore";
- change_color: "colorize" when the original is greyscale (DDColor runs on these only), else "recolor" (one region
  painted in a new colour: plan_v4 and a mask, for SDXL Inpainting);
- formatting: half "Image-Formatting <subject>" ("format": plan_v4 names the action: crop, rotate, resize, extend,
  ...), half "Remove Background" ("cutout": BEN2 and BiRefNet), which the remove task leaves out;
- open_ended: "Open-Ended" and "Other" ("open"; the generalists of list 1 only).
A sample is in one task only: these tasks skip every sample already in another task's file (the sample files of
remove and add_replace are not changed by them), so the plans and masks, which are stored by sample ID, stay
unambiguous.

Output: work/editors/samples/<task>.jsonl, one line per sample in run order:
{"id", "op" (remove | add | replace | background | light | enhance | restore | colorize | recolor | cutout | format |
open), "category", "categories", "instruction", "request"}.
"""

import re
from collections import defaultdict

from ..common.dataset import text_of
from ..common.io import iter_jsonl, read_jsonl, write_jsonl
from ..paths import EDITORS_DIR, FINAL_PAIRS, SAMPLES_DIR
from ..scorer.data import sample_order, split_of

DEFAULT_SIZE = 30
EXCLUDED = {"Remove Background"}
OPERATIONS = {"remove": ("Remove",), "add_replace": ("Add", "Replace")}
UNSUITABLE = {
    "remove": ("cut out", "cutout", "cut-out", "transparent", "isolate", "extract", "png",
               "discolor", "discolour", "stain", "sweat", "acne", "blemish", "pimple", "wrinkle", "shine", "glare",
               "red eye", "red-eye", "redness", "dark circle", "bags under", "spots on", "skin",
               "crop", "keep only", "keeping only", "leave only", "leaving only", "onto", "restore"),
    "add_replace": ("longer", "shorter", "bigger", "larger", "smaller", "taller", "wider", "thinner", "slimmer",
                    "move", "resize", "stretch",
                    "swap", "transform", "turn", "themed", "style", "blend", "realistic", "clean version",
                    "watermark"),
}
OTHER_TASKS = {operation: {other for task, ops in OPERATIONS.items() if operation not in ops for other in ops}
               for ops in OPERATIONS.values() for operation in ops}     # Remove -> {Add, Replace}, ...

# Tasks of lists 1 and 4 to 7: task -> groups of (category prefix, op); each group gets an equal share of the samples
GENERIC = {
    "enhance": [("Enhance", "enhance")],
    "restore": [("Restore", "restore")],
    "change_color": [("Change-Color", "recolor")],
    "formatting": [("Image-Formatting", "format"), ("Remove Background", "cutout")],
    "open_ended": [("Open-Ended", "open"), ("Other", "open")],
}
GENERIC_UNSUITABLE = ("humorous",)
LOW_LIGHT = re.compile(r"\b(dark\w*|dim\w*|bright\w*|lighten\w*|low[- ]light|underexpos\w*|exposure|night\w*|"
                       r"illuminat\w*|lighting|adjust (the )?light)\b", re.I)
GREY_SPREAD = 4                 # mean |R-G| + |G-B| (0-255 scale) below which a photo counts as greyscale


def is_greyscale(sample_id: str) -> bool:
    """True when the original has (almost) no colour: a black-and-white photo to colourise."""
    import numpy as np
    from PIL import Image

    with Image.open(SAMPLES_DIR / sample_id / "original.jpg") as im:
        image = im.convert("RGB")
    image.thumbnail((256, 256))
    pixels = np.asarray(image, dtype=np.int16)
    spread = np.abs(pixels[..., 0] - pixels[..., 1]) + np.abs(pixels[..., 1] - pixels[..., 2])
    return float(spread.mean()) < GREY_SPREAD


def generic_op(task: str, op: str, sample_id: str, instruction: str) -> str:
    if task == "enhance" and LOW_LIGHT.search(instruction):
        return "light"
    if task == "change_color" and is_greyscale(sample_id):
        return "colorize"
    return op


def taken_elsewhere(task: str) -> set[str]:
    """Sample IDs already in the sample file of another task."""
    from . import TASKS
    return {row["id"] for other in TASKS if other != task for row in iter_jsonl(samples_path(other))}


def generic_candidates(requests: dict[str, dict], prefix: str, taken: set[str]) -> list[tuple[str, str]]:
    """(sample ID, category) of the dev samples with a category that starts with `prefix`, in run order: single
    category first, round-robin over subjects (as `candidates`)."""
    single, multi = defaultdict(list), defaultdict(list)
    for sample_id in sorted(requests, key=sample_order):
        if split_of(sample_id) != "dev" or sample_id in taken:
            continue
        categories = requests[sample_id]["categories"]
        instruction = " ".join((requests[sample_id]["instruction"] or "").lower().split())
        if re.sub(r"[^a-z]", "", instruction) in GENERIC_UNSUITABLE:
            continue
        matching = [c for c in categories if c == prefix or c.startswith(prefix + " ")]
        if prefix == "Remove Background":
            matching = [c for c in categories if c == prefix]
        if matching:
            subject = matching[0].split(" ", 1)[1] if " " in matching[0] else "Other"
            (single if len(categories) == 1 else multi)[subject].append((sample_id, matching[0]))
    pick = {}
    for groups in (single, multi):
        for sample_id, category in round_robin(groups):
            pick.setdefault(sample_id, category)
    return list(pick.items())


def select_generic(task: str, requests: dict[str, dict], size: int) -> list[dict]:
    groups = GENERIC[task]
    taken = taken_elsewhere(task)
    shares = [size // len(groups) + (index < size % len(groups)) for index in range(len(groups))]
    found = [generic_candidates(requests, prefix, taken) for prefix, _ in groups]
    shares = [min(share, len(candidates_of)) for share, candidates_of in zip(shares, found)]
    leftover = size - sum(shares)                 # a short group leaves its share to the others
    for group in range(len(groups)):
        extra = min(leftover, len(found[group]) - shares[group])
        shares[group] += extra
        leftover -= extra
    if leftover:
        print(f"only {size - leftover} dev samples for {task} (wanted {size})")
    chosen, seen = [], set()
    for index in range(max(shares)):              # interleave the groups
        for group, (prefix, op) in enumerate(groups):
            if index < shares[group] and found[group][index][0] not in seen:
                seen.add(found[group][index][0])
                chosen.append((*found[group][index], op))
    rows = []
    for sample_id, category, op in chosen:
        request = requests[sample_id]
        instruction = text_of({**request, "id": sample_id}, "instruction")
        rows.append({"id": sample_id, "op": generic_op(task, op, sample_id, instruction), "category": category,
                     "categories": request["categories"], "instruction": instruction,
                     "request": request["request"]})
    return rows


def samples_path(task: str):
    return EDITORS_DIR / "samples" / f"{task}.jsonl"


def load_task_samples(task: str, limit: int | None = None) -> list[dict]:
    rows = read_jsonl(samples_path(task))
    if not rows:
        raise SystemExit(f"no samples for {task}: run `python -m src.editors samples --task {task}` first")
    return rows[:limit] if limit is not None else rows


def dataset_requests() -> dict[str, dict]:
    """Sample ID -> {"request", "instruction", "categories"} from data/pairs.jsonl (one line per pair; every pair
    of a sample carries the same request fields)."""
    requests = {}
    for pair in iter_jsonl(FINAL_PAIRS):
        sample_id = pair["a"].split("/")[0]
        if sample_id not in requests:
            requests[sample_id] = {"request": pair["request"], "instruction": pair.get("instruction"),
                                   "categories": pair.get("categories") or []}
    return requests


def round_robin(groups: dict[str, list[str]]) -> list[str]:
    """One sample of every subject in turn (subjects in name order), until every list is used up."""
    order, queues = [], {subject: list(ids) for subject, ids in sorted(groups.items())}
    while any(queues.values()):
        for queue in queues.values():
            if queue:
                order.append(queue.pop(0))
    return order


def candidates(requests: dict[str, dict], operation: str) -> list[tuple[str, str]]:
    """(sample ID, category) of every dev sample with a category of `operation`, in run order."""
    single, multi = defaultdict(list), defaultdict(list)
    for sample_id in sorted(requests, key=sample_order):
        if split_of(sample_id) != "dev":
            continue
        categories = requests[sample_id]["categories"]
        instruction = " ".join((requests[sample_id]["instruction"] or "").lower().split())
        if re.sub(r"[^a-z]", "", instruction) == "humorous":
            continue
        task = next(task for task, ops in OPERATIONS.items() if operation in ops)
        if any(re.search(rf"\b{re.escape(word)}", instruction) for word in UNSUITABLE[task]):
            continue                                  # words match at a word start: "move" never matches "remove"
        if any(c.split(" ")[0] in OTHER_TASKS[operation] for c in categories):
            continue                                  # a sample belongs to one task only (plans, masks by ID)
        matching = [c for c in categories if c.split(" ")[0] == operation and c not in EXCLUDED]
        if matching:
            subject = matching[0].split(" ", 1)[1] if " " in matching[0] else "Other"
            (single if len(categories) == 1 else multi)[subject].append((sample_id, matching[0]))
    pick = {}
    for groups in (single, multi):
        for sample_id, category in round_robin(groups):
            pick.setdefault(sample_id, category)
    return list(pick.items())


def select(task: str, size: int = DEFAULT_SIZE) -> list[dict]:
    """The task's samples; written to work/editors/samples/<task>.jsonl."""
    requests = dataset_requests()
    if not requests:
        raise SystemExit(f"no pairs in {FINAL_PAIRS}: set EJ_DATA_DIR to the final dataset")
    if not any(r["categories"] for r in requests.values()):
        raise SystemExit(f"{FINAL_PAIRS} has no categories: use the dataset version exported after the request "
                         "analysis (2026-11-07 or later)")
    if task in GENERIC:
        rows = select_generic(task, requests, size)
        write_jsonl(samples_path(task), rows)
        return rows
    operations = OPERATIONS[task]
    shares = [size // len(operations) + (index < size % len(operations)) for index in range(len(operations))]
    per_operation = []
    for operation, share in zip(operations, shares):
        found = candidates(requests, operation)
        if len(found) < share:
            print(f"only {len(found)} dev samples for {operation} (wanted {share})")
        per_operation.append([(sample_id, category, operation) for sample_id, category in found[:share]])
    chosen, seen = [], set()
    for index in range(max(map(len, per_operation))):       # interleave the operations: Add, Replace, Add, ...
        for found in per_operation:
            if index < len(found) and found[index][0] not in seen:
                seen.add(found[index][0])
                chosen.append(found[index])
    rows = []
    for sample_id, category, operation in chosen:
        request = requests[sample_id]
        op = "background" if category.endswith(" Background") and operation != "Remove" else operation.lower()
        rows.append({"id": sample_id, "op": op, "category": category,
                     "categories": request["categories"],
                     "instruction": text_of({**request, "id": sample_id}, "instruction"),
                     "request": request["request"]})
    write_jsonl(samples_path(task), rows)
    return rows
