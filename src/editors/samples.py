"""The small benchmark set of each task: dev samples whose request categories match the task.

- remove: a category "Remove <subject>", except "Remove Background" (it asks for a plain or new background, which
  the cut-out models of list 7 handle; masking "the background" gives the erasers nothing sensible to fill);
- add_replace: a category "Add <subject>" or "Replace <subject>", half of each.

A request that also has a category of the other task (e.g. "Remove Person" and "Add Person") is left out, so a
sample belongs to one task only and its plan and mask are unambiguous. Requests with a single category come first (a request that also asks for other changes cannot be done by one
inpainting step), then the rest. Within that, samples are dealt round-robin over the subjects (Person,
Text/Patterns, ...), each subject in the fixed order of scorer/data.py:sample_order, so a small set still covers
many kinds of target and the first 5 (the smoke run) already differ. Only dev samples are used: the scorer chose
its run on dev but never trained on it, and test stays untouched.

Output: work/editors/samples/<task>.jsonl, one line per sample in run order:
{"id", "op" (remove | add | replace), "category", "categories", "instruction", "request"}.
"""

from collections import defaultdict

from ..common.dataset import text_of
from ..common.io import iter_jsonl, read_jsonl, write_jsonl
from ..paths import EDITORS_DIR, FINAL_PAIRS
from ..scorer.data import sample_order, split_of

DEFAULT_SIZE = 30
EXCLUDED = {"Remove Background"}
OPERATIONS = {"remove": ("Remove",), "add_replace": ("Add", "Replace")}
OTHER_TASKS = {operation: {other for task, ops in OPERATIONS.items() if operation not in ops for other in ops}
               for ops in OPERATIONS.values() for operation in ops}     # Remove -> {Add, Replace}, ...


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
        if (requests[sample_id]["instruction"] or "").strip().lower() == "humorous":
            continue
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
        rows.append({"id": sample_id, "op": operation.lower(), "category": category,
                     "categories": request["categories"],
                     "instruction": text_of({**request, "id": sample_id}, "instruction"),
                     "request": request["request"]})
    write_jsonl(samples_path(task), rows)
    return rows
