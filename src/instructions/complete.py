"""Complete the request analysis of the final dataset with the fewest model calls (PROJECT.md Part II Steps 2 and
4), and merge it into one file.

    python -m src.instructions complete --client qwen2.5-vl-7b --max-hours 10
    python -m src.instructions merge --client qwen2.5-vl-7b

- Instructions: RealEdit's own, where RealEdit published one (realedit.py); the model writes the others.
- Categories: RealEdit publishes none, so the model labels every request.
So `complete` runs two passes with one loaded model, within one time budget:
1. analyze_v2 (instruction and categories) on the samples without a RealEdit instruction (non-RealEdit sources);
2. categorize_v1 (categories only, RealEdit's taxonomy prompt) on the samples with a RealEdit instruction.
Results go to the usual run files (work/instructions/<client>/<prompt>.jsonl), so a stopped run goes on where it
stopped, and every merge writes work/instructions/<client>/final.jsonl, one line per analysed sample:

    {"id": "...", "instruction": "Remove the text.", "instruction_source": "realedit",
     "categories": ["Remove Text/Patterns"], "problems": []}

and final.summary.json (done, left, the hours left at the measured speed).
"""

import json
import time
from collections import Counter

from ..common.dataset import iter_samples
from ..common.io import iter_jsonl, write_json, write_jsonl
from ..paths import INSTRUCTIONS_DIR
from .realedit import realedit_instructions
from .run import result_files, run

PASSES = (("analyze", 2), ("categorize", 1))


def results_of(client_name: str, prompt_id: str) -> dict[str, dict]:
    return {row["sample_id"]: row for path in result_files(client_name, prompt_id) for row in iter_jsonl(path)}


def seconds_per_request(client_name: str) -> dict[str, float]:
    """Measured seconds per request of each pass (prompt id -> seconds)."""
    speeds = {}
    for prompt_name, version in PASSES:
        meta = INSTRUCTIONS_DIR / client_name / f"{prompt_name}_v{version}.meta.json"
        if meta.exists() and json.loads(meta.read_text()).get("seconds_per_request"):
            speeds[f"{prompt_name}_v{version}"] = json.loads(meta.read_text())["seconds_per_request"]
    return speeds


def merge(client_name: str, instructions: dict[str, str] | None = None) -> dict:
    instructions = realedit_instructions() if instructions is None else instructions
    full = results_of(client_name, "analyze_v2")
    categorized = results_of(client_name, "categorize_v1")
    lines, left, problems = [], Counter(), Counter()
    for sample in iter_samples():
        sample_id = sample["id"]
        if sample_id in instructions:
            instruction, source = instructions[sample_id], "realedit"
            label_row = categorized.get(sample_id) or full.get(sample_id)
            if label_row is None:
                left["categorize_v1"] += 1
                continue
            row_problems = [p for p in label_row["problems"] if p != "no instruction"]
        else:
            label_row = full.get(sample_id)
            if label_row is None:
                left["analyze_v2"] += 1
                continue
            instruction, source, row_problems = label_row["instruction"], "qwen", label_row["problems"]
        categories = [f"{c['operation']} {c['subject']}" for c in label_row["categories"]]
        problems.update(row_problems)
        lines.append({"id": sample_id, "instruction": instruction, "instruction_source": source,
                      "categories": categories, "problems": row_problems})
    folder = INSTRUCTIONS_DIR / client_name
    write_jsonl(folder / "final.jsonl", lines)
    speeds = seconds_per_request(client_name)
    slowest = max(speeds.values(), default=None)
    seconds_left = sum(count * speeds.get(prompt_id, slowest or 0) for prompt_id, count in left.items())
    total_left = sum(left.values())
    summary = {"client": client_name, "done": len(lines), "left": dict(left), "complete": total_left == 0,
               "instruction_source": dict(Counter(line["instruction_source"] for line in lines)),
               "with_problems": sum(bool(line["problems"]) for line in lines), "problems": dict(problems),
               "seconds_per_request": speeds,
               "hours_left": round(seconds_left / 3600, 1) if speeds else None}
    write_json(folder / "final.summary.json", summary)
    return summary


def complete(client, max_hours: float | None = None, device: str | None = None, allow_cpu: bool = False,
             max_tokens: int = 256, limit: int | None = None) -> dict:
    started = time.monotonic()
    instructions = realedit_instructions()
    sample_ids = {sample["id"] for sample in iter_samples()}
    groups = {"analyze": sample_ids - set(instructions), "categorize": sample_ids & set(instructions)}
    print(f"{len(sample_ids)} samples: {len(groups['categorize'])} with a RealEdit instruction (categories only), "
          f"{len(groups['analyze'])} without (instruction and categories)", flush=True)
    passes = {}
    for prompt_name, version in PASSES:
        hours_left = None if max_hours is None else max_hours - (time.monotonic() - started) / 3600
        if hours_left is not None and hours_left <= 0:
            break
        passes[f"{prompt_name}_v{version}"] = run(client, prompt_name, version, limit=limit, device=device,
                                                  allow_cpu=allow_cpu, max_tokens=max_tokens, max_hours=hours_left,
                                                  only=groups[prompt_name])
    return {"passes": passes, "merged": merge(client.name, instructions),
            "hours": round((time.monotonic() - started) / 3600, 2)}
