"""Turn a model's answer into a RequestAnalysis: the JSON object asked for by the prompt, with labels mapped to
RealEdit's names. Answers that are not clean JSON are still read where possible ("operation subject" lines), and
every problem is recorded instead of raising."""

import json
import re

from .schema import HUMOROUS, OPERATIONS, SUBJECTS, RequestAnalysis, normalize


def split_category(text: str) -> tuple[str | None, str | None]:
    """'Change-Color Background' -> ('Change-Color', 'Background'). Operation names may contain a space
    ("Change Color", "Open Ended"), so the longest matching operation prefix wins."""
    words = text.strip().strip("-*").split()
    for cut in (2, 1):
        if len(words) > cut:
            op = normalize(" ".join(words[:cut]), OPERATIONS)
            if op:
                return op, normalize(" ".join(words[cut:]), SUBJECTS)
    if len(words) == 1:
        return normalize(words[0], OPERATIONS), None
    return None, None


def parse_answer(text: str, sample_id: str, model: str, prompt: str) -> RequestAnalysis:
    problems, data = [], None
    match = re.search(r"\{.*\}", text, re.S)
    if match:
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            problems.append("invalid JSON")
    else:
        problems.append("no JSON object")
    if isinstance(data, dict):
        instruction = str(data.get("instruction") or "").strip()
        raw_categories = data.get("categories") or []
        if isinstance(raw_categories, str):
            raw_categories = [raw_categories]
    else:                                   # fall back to "operation subject" lines anywhere in the answer
        instruction = ""
        raw_categories = re.findall(r"^\s*[-*]?\s*([A-Za-z-]+(?: [A-Za-z/-]+){1,2})\s*$", text, re.M)
    if not instruction:
        problems.append("no instruction")
    if instruction.strip().strip(".").lower() == HUMOROUS:
        instruction = HUMOROUS
    categories = []
    for item in raw_categories:
        op, subject = split_category(str(item))
        if op is None or subject is None:
            problems.append(f"unknown category {item!r}")
            continue
        if {"operation": op, "subject": subject} not in categories:
            categories.append({"operation": op, "subject": subject})
    if not categories:
        problems.append("no category")
    return RequestAnalysis(sample_id, model, prompt, instruction, categories, text, problems)
