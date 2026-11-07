"""Turn a model's answer into a RequestAnalysis: the JSON object asked for by the prompt, with labels mapped to
RealEdit's names. Answers that are not clean JSON are still read where possible ("operation subject" lines), and
every problem is recorded instead of raising. Labels outside the taxonomy in a JSON answer ("Move Person",
"Change-Color Helmet", a bare "Enhance") are mapped to the closest taxonomy name (loose_category) and recorded."""

import json
import re

from .schema import (BACKGROUND_REMOVALS, ENHANCE_REMOVALS, HUMOROUS, OPERATION_WORDS, OPERATIONS, SUBJECT_WORDS,
                     SUBJECTS, RequestAnalysis, normalize)

LOOSE_OPERATIONS = {word: name for name, words in OPERATION_WORDS.items() for word in words}
LOOSE_SUBJECTS = {word: name for name, words in SUBJECT_WORDS.items() for word in words}
PLACEHOLDER = {"operation": "Other", "subject": "Other"}


def word_key(text: str) -> str:
    return text.lower().replace("_", " ").replace("-", " ").strip(".").strip()


def loose_category(text: str) -> tuple[str, str]:
    """The closest taxonomy category of a label outside the taxonomy: the operation from its first words (a
    taxonomy name or a word of OPERATION_WORDS, else "Other"), the subject from the first later word that names
    one (else "Other"). "Remove Reflection" is Enhance, "Isolate Animal" is Remove Background."""
    words = text.replace("-", " ").split()       # "Change-Clothing" reads as "Change Clothing"
    if words and word_key(words[0]) in BACKGROUND_REMOVALS:
        return "Remove", "Background"
    operation, rest = None, words
    for cut in (2, 1):
        if len(words) >= cut:
            head = " ".join(words[:cut])
            operation = normalize(head, OPERATIONS) or LOOSE_OPERATIONS.get(word_key(head))
            if operation:
                rest = words[cut:]
                break
    if operation == "Remove" and word_key(" ".join(rest)) in ENHANCE_REMOVALS:
        return "Enhance", "Other"
    for start in range(len(rest)):
        for cut in (2, 1):
            if start + cut <= len(rest):
                part = " ".join(rest[start:start + cut])
                subject = normalize(part, SUBJECTS) or LOOSE_SUBJECTS.get(word_key(part))
                if subject:
                    return operation or "Other", subject
    return operation or "Other", "Other"


def first_subject(words: list[str]) -> tuple[str | None, int]:
    """The subject at the start of `words` (one or two words, e.g. "Body Part") and how many words it takes."""
    for cut in (2, 1):
        if len(words) >= cut:
            subject = normalize(" ".join(words[:cut]), SUBJECTS)
            if subject:
                return subject, cut
    return None, 0


def split_category(text: str) -> tuple[str | None, str | None]:
    """'Change-Color Background' -> ('Change-Color', 'Background'). Operation names may contain a space
    ("Change Color", "Open Ended"), so the longest matching operation prefix wins."""
    op, subject, _ = read_category(text)
    return op, subject


def read_category(text: str) -> tuple[str | None, str | None, bool]:
    """(operation, subject, extra): like split_category, but when the subject is followed by more words
    ("Remove Background Person") the first subject is kept and `extra` is True."""
    words = text.strip().strip("-*").split()
    for cut in (2, 1):
        if len(words) > cut:
            op = normalize(" ".join(words[:cut]), OPERATIONS)
            if op:
                rest = words[cut:]
                subject = normalize(" ".join(rest), SUBJECTS)
                if subject:
                    return op, subject, False
                subject, used = first_subject(rest)
                return op, subject, subject is not None and used < len(rest)
    if len(words) == 1:
        return normalize(words[0], OPERATIONS), None, False
    return None, None, False


def join_split_pairs(items: list[str]) -> list[str]:
    """["Remove", "Background"] -> ["Remove Background"]: a bare operation followed by a bare subject is one
    category written as two list items."""
    out, i = [], 0
    while i < len(items):
        if i + 1 < len(items) and normalize(items[i], OPERATIONS) and normalize(items[i + 1], SUBJECTS):
            out.append(f"{items[i]} {items[i + 1]}")
            i += 2
        else:
            out.append(items[i])
            i += 1
    return out


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
    if not instruction and not prompt.startswith("categorize"):   # categorize prompts ask for categories only
        problems.append("no instruction")
    if instruction.strip().strip(".").lower() == HUMOROUS:
        instruction = HUMOROUS
    categories = []
    for item in join_split_pairs([str(raw) for raw in raw_categories]):
        op, subject, extra = read_category(item)
        if op is None and subject is None and categories and normalize(item, SUBJECTS):
            # ["Remove Background", "Person"]: a second subject of the same operation
            op, subject = categories[-1]["operation"], normalize(item, SUBJECTS)
            problems.append(f"operation of the previous category for {item!r}")
        if (op is None or subject is None) and isinstance(data, dict):
            op, subject = loose_category(item)
            problems.append(f"mapped {item!r} to '{op} {subject}'")
        if op is None or subject is None:
            problems.append(f"unknown category {item!r}")
            continue
        if extra:
            problems.append(f"extra words in {item!r}")
        if {"operation": op, "subject": subject} not in categories:
            categories.append({"operation": op, "subject": subject})
    if len(categories) > 1 and PLACEHOLDER in categories:    # a bare "Other" next to real categories
        categories.remove(PLACEHOLDER)
    if not categories:
        problems.append("no category")
    return RequestAnalysis(sample_id, model, prompt, instruction, categories, text, problems)
