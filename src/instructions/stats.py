"""Statistics of one run, compared with RealEdit's published test-set distribution, and agreement between two
runs (two models, or two prompt versions) on the requests both analysed."""

from collections import Counter

from ..common.io import read_jsonl
from .run import result_files
from .schema import HUMOROUS

# RealEdit test set, Table 5 (operations, % of categorised requests), with the table's short names mapped to the
# taxonomy names
REALEDIT_OPERATIONS = {"Remove": 31.9, "Enhance": 14.5, "Add": 12.5, "Change-Color": 11.5, "Restore": 9.7,
                       "Replace": 8.4, "Open-Ended": 7.9, "Image-Formatting": 3.4, "Other": 0.1}


def load_run(run_id: str) -> dict[str, dict]:
    """run_id is "<client>:<prompt>", e.g. "qwen2.5-vl-7b:analyze_v1"; returns sample_id -> row, from the main
    result file and the shard files."""
    client, prompt = run_id.split(":")
    return {r["sample_id"]: r for f in result_files(client, prompt) for r in read_jsonl(f)}


def stats(run_id: str) -> dict:
    rows = list(load_run(run_id).values())
    if not rows:
        raise SystemExit(f"no results for {run_id}")
    ops = Counter(c["operation"] for r in rows for c in r["categories"])
    subjects = Counter(c["subject"] for r in rows for c in r["categories"])
    n_ops = sum(ops.values()) or 1
    problems = Counter(p.split(" '")[0] for r in rows for p in r["problems"])
    return {
        "run": run_id,
        "requests": len(rows),
        "with_problems": sum(bool(r["problems"]) for r in rows),
        "usable": sum(bool(r["instruction"] and r["categories"]) for r in rows),   # instruction and >= 1 category
        "problems": dict(problems.most_common()),
        "humorous": sum(r["instruction"] == HUMOROUS for r in rows),
        "categories_per_request": dict(sorted(Counter(len(r["categories"]) for r in rows).items())),
        "operations_percent": {op: {"ours": round(100 * ops[op] / n_ops, 1), "realedit": REALEDIT_OPERATIONS[op]}
                               for op in REALEDIT_OPERATIONS},
        "subjects_percent": {s: round(100 * k / n_ops, 1) for s, k in subjects.most_common()},
    }


def kappa(a: list[str], b: list[str]) -> float:
    """Cohen's kappa of two label lists."""
    n = len(a)
    if not n:
        return 0.0
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[k] * cb[k] for k in ca) / (n * n)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def agree(run_a: str, run_b: str) -> dict:
    a, b = load_run(run_a), load_run(run_b)
    common = [i for i in a if i in b and not a[i]["problems"] and not b[i]["problems"]]
    pairs = lambda r: {(c["operation"], c["subject"]) for c in r["categories"]}
    ops = lambda r: {c["operation"] for c in r["categories"]}
    first = lambda r: r["categories"][0]["operation"]
    jaccard = [len(pairs(a[i]) & pairs(b[i])) / len(pairs(a[i]) | pairs(b[i])) for i in common]
    return {
        "runs": [run_a, run_b],
        "requests_compared": len(common),
        "same_categories": round(sum(pairs(a[i]) == pairs(b[i]) for i in common) / max(len(common), 1), 3),
        "same_operations": round(sum(ops(a[i]) == ops(b[i]) for i in common) / max(len(common), 1), 3),
        "mean_jaccard_categories": round(sum(jaccard) / max(len(jaccard), 1), 3),
        "first_operation_kappa": round(kappa([first(a[i]) for i in common], [first(b[i]) for i in common]), 3),
        "humorous_agreement": round(sum((a[i]["instruction"] == HUMOROUS) == (b[i]["instruction"] == HUMOROUS)
                                        for i in common) / max(len(common), 1), 3),
    }
