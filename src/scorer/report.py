"""One report of every trained run and every baseline (work/scorer/report.md and report.json).

Needs only small files, so it runs on a laptop after the Kaggle runs: runs/<name>/metrics.json and scores.jsonl,
baselines/*.jsonl (per-edit scores) and baselines.json (human ceiling).

- Trained runs: dev and test accuracy of the ensemble over seeds, with the mean and deviation of the single heads.
  Exactly one run is "selected": the best ensemble on dev. The others are listed for comparison; test results
  must never be used to choose.
- Baselines: dev and test accuracy on the pairs they scored (the reward models score the test split only).
- Paired differences on the test split: the selected run minus each baseline, on the pairs both scored, with a
  95% bootstrap interval over samples. An interval above 0 means the selected run is better.
- Latency of one request (two edits of a pair, latency.py), next to the test accuracy: latency.json (trained runs,
  similarity baselines) and baselines/<model>.latency.json (reward models, VLM judges).
"""

import json

from ..common.io import iter_jsonl, write_json
from ..paths import SCORER_DIR
from .evaluate import BASELINES_DIR, baseline_scores, eval_data, evaluate_scores, paired_difference
from .latency import LATENCY_FILE, load_latency


def load_runs() -> list[dict]:
    return [json.loads(path.read_text(encoding="utf-8"))
            for path in sorted((SCORER_DIR / "runs").glob("*/metrics.json"))]


def run_scores(name: str) -> dict[str, float]:
    return {row["image_id"]: row["score"] for row in iter_jsonl(SCORER_DIR / "runs" / name / "scores.jsonl")}


def load_latencies() -> dict[str, dict]:
    """Name -> latency of every trained run and baseline that was timed."""
    scorer = load_latency(LATENCY_FILE)
    latencies = {**scorer.get("runs", {}), **scorer.get("baselines", {})}
    for path in sorted(BASELINES_DIR.glob("*.latency.json")):
        latencies[path.name.removesuffix(".latency.json")] = load_latency(path)
    return latencies


def interval(result: dict) -> str:
    low, high = result.get("ci95") or (None, None)
    return f"{result['accuracy']} [{low}; {high}]" if result.get("accuracy") is not None else "-"


def report(version: str = "all") -> dict:
    pairs, rankings = eval_data(version)
    runs = load_runs()
    baselines = {name: evaluate_scores(scores, pairs, rankings) for name, scores in baseline_scores().items()}
    selected = max(runs, key=lambda run: run["ensemble"]["dev"]["accuracy"] or 0)["name"] if runs else None

    lines = ["## Trained runs (ensemble over seeds; seed mean +- sd in brackets)", "",
             "| run | status | dev | test (95% CI) | test seeds | test confident | test top-1 | best epochs |",
             "|---|---|---|---|---|---|---|---|"]
    for run in sorted(runs, key=lambda run: -(run["ensemble"]["dev"]["accuracy"] or 0)):
        dev, test, seeds = run["ensemble"]["dev"], run["ensemble"]["test"], run["seed_summary"]
        lines.append(f"| {run['name']} | {'selected' if run['name'] == selected else 'for comparison'} | "
                     f"{dev['accuracy']} | {interval(test)} | {seeds['test_mean']} +- {seeds['test_sd']} | "
                     f"{test['confident_accuracy']} | {test['top1']} (chance {test['top1_chance']}) | "
                     f"{[row['best_epoch'] for row in run['seeds']]} |")
    lines += ["", "## Baselines (no training on our labels)", "",
              "| baseline | dev | test (95% CI) | test pairs | test confident | test top-1 |",
              "|---|---|---|---|---|---|"]
    for name, result in sorted(baselines.items(), key=lambda item: -(item[1]["test"]["accuracy"] or 0)):
        dev, test = result["dev"], result["test"]
        lines.append(f"| {name} | {dev['accuracy'] if dev['pairs'] else '-'} | {interval(test)} | {test['pairs']} | "
                     f"{test['confident_accuracy']} | {test['top1']} (chance {test['top1_chance']}) |")

    differences = {}
    if selected:
        selected_scores = run_scores(selected)
        for name, scores in baseline_scores().items():
            differences[name] = paired_difference(pairs.get("test", []), selected_scores, scores)
        lines += ["", f"## {selected} minus each baseline (test, same pairs)", "",
                  "| baseline | pairs | difference (95% CI) |", "|---|---|---|"]
        for name, result in sorted(differences.items(), key=lambda item: -(item[1]["difference"] or 0)):
            low, high = result["ci95"]
            lines.append(f"| {name} | {result['pairs']} | {result['difference']} [{low}; {high}] |")

    latencies = load_latencies()
    if latencies:
        accuracy = {run["name"]: run["ensemble"]["test"]["accuracy"] for run in runs}
        accuracy.update({name: result["test"]["accuracy"] for name, result in baselines.items()})
        lines += ["", "## Latency of one request (2 edits of a pair, batch 1, after warm-up)", "",
                  "| model | kind | median ms | p90 ms | peak GPU memory GB | GPU | test accuracy |",
                  "|---|---|---|---|---|---|---|"]
        run_names = {run["name"] for run in runs}
        for name, latency in sorted(latencies.items(), key=lambda item: item[1]["median_ms"]):
            lines.append(f"| {name} | {'trained run' if name in run_names else 'baseline'} | {latency['median_ms']} | "
                         f"{latency['p90_ms']} | {latency.get('peak_gpu_memory_gb')} | "
                         f"{latency.get('gpu') or latency.get('device')} | {accuracy.get(name, '-')} |")

    human_path = SCORER_DIR / "baselines.json"
    human = json.loads(human_path.read_text(encoding="utf-8")).get("human", {}) if human_path.exists() else {}
    test_human = human.get("test", {})
    if test_human.get("annotator_vs_other_two") is not None:
        lines += ["", f"Human ceiling (test): one annotator agrees with the other two (when they agree) on "
                      f"{test_human['annotator_vs_other_two']} of the pairs."]
    (SCORER_DIR / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    result = {"selected": selected, "human": human,
              "runs": {run["name"]: {"dev": run["ensemble"]["dev"]["accuracy"],
                                     "test": run["ensemble"]["test"]["accuracy"],
                                     "test_ci95": run["ensemble"]["test"]["ci95"], **run["seed_summary"]}
                       for run in runs},
              "baselines": {name: {"dev": result["dev"]["accuracy"], "test": result["test"]["accuracy"],
                                   "test_ci95": result["test"]["ci95"], "test_pairs": result["test"]["pairs"]}
                            for name, result in baselines.items()},
              "selected_minus_baseline": differences, "latency": latencies}
    write_json(SCORER_DIR / "report.json", result)
    return result
