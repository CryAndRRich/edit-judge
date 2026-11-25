"""Import the small HF result ZIPs without extracting images or recreating missing coordinator files."""

import json
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

from ...common.io import write_json, write_jsonl
from .common import REVIEW_DIR, dataset, digest, identity, originals

MAX_FILE = 20 * 1024 * 1024


def documents(path: Path):
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            for item in archive.infolist():
                if item.filename.startswith("__MACOSX/") or Path(item.filename).name.startswith("._"):
                    continue
                if item.filename.endswith(".json"):
                    if item.file_size > MAX_FILE:
                        raise ValueError(f"oversized result JSON: {item.filename}")
                    yield f"{path.name}:{item.filename}", json.loads(archive.read(item).decode("utf-8-sig"))
    elif path.suffix == ".json":
        if path.stat().st_size > MAX_FILE:
            raise ValueError(f"oversized result JSON: {path.name}")
        yield path.name, json.loads(path.read_text(encoding="utf-8-sig"))


def prepare(results: Path | None = None, hf_repo: str | None = None, aliases: Path | None = None) -> dict:
    paths = sorted(results.rglob("*")) if results and results.is_dir() else ([results] if results else [])
    if hf_repo:
        from huggingface_hub import HfApi, hf_hub_download
        api = HfApi()
        revision = api.repo_info(hf_repo, repo_type="dataset").sha
        for item in api.list_repo_tree(hf_repo, repo_type="dataset", revision=revision, recursive=True):
            if item.path.startswith('checkpoints/'):
                continue
            if hasattr(item, "size") and item.path.endswith((".zip", ".json")):
                if item.size > MAX_FILE:
                    raise ValueError(f"refusing a large result archive: {item.path}")
                paths.append(Path(hf_hub_download(hf_repo, item.path, repo_type="dataset", revision=revision,
                                                  local_dir=str(REVIEW_DIR / "hf-cache"))))
    if not paths:
        raise ValueError("provide --results DIR or --hf-results IntroDS/RankingResults")
    rename = {identity(k): identity(v) for k, v in json.loads(aliases.read_text(encoding="utf-8")).items()} if aliases else {}
    found, files, skip_reasons = {}, 0, defaultdict(set)
    for path in paths:
        if not path.is_file():
            continue
        for source, doc in documents(path):
            if not isinstance(doc, dict) or not isinstance(doc.get("records"), dict):
                continue
            if doc.get("version") not in ("v1", "v2"):
                raise ValueError(f"{source}: expected original v1/v2 results")
            who = identity(doc["annotator"])
            who = rename.get(who, who)
            files += 1
            for sid, record in doc["records"].items():
                row = found.setdefault(sid, {"id": sid, "version": doc["version"], "rankings": {}, "skips": {}})
                if row["version"] != doc["version"]:
                    raise ValueError(f"{sid} appears in both v1 and v2")
                skip = record.get("skipped")
                rank = record.get("ranking")
                if not isinstance(skip, bool) or (not skip and (not isinstance(rank, list) or len(set(rank)) != len(rank))):
                    raise ValueError(f"{source}: invalid record {sid}")
                if (who in row["rankings"] and (skip or row["rankings"][who] != rank)
                        or who in row["skips"] and not skip):
                    raise ValueError(f"conflicting original results for {sid}/{who}; reconcile the ZIPs or aliases")
                if skip:
                    reason = str(record.get("skip_reason") or "").strip()
                    if not reason:
                        raise ValueError(f"{source}: skipped sample {sid} needs a reason")
                    skip_reasons[(sid, who)].add(reason)
                    row["skips"][who] = "\n---\n".join(sorted(skip_reasons[(sid, who)]))
                else:
                    row["rankings"][who] = rank
    if not found:
        raise ValueError("no original ranking records found")
    rows = []
    for sid, row in sorted(found.items()):
        row["annotators"] = sorted(set(row["rankings"]) | set(row["skips"]))
        rows.append(row)
    _, samples = dataset()
    missing = set(samples) - set(found)
    if missing:
        raise ValueError(f"missing original results for {len(missing)} final samples (e.g. {sorted(missing)[:3]})")
    for sid, sample in samples.items():
        for rank in found[sid]["rankings"].values():
            if not set(sample["keys"]) <= set(rank):
                raise ValueError(f"{sid}: original ranking does not cover the final edits")
    output = REVIEW_DIR / "originals.jsonl"
    if output.exists() and digest(list(originals().values())) != digest(rows):
        if list(REVIEW_DIR.glob("round-*/selected.jsonl")):
            raise ValueError("original reviewer identities are frozen after selection")
    write_jsonl(output, rows)
    report = {"files": files, "samples": len(rows), "final_samples": len(samples), "missing": 0,
              "duplicate_skip_reason_variants": sum(len(reasons) > 1 for reasons in skip_reasons.values()),
              "by_version": dict(Counter(row["version"] for row in rows)),
              "reviewers": sorted({name for row in rows for name in row["annotators"]}),
              "coordinator_flags": "hard-pair/noisy replacement flags used when work/ranking/v*/final is available"}
    write_json(REVIEW_DIR / "prepare.json", report)
    return report
