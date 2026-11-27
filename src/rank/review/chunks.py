"""Reviewer-specific blinded chunks. The decoding map stays with the coordinator."""

import itertools
import json
import random
import zipfile
from collections import Counter, defaultdict

from ...paths import FINAL_PAIRS, RANKING_APP, RANKING_REQUIREMENTS, SAMPLES_DIR
from .common import dataset, digest, fingerprint, frozen_json, identity, round_dir, selected


def make_chunks(number: int, reviewers: list[str], chunk_size: int = 25, seed: int = 42,
                calibration: int = 10) -> dict:
    reviewers = sorted({identity(name) for name in reviewers})
    if len(reviewers) < 2 or chunk_size < 1 or calibration < 0:
        raise ValueError("need at least two reviewers, positive chunk size and nonnegative calibration size")
    folder = round_dir(number)
    picks = selected(number)
    selection_inputs = json.loads((folder / "selection-inputs.json").read_text(encoding="utf-8"))
    if fingerprint(FINAL_PAIRS) != selection_inputs["pairs_sha256"]:
        raise ValueError("labels changed after selection")
    _, samples = dataset()
    rng, loads, assigned = random.Random(seed), Counter(), defaultdict(list)
    assignment = {}
    for pick in sorted(picks, key=lambda row: row["id"]):
        candidates = [pair for pair in itertools.combinations(reviewers, 2)
                      if set(pair) - set(pick["original_annotators"])]
        if not candidates:
            raise ValueError(f"{pick['id']}: roster needs a reviewer outside the original annotators")
        rng.shuffle(candidates)
        pair = min(candidates, key=lambda pair: (max(loads[name] for name in pair), sum(loads[name] for name in pair)))
        assignment[pick["id"]] = list(pair)
        for name in pair:
            assigned[(name, "review")].append(pick["id"])
            loads[name] += 1
    calibration_ids = rng.sample(sorted(assignment), min(calibration, len(assignment)))
    if calibration_ids:
        for name in reviewers:
            assigned[(name, "calibration")] = calibration_ids.copy()
    mapping, manifest = {}, {"round": number, "assignments": assignment, "chunks": {},
                             "calibration_ids": calibration_ids, "reviewers": reviewers}
    payloads = []
    for (name, kind), ids in sorted(assigned.items()):
        rng.shuffle(ids)
        tag = digest(name)[:12]
        for start in range(0, len(ids), chunk_size):
            chunk = f"review{number:02d}_{kind}_{tag}_{start // chunk_size + 1:02d}"
            records, images, decode = [], [], {}
            for sid in ids[start:start + chunk_size]:
                sample = samples[sid]
                keys = sample["keys"].copy()
                rng.shuffle(keys)
                blinded = {f"E{i + 1:02d}": key for i, key in enumerate(keys)}
                decode[sid] = blinded
                record = {"id": sid, "request": sample["request"],
                          "original": f"samples/{sid}/original.jpg", "edits": []}
                images.append((SAMPLES_DIR / sid / "original.jpg", f"{chunk}/{record['original']}"))
                for label, key in blinded.items():
                    relative = f"samples/{sid}/{label}.jpg"
                    record["edits"].append({"key": label, "image": relative, "note": None})
                    images.append((SAMPLES_DIR / sid / f"{key}.jpg", f"{chunk}/{relative}"))
                records.append(record)
            for source, _ in images:
                if not source.is_file():
                    raise ValueError(f"missing final dataset image: {source}")
            meta = {"version": f"review-{number:02d}", "chunk": chunk, "review": True,
                    "reviewer": name, "samples": records}
            # Kind (calibration/review) is already in the chunk name; never include split, signals or old ranks.
            manifest["chunks"][chunk] = {"reviewer": name, "kind": kind, "ids": list(decode)}
            mapping[chunk] = decode
            payloads.append((chunk, meta, images))
    frozen_json(folder / "chunks-inputs.json", {"reviewers": reviewers, "chunk_size": chunk_size, "seed": seed,
                "calibration": calibration, "selected_sha256": digest(picks), "mapping_sha256": digest(mapping),
                "manifest_sha256": digest(manifest)})
    frozen_json(folder / "chunk-map.json", mapping)
    frozen_json(folder / "manifest.json", manifest)
    distribution = folder / "dist"
    distribution.mkdir(exist_ok=True)
    for chunk, meta, images in payloads:
        with zipfile.ZipFile(distribution / f"{chunk}.zip", "w", compression=zipfile.ZIP_STORED) as archive:
            archive.writestr(f"{chunk}/chunk.json", json.dumps(meta, ensure_ascii=False))
            for source, target in images:
                archive.write(source, target)
    with zipfile.ZipFile(distribution / "ranking_app_review.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for name in ("app.py", "requirements.txt", "run_mac.command", "run_windows.bat"):
            source = RANKING_REQUIREMENTS if name == "requirements.txt" else RANKING_APP / name
            archive.write(source, f"ranking_app/{name}")
        archive.writestr("ranking_app/README.txt", "Put your assigned chunk ZIPs in chunks/. Run run_windows.bat "
                         "or run_mac.command. Enter the assigned reviewer name. Return results/*.json to the "
                         "coordinator. Judge request fulfillment, naturalness, then preservation. Keep photos private.")
    return {"samples": len(assignment), "chunks": len(payloads), "reviews": sum(loads.values()),
            "reviewer_load": dict(loads), "calibration_samples": len(calibration_ids), "dist": str(distribution)}
