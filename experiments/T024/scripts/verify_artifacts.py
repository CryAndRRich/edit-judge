"""Recompute acceptance checks against on-disk measurements; never creates scores."""
import csv
import json
import math
import subprocess
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.download_data import MODELS, sha256, validate_manifest, write_json


def main():
    output = Path(sys.argv[1] if len(sys.argv) > 1 else 'reports/latest')
    result = json.loads((output/'results.json').read_text(encoding='utf-8'))
    with (output/'predictions.csv').open(encoding='utf-8', newline='') as f:
        predictions = list(csv.DictReader(f))
    public = validate_manifest('data/public_122')
    synthetic = json.loads(Path('data/synthetic_manifest.json').read_text(encoding='utf-8'))
    expected = {r['id']: r for r in public+synthetic}
    assert len(expected) == 182
    image_bytes = sum(Path(r['path']).stat().st_size for r in expected.values())
    assert image_bytes <= result['config']['data_limit_mb']*1_000_000
    lookup = {}
    for p in predictions:
        key = (p['model'], p['id'])
        assert key not in lookup
        lookup[key] = p
        assert p['id'] in expected
        assert int(p['true_label']) == expected[p['id']]['true_label']
        score = float(p['score'])
        assert math.isfinite(score) and 0 <= score <= 1
        assert int(p['label']) == int(score >= result['config']['threshold'])
        assert float(p['timing_ms']) > 0 and float(p['model_only_ms']) > 0
        json.loads(p['boxes'])
    for model, meta in result['models'].items():
        assert meta['status'] == 'ok'
        assert {p['id'] for p in predictions if p['model'] == model} == set(expected)
        if model in MODELS:
            _, filename = MODELS[model]
            checkpoint = Path('checkpoints')/model/filename
            assert sha256(checkpoint) == meta['checkpoint']['sha256'] == meta['checkpoint']['remote_lfs_sha256']
    for s in result['summary']:
        rows = [p for p in predictions if p['model'] == s['model'] and p['dataset'] == s['dataset']]
        counts = Counter((int(p['true_label']), int(p['label'])) for p in rows)
        tn, fp, fn, tp = (counts[x] for x in [(0,0),(0,1),(1,0),(1,1)])
        assert [tn,fp,fn,tp] == [s[x] for x in ['TN','FP','FN','TP']]
        assert len(rows) == s['n'] == s['expected_n']
        assert math.isclose((tp+tn)/len(rows), s['accuracy'])
        measured = sorted(float(p['timing_ms']) for p in rows)
        assert math.isclose(sum(measured)/len(measured), s['e2e_mean_ms'])
    for image_id in expected:
        vote = lookup['ensemble_voting', image_id]
        singles = [lookup[name, image_id] for name in MODELS]
        assert int(vote['label']) == int(sum(int(p['label']) for p in singles) >= 2)
        assert math.isclose(float(vote['timing_ms']), sum(float(p['timing_ms']) for p in singles))
    for path, digest in result['code_sha256'].items():
        assert sha256(path) == digest, f'Runtime source changed since measured run: {path}'
    assert result['errors'] == json.loads((output/'errors.json').read_text(encoding='utf-8')) == []
    with (output/'failure_cases/manifest.csv').open(encoding='utf-8', newline='') as f:
        gallery = list(csv.DictReader(f))
    for row in gallery:
        assert sha256(row['source']) == row['source_sha256']
        assert (output/'failure_cases'/row['preview']).exists()
        pred = lookup[row['model'], row['id']]
        assert pred['label'] != pred['true_label']
    for model in result['models']:
        for kind, truth in [('FP',0), ('FN',1)]:
            failures = [p for p in predictions if p['model'] == model and int(p['true_label']) == truth and p['label'] != p['true_label']]
            chosen = [r for r in gallery if r['model'] == model and r['type'] == kind]
            assert len(chosen) == min(5,len(failures))
    tracked = subprocess.check_output(['git','ls-files','data','checkpoints'],text=True)
    assert not tracked.strip()
    subprocess.run(['git','check-ignore','data/public_122/validation_data.csv','checkpoints/yolov8n/best.pt','.venv-t024/pyvenv.cfg'],check=True,capture_output=True)
    report_images = sum(p.stat().st_size for p in output.rglob('*') if p.suffix.lower() in ('.jpg','.jpeg','.png'))
    assert image_bytes+report_images <= 25_000_000
    verification = dict(status='passed', predictions_total=len(predictions), neural_models=3,
                        model_inference_predictions=sum(p['model'] in MODELS for p in predictions),
                        voting_derived_predictions=sum(p['model'] == 'ensemble_voting' for p in predictions),
                        per_model_images=182, public_images=122, synthetic_images=60,
                        original_and_synthetic_image_bytes=image_bytes, report_preview_bytes=report_images,
                        including_report_previews_bytes=image_bytes+report_images, failure_previews=len(gallery),
                        checkpoint_hashes_match_remote=True, metrics_recomputed=True, voting_verified=True,
                        source_hashes_match_run=True, data_weights_gitignored=True,
                        tests_log=(output/'tests.log').read_text(encoding='utf-16' if (output/'tests.log').read_bytes().startswith(b'\xff\xfe') else 'utf-8'))
    write_json(output/'verification.json',verification)
    print(json.dumps(verification,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
