"""Single-image benchmark, independent per-model failures, checkpoint-safe warmup."""
import csv
import gc
import json
import time
import traceback
from pathlib import Path

import torch
from PIL import Image

from .download_data import MODELS, download_checkpoint, write_json
from .evaluate import metrics, latency, majority_vote


def make_adapter(name, checkpoint, threshold):
    if name == 'yolov8n':
        from .adapters.yolov8n import YoloAdapter
        cls = YoloAdapter
    elif name == 'efficientnetv2_b0':
        from .adapters.efficientnetv2_b0 import EfficientnetAdapter
        cls = EfficientnetAdapter
    else:
        from .adapters.convnext_tiny import ConvnextAdapter
        cls = ConvnextAdapter
    return cls(checkpoint, threshold)


def benchmark_model(name, rows, config, output):
    predictions, errors = [], []
    metadata = dict(status='starting', succeeded=0)
    adapter = None
    try:
        path, checkpoint = download_checkpoint(name, config['checkpoints'], output/'revisions.json')
        metadata['checkpoint'] = checkpoint
        adapter = make_adapter(name, path, config['threshold'])
        start = time.perf_counter()
        adapter.load(config['resolved_device'])
        metadata.update(load_s=time.perf_counter()-start, adapter=adapter.metadata)
        # Both labels exercised in warmup, before the measured loop.
        warm = [next(r for r in rows if r['true_label'] == y) for y in (0, 1)]
        start = time.perf_counter()
        for i in range(config['warmup']):
            with Image.open(warm[i % 2]['path']) as im:
                im.load(); adapter.predict(im)
        metadata.update(warmup_s=time.perf_counter()-start, warmup_iterations=config['warmup'])
        for i, row in enumerate(rows):
            try:
                # Decode outside timing; EXIF/RGB + preprocess + transfer + forward + postprocess inside.
                with Image.open(row['path']) as im:
                    im.load()
                    prediction = adapter.predict(im)
                predictions.append({k: row[k] for k in ('id', 'dataset', 'path', 'true_label')} |
                                   dict(model=name, **prediction))
            except Exception as exc:
                errors.append(dict(model=name, stage='predict', image_id=row['id'], error=str(exc), traceback=traceback.format_exc()))
            if (i+1) % 30 == 0 or i+1 == len(rows):
                print(f'{name}: {i+1}/{len(rows)} images, successful={len(predictions)}', flush=True)
        metadata.update(status='ok' if len(predictions) == len(rows) else 'partial', succeeded=len(predictions))
    except Exception as exc:
        metadata.update(status='failed', error=str(exc))
        errors.append(dict(model=name, stage='load_or_warmup', error=str(exc), traceback=traceback.format_exc()))
        print(f'{name}: FAILED: {exc}', flush=True)
    finally:
        del adapter
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    write_json(output/f'{name}_run.json', dict(metadata=metadata, predictions=predictions, errors=errors))
    return predictions, metadata, errors


def summarize(predictions, metadata, device, expected_counts):
    summaries = []
    for name, meta in metadata.items():
        for dataset, count in expected_counts.items():
            subset = [p for p in predictions if p['model'] == name and p['dataset'] == dataset]
            size = meta.get('checkpoint', {}).get('bytes', 0)
            row = dict(model=name, dataset=dataset, expected_n=count, model_bytes=size, device=device,
                       status='ok' if len(subset) == count and count > 0 else meta['status'], n=len(subset))
            if subset:
                row.update(metrics([p['true_label'] for p in subset], [p['label'] for p in subset]))
                row.update(latency([p['timing_ms'] for p in subset]))
                if all(p.get('model_only_ms') for p in subset):
                    row.update(latency([p['model_only_ms'] for p in subset], 'model'))
            summaries.append(row)
    return summaries


def ensemble(predictions, metadata, rows):
    if not all(metadata.get(name, {}).get('status') == 'ok' for name in MODELS):
        return [], dict(status='skipped_requires_three_successful_models')
    lookups = [{p['id']: p for p in predictions if p['model'] == name} for name in MODELS]
    expected = {r['id'] for r in rows}
    if any(set(x) != expected for x in lookups):
        return [], dict(status='skipped_unaligned_predictions')
    result = []
    for row in rows:
        votes = [lookup[row['id']] for lookup in lookups]
        score = sum(p['label'] for p in votes)/3
        result.append({k: row[k] for k in ('id', 'dataset', 'path', 'true_label')} |
                      dict(model='ensemble_voting', score=score, label=int(score >= .5), boxes=[],
                           timing_ms=sum(p['timing_ms'] for p in votes),
                           model_only_ms=sum(p['model_only_ms'] for p in votes)))
    return result, dict(status='ok', succeeded=len(result),
                        checkpoint={'bytes': sum(m['checkpoint']['bytes'] for m in metadata.values())},
                        timing_policy='Sum of measured per-image adapter times; sequential estimate, not separately timed voting pipeline')


def write_csv(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    if not fields:
        fields = ['model', 'dataset', 'n', 'status']
    with Path(path).open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v for k, v in row.items()})
