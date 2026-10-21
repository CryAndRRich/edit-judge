"""One-command T-024 data → inference → measured report pipeline."""
import argparse
import datetime as dt
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

# Disable optional library auto-installs and telemetry before importing adapters.
os.environ.setdefault('YOLO_AUTOINSTALL', 'false')
os.environ.setdefault('YOLO_CONFIG_DIR', str(Path('.venv-t024/yolo_config').resolve()))
Path(os.environ['YOLO_CONFIG_DIR']).mkdir(parents=True, exist_ok=True)
os.environ.setdefault('HF_HUB_DISABLE_TELEMETRY', '1')

import torch
from .benchmark import benchmark_model, ensemble, summarize, write_csv
from .download_data import MODELS, download_dataset, write_json, sha256
from .generate_synthetic import generate_synthetic
from .report import failure_gallery, confusion_svg, markdown_report


def now():
    return dt.datetime.now(dt.timezone(dt.timedelta(hours=7))).isoformat()


def environment():
    packages = ['torch', 'torchvision', 'timm', 'ultralytics', 'huggingface_hub', 'numpy', 'Pillow', 'pytest', 'opencv-python', 'safetensors']
    versions = {}
    for name in packages:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = 'not installed'
    cpu = platform.processor()
    if os.name == 'nt':
        try:
            cpu = subprocess.check_output(['powershell', '-NoProfile', '-Command', '(Get-CimInstance Win32_Processor).Name'], text=True, timeout=20).strip()
        except Exception:
            pass
    return dict(python=sys.version, executable=sys.executable, platform=platform.platform(), cpu=cpu,
                logical_cpus=os.cpu_count(), versions=versions, torch_cuda=torch.version.cuda,
                cuda_available=torch.cuda.is_available(), cudnn=torch.backends.cudnn.version(),
                gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                gpu_memory_bytes=torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else None,
                torch_threads=torch.get_num_threads(), interop_threads=torch.get_num_interop_threads(),
                deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                cudnn_benchmark=torch.backends.cudnn.benchmark,
                matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                system_site_packages='venv inherits existing CUDA torch; see pyvenv.cfg')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--data-limit-mb', type=float, default=25)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--threshold', type=float, default=.5)
    parser.add_argument('--output', default='reports/latest')
    parser.add_argument('--data', default='data')
    parser.add_argument('--checkpoints', default='checkpoints')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--warmup', type=int, default=8)
    args = parser.parse_args()
    if not 0 <= args.threshold <= 1 or args.warmup < 5 or args.threads < 1:
        parser.error('threshold in [0,1], warmup >=5, threads >=1 required')
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    config = vars(args).copy()
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    config['resolved_device'] = device if args.device == 'auto' else args.device
    result = dict(task='T-024', started_at=now(), config=config, environment=environment(), models={}, errors=[], summary=[])
    print(json.dumps(result['environment'], ensure_ascii=False), flush=True)
    rows = []
    try:
        public, dataset_meta = download_dataset(args.data, output/'revisions.json', args.data_limit_mb)
        result['dataset'] = dataset_meta
        rows.extend(public)
        shutil.copyfile(Path(args.data)/'public_manifest.json', output/'public_manifest.json')
        try:
            synthetic, meta = generate_synthetic(public, args.data, output, args.seed, args.data_limit_mb)
            result['synthetic'] = meta
            rows.extend(synthetic)
            if synthetic:
                shutil.copyfile(Path(args.data)/'synthetic_manifest.csv', output/'synthetic_manifest.csv')
        except Exception as exc:
            result['errors'].append(dict(stage='synthetic', error=str(exc), traceback=traceback.format_exc()))
    except Exception as exc:
        result['errors'].append(dict(stage='data', error=str(exc), traceback=traceback.format_exc()))
        print(f'Data failed: {exc}', flush=True)
    predictions = []
    for name in MODELS:
        if rows:
            pred, metadata, errors = benchmark_model(name, rows, config, output)
            predictions.extend(pred)
            result['models'][name] = metadata
            result['errors'].extend(errors)
        else:
            # Downloads and source verification are independent of data validation.
            from .download_data import download_checkpoint
            try:
                _, metadata = download_checkpoint(name, args.checkpoints, output/'revisions.json')
                result['models'][name] = dict(status='blocked_data', checkpoint=metadata)
            except Exception as exc:
                result['models'][name] = dict(status='failed', error=str(exc))
                result['errors'].append(dict(model=name, stage='checkpoint', error=str(exc), traceback=traceback.format_exc()))
        write_json(output/'progress.json', result)
    if rows:
        vote, vote_meta = ensemble(predictions, result['models'], rows)
        result['ensemble'] = vote_meta
        if vote:
            result['models']['ensemble_voting'] = vote_meta
            predictions.extend(vote)
    counts = {dataset: sum(r['dataset'] == dataset for r in rows) for dataset in ('public_122', 'synthetic')}
    counts = {k: v for k, v in counts.items() if v} or {'public_122': 122}
    result['summary'] = summarize(predictions, result['models'], config['resolved_device'], counts)
    result['failure_cases'] = failure_gallery(predictions, output)
    result['mapping_examples'] = []
    for name in MODELS:
        for label in (0, 1):
            examples = [p for p in predictions if p['model'] == name and p['dataset'] == 'public_122' and p['true_label'] == label][:3]
            result['mapping_examples'].extend({k: p[k] for k in ('model', 'id', 'true_label', 'score', 'label')} for p in examples)
    result['code_sha256'] = {str(p).replace('\\','/'): sha256(p) for p in Path('src').rglob('*.py')}
    result['source_provenance'] = dict(
        convnext_code_repo='https://github.com/boomb0om/watermark-detection',
        convnext_code_revision='1b770cf6075a895c61200633699cd5f90664b389',
        reviewed_sources_sha256={str(p.name): sha256(p) for p in (output/'sources').glob('*') if p.is_file()})
    result['timing_policy'] = dict(
        end_to_end='EXIF/RGB + preprocess + device transfer + forward + postprocess, excludes image decode/disk I/O/download/load/warmup',
        model_only='classifier forward only on prepared device tensor; YOLO library inference timer',
        ensemble='sum of per-image adapter measurements; sequential estimate, voting pipeline not separately timed')
    result['finished_at'] = now()
    write_csv(output/'summary.csv', result['summary'])
    write_csv(output/'predictions.csv', predictions)
    write_json(output/'errors.json', result['errors'])
    write_json(output/'results.json', result)
    confusion_svg(result['summary'], output)
    markdown_report(result, output)
    print(f'Finished: {output.resolve()}; predictions={len(predictions)}; errors={len(result["errors"])}', flush=True)
    return 0 if all(result['models'][name]['status'] == 'ok' for name in MODELS) else 1


if __name__ == '__main__':
    raise SystemExit(main())
