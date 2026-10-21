"""Pinned, allowlisted Hugging Face downloads and strict dataset validation."""
import csv
import hashlib
import json
import time
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download, snapshot_download
from PIL import Image

DATA_REPO = 'boomb0om/watermarks-validation'
MODELS = {
    'yolov8n': ('qfisch/yolov8n-watermark-detection', 'best.pt'),
    'efficientnetv2_b0': ('qwertyforce/watermark_detection', 'model.pt'),
    'convnext_tiny': ('boomb0om/watermark-detectors', 'convnext-tiny_watermarks_detector.pth'),
}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    temp.replace(path)


def pinned_info(repo, kind, lock_path):
    """Reuse the first resolved commit on subsequent runs; never silently update."""
    lock_path = Path(lock_path)
    locks = json.loads(lock_path.read_text(encoding='utf-8')) if lock_path.exists() else {}
    key = f'{kind}/{repo}'
    rev = locks.get(key)
    info = HfApi().repo_info(repo, repo_type=kind, revision=rev, files_metadata=True)
    if rev and info.sha != rev:
        raise ValueError('Revision mismatch')
    locks[key] = info.sha
    write_json(lock_path, locks)
    return info


def validate_manifest(root, expected_n=122, expected_positive=62):
    root = Path(root).resolve()
    rows = []
    seen = set()
    with (root / 'validation_data.csv').open(encoding='utf-8-sig', newline='') as f:
        for row in csv.DictReader(f):
            rel = row['path'].replace('\\', '/')
            label = row['watermark'].strip().lower()
            if label not in ('true', 'false'):
                raise ValueError(f'Invalid label: {row}')
            path = (root / rel).resolve()
            if not path.is_relative_to(root) or rel in seen:
                raise ValueError(f'Unsafe/duplicate path: {rel}')
            if path.parts[len(root.parts)] not in ('clean', 'watermark'):
                raise ValueError(f'Unexpected directory: {rel}')
            seen.add(rel)
            with Image.open(path) as im:
                im.load()
                width, height = im.size
            rows.append(dict(id=f'public_122/{rel}', dataset='public_122', path=str(path),
                             relative_path=rel, true_label=int(label == 'true'),
                             sha256=sha256(path), bytes=path.stat().st_size, width=width, height=height))
    if len(rows) != expected_n or sum(r['true_label'] for r in rows) != expected_positive:
        raise ValueError(f'Expected {expected_n}/{expected_positive}; got {len(rows)}/{sum(r["true_label"] for r in rows)}')
    actual = {p.resolve() for folder in ('clean', 'watermark') for p in (root / folder).rglob('*') if p.is_file()}
    if actual != {Path(r['path']) for r in rows}:
        raise ValueError('Unlisted or missing image files')
    return rows


def download_dataset(data_root, lock_path, limit_mb=25):
    info = pinned_info(DATA_REPO, 'dataset', lock_path)
    allowed = [f for f in info.siblings if f.rfilename.startswith(('clean/', 'watermark/')) or f.rfilename == 'validation_data.csv']
    advertised = sum(f.size or 0 for f in allowed)
    if advertised > limit_mb * 1_000_000:
        raise ValueError(f'Remote dataset exceeds image budget: {advertised}')
    start = time.perf_counter()
    root = Path(data_root) / 'public_122'
    snapshot_download(repo_id=DATA_REPO, repo_type='dataset', revision=info.sha,
                      allow_patterns=['clean/**', 'watermark/**', 'validation_data.csv'],
                      local_dir=root, max_workers=2, token=False)
    rows = validate_manifest(root)
    image_bytes = sum(r['bytes'] for r in rows)
    if image_bytes > limit_mb * 1_000_000:
        raise ValueError('Downloaded images exceed budget')
    write_json(Path(data_root) / 'public_manifest.json', rows)
    return rows, dict(repo=DATA_REPO, revision=info.sha, image_bytes=image_bytes,
                      csv_sha256=sha256(root / 'validation_data.csv'), download_or_cache_s=time.perf_counter()-start)


def download_checkpoint(name, root, lock_path):
    repo, filename = MODELS[name]
    info = pinned_info(repo, 'model', lock_path)
    meta = next(f for f in info.siblings if f.rfilename == filename)
    start = time.perf_counter()
    path = Path(hf_hub_download(repo_id=repo, filename=filename, revision=info.sha,
                               local_dir=Path(root)/name, token=False))
    digest = sha256(path)
    expected = getattr(meta.lfs, 'sha256', None) if meta.lfs else None
    if expected and digest != expected:
        raise ValueError(f'Checkpoint SHA-256 mismatch: {name}')
    return path, dict(repo=repo, filename=filename, revision=info.sha, sha256=digest,
                      remote_lfs_sha256=expected, bytes=path.stat().st_size,
                      download_or_cache_s=time.perf_counter()-start)
