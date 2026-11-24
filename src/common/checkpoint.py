"""Explicit, private HF checkpoints of derived outputs; never dataset images or credentials."""

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path, PurePosixPath

ALLOWED_ROOTS = {'review', 'scorer', 'coordinator', 'recovery'}
EXCLUDED = {'hf-cache', 'models', 'images', 'samples', 'unranked', 'dist', 'logs', '__pycache__'}
EXTENSIONS = {'.json', '.jsonl', '.npy', '.npz', '.pt', '.pth', '.md', '.html', '.toml'}


def safe_path(name):
    path = PurePosixPath(name)
    return (not path.is_absolute() and bool(path.parts) and path.parts[0] in ALLOWED_ROOTS
            and not any(p in EXCLUDED or p.startswith('.') or '\\' in p or ':' in p for p in path.parts)
            and path.suffix in EXTENSIONS)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


class Backup:
    def __init__(self, root, repo, prefix, api=None):
        from functools import partial
        from huggingface_hub import HfApi, hf_hub_download
        self.root = Path(root)
        self.repo = repo
        self.prefix = prefix.strip('/')
        if not self.prefix or any(p in {'.', '..'} for p in self.prefix.split('/')):
            raise ValueError('Invalid checkpoint prefix')
        self.api = api or HfApi(token=os.environ.get('HF_TOKEN'))
        self.download = partial(hf_hub_download, token=getattr(self.api, 'token', None))
        if not self.api.repo_info(repo, repo_type='dataset').private:
            raise ValueError('Checkpoint repository must be private')
        self.last_hash = None

    def sync(self, stage):
        # Caller invokes this only at a completed command/seed boundary, when outputs are stable.
        with tempfile.TemporaryDirectory(prefix='ej-checkpoint-') as temporary:
            snapshot = Path(temporary)
            files = {}
            for path in sorted(self.root.rglob('*')):
                relative = path.relative_to(self.root).as_posix()
                if not path.is_file() or path.is_symlink() or not safe_path(relative):
                    continue
                target = snapshot / 'files' / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, target)
                digest = sha256(target)
                if digest != sha256(path):
                    raise RuntimeError('Output changed during checkpoint; retry at a stable boundary')
                files[relative] = {'sha256': digest, 'bytes': target.stat().st_size}
            manifest = {'schema': 1, 'stage': stage, 'files': files}
            encoded = json.dumps(manifest, sort_keys=True).encode()
            digest = hashlib.sha256(encoded).hexdigest()
            if digest == self.last_hash:
                return {'stage': stage, 'files': len(files), 'unchanged': True}
            (snapshot / 'checkpoint-manifest.json').write_bytes(encoded)
            result = self.api.upload_folder(repo_id=self.repo, repo_type='dataset', folder_path=str(snapshot),
                                           path_in_repo=self.prefix, commit_message=f'T-045 checkpoint: {stage}')
            # A revision pins manifest and files to the same atomic HF commit.
            remote = self.download(self.repo, self.prefix + '/checkpoint-manifest.json',
                                              repo_type='dataset', revision=result.oid)
            if Path(remote).read_bytes() != encoded:
                raise RuntimeError('Remote checkpoint verification failed')
            self.last_hash = digest
            status = {'stage': stage, 'files': len(files), 'bytes': sum(v['bytes'] for v in files.values()),
                      'revision': result.oid, 'repo': self.repo, 'prefix': self.prefix}
            (self.root / 'backup-status.json').write_text(json.dumps(status, indent=2), encoding='utf-8')
            print('BACKUP_VERIFIED ' + json.dumps(status), flush=True)
            return status

    def restore(self):
        from huggingface_hub.errors import EntryNotFoundError
        revision = self.api.repo_info(self.repo, repo_type='dataset').sha
        try:
            path = self.download(self.repo, self.prefix + '/checkpoint-manifest.json',
                                           repo_type='dataset', revision=revision)
        except EntryNotFoundError:
            return {'restored': 0, 'checkpoint_exists': False}
        manifest = json.loads(Path(path).read_text(encoding='utf-8'))
        if manifest.get('schema') != 1 or not isinstance(manifest.get('files'), dict):
            raise ValueError('Unsupported checkpoint manifest')
        # Verify every file before changing any current output.
        ready = []
        for name, expected in manifest['files'].items():
            if not safe_path(name):
                raise ValueError('Unsafe checkpoint path')
            cached = Path(self.download(self.repo, self.prefix + '/files/' + name,
                                                  repo_type='dataset', revision=revision))
            if cached.stat().st_size != expected['bytes'] or sha256(cached) != expected['sha256']:
                raise ValueError('Checkpoint content verification failed')
            target = self.root / name
            if target.exists() and sha256(target) != expected['sha256']:
                raise ValueError('Restore refused to overwrite a different current output')
            ready.append((cached, target))
        for cached, target in ready:
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name('.' + target.name + '.restore.tmp')
            shutil.copyfile(cached, temporary)
            os.replace(temporary, target)
        status = {'restored': len(ready), 'checkpoint_exists': True, 'revision': revision,
                  'stage': manifest['stage']}
        print('BACKUP_RESTORED ' + json.dumps(status), flush=True)
        return status


def configured_backup(root=None):
    repo, prefix = os.environ.get('EJ_BACKUP_REPO'), os.environ.get('EJ_BACKUP_PREFIX')
    if not repo and not prefix:
        return None
    if not repo or not prefix:
        raise ValueError('Set both EJ_BACKUP_REPO and EJ_BACKUP_PREFIX')
    if root is None:
        from ..paths import OUTPUT_DIR
        root = OUTPUT_DIR
    return Backup(root, repo, prefix)


def checkpoint(stage):
    backup = configured_backup()
    return backup.sync(stage) if backup else None
