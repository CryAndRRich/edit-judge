"""Downloads shared by the editors: a GitHub repository's code at a pinned commit (for models that ship code, not a
package), unpacked once into EJ_MODELS_DIR, as objectclear.py does for ObjectClear."""


def repository_code(repo: str, commit: str):
    """The code of gh:`repo` at `commit`, unpacked once into EJ_MODELS_DIR/<name>-<commit>; returns its folder."""
    import tarfile
    import urllib.request

    from ...paths import MODELS_DIR
    name = repo.split("/")[1]
    folder = MODELS_DIR / f"{name}-{commit}"
    if not folder.exists():
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        archive = MODELS_DIR / f"{name}-{commit[:7]}.tar.gz"
        urllib.request.urlretrieve(f"https://github.com/{repo}/archive/{commit}.tar.gz", archive)
        with tarfile.open(archive) as tar:
            try:
                tar.extractall(MODELS_DIR, filter="data")
            except TypeError:                         # Python before 3.11.4: no extraction filters
                tar.extractall(MODELS_DIR)
        archive.unlink()
    return folder


def pipeline_files(repo: str, revision: str | None = None, variant: str | None = None, **passed) -> None:
    """The files a diffusers pipeline's from_pretrained loads (DiffusionPipeline.download: the same filters; the
    components passed as keyword arguments, e.g. safety_checker=None, are not downloaded)."""
    from diffusers import DiffusionPipeline
    DiffusionPipeline.download(repo, revision=revision, variant=variant, **passed)


def release_file(url: str, folder, size: int):
    """`url` saved as folder/<its file name> (once; a file of the wrong size is fetched again)."""
    import urllib.request
    from pathlib import Path

    path = Path(folder) / url.rsplit("/", 1)[1]
    if not path.exists() or path.stat().st_size != size:
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_suffix(".part")
        urllib.request.urlretrieve(url, partial)
        if partial.stat().st_size != size:
            raise RuntimeError(f"{url}: got {partial.stat().st_size} bytes, expected {size}")
        partial.replace(path)
    return path
