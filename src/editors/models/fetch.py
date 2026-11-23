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
