"""Every location the code reads or writes. All data lives under data/, which is local only (git-ignored):
it holds photos of real people.

Two environment variables let the same code run elsewhere, e.g. on Kaggle:
- EJ_DATA_DIR: the data folder to read (default: data/ in the repository); may be read-only;
- EJ_OUTPUT_DIR: where new results are written (default: the data folder), e.g. /kaggle/working/out.
"""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = REPO_ROOT / ".env"                 # secrets and per-user settings, never committed

DATA_DIR = Path(os.environ.get("EJ_DATA_DIR") or REPO_ROOT / "data")
OUTPUT_DIR = Path(os.environ.get("EJ_OUTPUT_DIR") or DATA_DIR)

# Stage 1: dataset and ranking
SAMPLES_DIR = DATA_DIR / "samples"            # the dataset: one folder per sample
SAMPLES_JSONL = DATA_DIR / "samples.jsonl"    # the dataset: one line per sample
RAW_DIR = DATA_DIR / "raw"                    # crawl outputs and published datasets, one folder per source
INDEX_DIR = DATA_DIR / "index"                # provenance, build ledger and image cache
PROVENANCE = INDEX_DIR / "provenance.jsonl"   # sample ID -> source, keys, raw paths, trainable flag
LEDGER = INDEX_DIR / "ledger.jsonl"           # every build decision (kept / rejected and why)
RANKING_DIR = DATA_DIR / "ranking"            # ranking versions (manifest, assignments, results, labels)
RANKING_DIST = RANKING_DIR / "dist"           # zip files handed out to annotators
RANKING_APP = REPO_ROOT / "apps" / "ranking"  # the standalone app annotators run

# Later stages: results of each task
LABELS_DIR = DATA_DIR / "labels"              # hand labels for evaluation: labels/<task>.jsonl
FILTERS_DIR = OUTPUT_DIR / "filters"          # detector scores and decisions: filters/<task>/<model>.jsonl
INSTRUCTIONS_DIR = OUTPUT_DIR / "instructions"  # LLM/VLM analyses of the requests
FEATURES_DIR = OUTPUT_DIR / "features"        # per-edit features for the scorer
SCORER_DIR = OUTPUT_DIR / "scorer"            # scorer datasets, runs and reports
EDITORS_DIR = OUTPUT_DIR / "editors"          # outputs and benchmark of editing models
MODELS_DIR = Path(os.environ.get("EJ_MODELS_DIR") or OUTPUT_DIR / "models")  # downloaded model weights

# Paths stored in crawl databases, the ledger and the provenance index ("raw/<source>/...") are relative
# to DATA_DIR, so the data folder can move between machines.
