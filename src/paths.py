"""Every location the code reads or writes. Both folders below are local only (git-ignored): they hold photos of
real people and team members' names.

- data/: the final dataset and nothing else: images/<image_id>.jpg and pairs.jsonl (one line per pair of edits of
  a ranked sample). An image_id is "<sample id>/<key>", key "original" or an edit key ("01", "02", ...).
- work/: everything used to build it and every later output: the full sample list, provenance, the ranking
  versions (manifests, every annotator's results, merged labels, final rankings), images of samples without a
  final ranking (unranked/), and stage outputs (filters, instructions, scorer, models).

Environment variables let the same code run elsewhere, e.g. on Kaggle:
- EJ_DATA_DIR: the data folder to read (default: data/ in the repository); may be read-only;
- EJ_WORK_DIR: the work folder (default: work/ in the repository); not needed on Kaggle;
- EJ_OUTPUT_DIR: where new results are written (default: the work folder), e.g. /kaggle/working/out.
"""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = REPO_ROOT / ".env"                 # secrets and per-user settings, never committed

DATA_DIR = Path(os.environ.get("EJ_DATA_DIR") or REPO_ROOT / "data")
WORK_DIR = Path(os.environ.get("EJ_WORK_DIR") or REPO_ROOT / "work")
OUTPUT_DIR = Path(os.environ.get("EJ_OUTPUT_DIR") or WORK_DIR)

# Stage 1: dataset and ranking
FINAL_PAIRS = DATA_DIR / "pairs.jsonl"        # the final dataset: one line per pair of edits
SAMPLES_DIR = DATA_DIR / "images"             # images of the ranked samples: <sample id>/<key>.jpg
SAMPLES_JSONL = WORK_DIR / "samples.jsonl"    # every built sample (17,755): request, file names, editor notes
UNRANKED_DIR = WORK_DIR / "unranked"          # samples without a final ranking (original.jpg, edits/<key>.jpg)
RAW_DIR = WORK_DIR / "raw"                    # crawl outputs and published datasets (crawling is closed)
INDEX_DIR = WORK_DIR / "index"                # provenance, build ledger and image cache
PROVENANCE = INDEX_DIR / "provenance.jsonl"   # sample ID -> source, keys, raw paths, trainable flag
LEDGER = INDEX_DIR / "ledger.jsonl"           # every build decision (kept / rejected and why)
RANKING_DIR = WORK_DIR / "ranking"            # ranking versions (manifest, assignments, results, labels)
RANKING_DIST = RANKING_DIR / "dist"           # zip files handed out to annotators
RANKING_APP = REPO_ROOT / "apps" / "ranking"  # the standalone app annotators run

# Later stages: results of each task
FILTERS_DIR = OUTPUT_DIR / "filters"          # detector scores and decisions: filters/<task>/<model>.jsonl
INSTRUCTIONS_DIR = OUTPUT_DIR / "instructions"  # LLM/VLM analyses of the requests
SCORER_DIR = OUTPUT_DIR / "scorer"            # scorer datasets, runs and reports
MODELS_DIR = Path(os.environ.get("EJ_MODELS_DIR") or OUTPUT_DIR / "models")  # downloaded model weights

# Paths stored in crawl databases, the ledger and the provenance index ("raw/<source>/...") were relative to the
# old data folder; crawling and building are closed, so they are kept only as a record.
