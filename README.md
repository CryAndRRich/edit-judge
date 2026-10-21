# EditJudge

**Learning to judge photo edits from the crowd.** On photo-editing communities, people post a photo with an edit
request and other members answer with their edits. EditJudge collects these requests and edits, has a team rank the
edits of every request, and uses the rankings to learn what makes an edit good (Introduction to Data Science,
IT4142E, class 176080).

The work is split into stages (the plan follows the course project proposal):

| Stage | Folder | Status |
|---|---|---|
| 1. Data preparation: crawl, build and normalize the dataset, team ranking | `src/crawl`, `src/process`, `src/rank`, `apps/ranking` | Done (v1 ranked) |
| 2. Filtering: sensitive images and watermarks, each by several detectors and a vote | `src/filters` | Framework ready |
| 2. Request analysis with open VLMs (RealEdit's prompts): simplify and categorise the requests | `src/instructions` | Smoke test run on Kaggle |
| 3. Features and the learned scorer | `src/features`, `src/scorer` | Architecture proposed (`docs/scorer-survey.md`); pilot ready to run on Kaggle |
| 4. Editing models per operation, and their benchmark | `src/editors` | Planned |
| 5. The editing assistant | `src/assistant` | Planned |

## Repository layout

```
src/                        Python package (run as `python -m src.<module>`)
  paths.py                  every data location, in one place (EJ_DATA_DIR / EJ_OUTPUT_DIR override it)
  common/                   shared helpers: JSONL files, dataset images, device and the no-GPU guard, .env
  crawl/                    crawler framework: polite HTTP, SQLite state, one module per source
  process/                  dataset build (adapters, filter, writer), streaming collection, image normalization
  rank/                     ranking chunks and assignments, result checking, final rankings
  filters/                  detector interface and runner, voting, evaluation
    nsfw/                   sensitive-content detectors, one file each
    watermark/              watermark detectors, one file each
  instructions/             request analysis (RealEdit): label sets, prompts/ (versioned), clients/ (VLMs), runner
  features/                 per-edit features for the scorer
  scorer/                   preference pairs and splits (data.py), scorer models and evaluation
  editors/                  editing-model wrappers and benchmark
  assistant/                the end-to-end assistant
apps/ranking/               Gradio app the annotators run on their own computers (handed out as a zip)
configs/                    experiment settings (TOML), e.g. filters.toml, scorer.toml
docs/                       design notes and surveys, e.g. scorer-survey.md
notebooks/                  Kaggle notebooks; kaggle_template.ipynb to start from
tests/                      unit tests on a tiny synthetic dataset (never on data/)
data/                       local only, git-ignored (photos of real people): see "Data"
requirements.txt            dependencies of stage 1
requirements-ml.txt         model libraries for the later stages
.env.example                settings and secrets; copy to .env (git-ignored)
```

## Setup

Python 3.10 or newer.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env        # then fill in what you need; never commit .env
.venv/bin/python -m unittest discover -s tests -t .
```

All commands run from the repository root, e.g. `.venv/bin/python -m src.rank --help`.

Model libraries (`requirements-ml.txt`) are for Kaggle, where most of them are preinstalled. Locally they are only
for small smoke tests; on an Intel Mac use Python 3.11 with `torch==2.2.2`, `torchvision==0.17.2`, `numpy<2` and
`transformers<5` (newer versions have no wheels there).

## Running on Kaggle

Heavy work (detectors over the whole dataset, LLM/VLM inference, features, editing models) runs on Kaggle with a
GPU, never on a laptop. Without a GPU, model runners refuse more than 20 items (`src/common/device.py`; override
with `--allow-cpu` only for an approved run).

- The data goes to Kaggle as a **private** dataset shared only with the team (it contains photos of real people):
  `python -m src.process pack` writes `data/exports/editjudge-data.zip` (about 10 GB: samples, provenance, ranking
  manifests and final rankings; no raw data, no annotator names). Upload it as the dataset `editjudge-data`.
- Start from `notebooks/kaggle_template.ipynb`: it clones the code from GitHub (the branch you set, so push first),
  sets `EJ_DATA_DIR` to the read-only input and `EJ_OUTPUT_DIR` to `/kaggle/working/out`, so the code runs unchanged
  and every result lands in one output folder. Turn on the GPU and internet in the notebook settings.
- Bring the results back into `data/` (same sub-folders) and record the run (model, version, time) in the pull
  request. Do not commit outputs.

## Working as a team

Every task has its own folder, so members can work in parallel and merge with few conflicts.

- Branch from `curr`, one branch per task and person: `<task>/<name>`, e.g. `filters-nsfw/minh`.
- Work inside the task's folder (for example `src/filters/nsfw/` plus its entry in `nsfw/__init__.py`). Changes to
  shared files (`src/paths.py`, `src/common/`, `requirements*.txt`, `README.md`) go in a separate small commit and
  are announced to the coordinator.
- Keep the shared formats: results keyed by `image_id` ("<sample id>/<key>") or sample ID, JSONL files under the
  task's data folder, settings in `configs/`.
- Add tests for new logic in `tests/` (synthetic data only), and run them before asking for a merge.
- Never commit `data/`, model weights, notebook outputs, or `.env`.

## Data

Everything in `data/` is local only: it contains photos of real people and text written by real users. Do not
commit, upload or share it outside the team.

```
data/
  samples.jsonl                     the dataset: one line per sample (sample.json plus "dir")
  samples/<id>/sample.json          {"id", "request", "original", "edits": [{"image", "note"}]}
  samples/<id>/original.jpg
  samples/<id>/edits/01.jpg, 02.jpg, ...
  index/provenance.jsonl            sample ID -> source, source keys, raw paths, "trainable"
  index/ledger.jsonl                every build decision (kept or rejected, and why)
  index/image_cache.json            image header and hash cache of the build
  raw/<source>/                     crawl outputs and published datasets (deleted once the dataset was frozen)
  raw/.salt                         salt for hashing usernames
  ranking/v1/, v2/                  ranking versions: manifest, assignments, results, labels, final rankings
  ranking/dist/                     zip files handed out to annotators, and collected result zips
  labels/<task>.jsonl               hand labels used to evaluate a task (e.g. nsfw, watermark)
  filters/, instructions/, features/, scorer/, editors/
                                    results of the later stages, one folder per task
  models/                           downloaded model weights
  exports/editjudge-data.zip        the data packed for Kaggle (`python -m src.process pack`)
  logs/                             run logs
```

- `request` is the edit request as written (title and first post for forum threads), not translated or cleaned.
- `note` is the text that came with an edit (the editor's comment, or a description of the change); may be null.
- Samples carry no source information. `index/provenance.jsonl` maps them back to their source, for licences,
  takedown requests, de-duplication and the VOZ no-training rule (`trainable: false`).
- Paths stored in crawl databases, the ledger and the provenance index (`raw/<source>/...`) are relative to `data/`.

**Current dataset (frozen on 2026-10-07):** 17,755 samples and 44,800 edits (62,555 images). RealEdit 16,422,
GIER 1,264, photoshopcontest 39, dprevived 26, mineo 3, PhotoshopGurus 1. Edits per sample: 2 (11,457),
3 (4,103), 4 (1,549), 5 (607), 6 to 14 (39). Data collection is closed. All images are normalized (see
"Image normalization"): 62,555 JPEG files, 9.5 GB.

## Crawling (`src.crawl`)

One framework for every source. All sources share the polite HTTP layer (pacing, robots.txt, refusal handling),
the resumable SQLite store, the image downloader and the JSONL export. A source module only knows how to
**discover** request threads on its site and how to **parse** them.

| Source | Module | Status |
|---|---|---|
| RealEdit (CVPR 2025, CC BY 4.0): r/PhotoshopRequest and r/estoration posts as image URLs | `sources/realedit.py` | Done: all 22,221 posts with at least 2 edits. Edits are ordered by upvotes |
| dprevived.com "Edit Me An Image" (Misago) | `sources/dprevived.py` | Done |
| mineo (Japan) | `sources/mineo.py` | Done; requests and edits share one thread and are paired by rules |
| fotokto.ru (Russia, phpBB) | `sources/fotokto.py` | Done (small yield) |
| VOZ (Vietnam, XenForo) | `sources/voz.py` | Done; robots.txt says `ai-train=no`: never used for training |
| photoshopcontest.com | `sources/photoshopcontest.py` | Tried on 2026-10-07 (39 samples), stopped by decision |
| b3ta.com image challenges | `sources/b3ta.py` | Tried; no usable samples (composites, dead image hosts) |
| Zhopped via the Wayback Machine | `sources/zhopped.py` | Mostly thumbnails; archive.org rate limits |
| Reddit API | `sources/reddit.py` | Blocked: API use needs prior approval since 2025-11-11. Refuses to run without credentials |
| PhotoshopGurus (XenForo) | `sources/photoshopgurus.py` | **Suspended**: the team account was banned after a logged-in crawl. Refuses to run unless `PSG_PERMISSION=granted`, which may only be set after a forum admin allows it |

GIER (ACCV 2020) is read from its published files by an adapter, not crawled.

```bash
.venv/bin/python -m src.crawl probe                  # check access to every source, save a report
.venv/bin/python -m src.crawl status                 # progress counters for every source
.venv/bin/python -m src.crawl dprevived discover     # find request threads
.venv/bin/python -m src.crawl dprevived threads      # fetch and parse them
.venv/bin/python -m src.crawl dprevived images --min-edit-images 2 --max-minutes 30
.venv/bin/python -m src.crawl dprevived export       # posts.jsonl and edits.jsonl
```

Options: `--section S`, `--first N --last M` (listing pages), `--max-pages K`, `--limit N`, `--refetch`,
`--min-edit-images K`, `--max-minutes M`, `--retry-thumbnails`, `--delay S`. Every stage is resumable: interrupt
with Ctrl-C and run the same command again. `restatus` and `reparse` work offline after a rule or parser change.

**Roles.** Each post gets a role when its thread is fetched: `request` (first post), `requester_reply` (a later post
by the same author), `edit` (another user's post with images) or `comment`. In mineo, image comments get the
neutral role `image_post` and are paired later.

**Safety rules (enforced in code).**
- Pacing: one request stream per source, 3 s between requests by default (5 s for archive.org).
- robots.txt is checked before every page request; a disallowed URL is skipped, never fetched.
- HTTP 401/403 or a bot-challenge page stops the run (exit code 2). Blocks are never bypassed.
- HTTP 429 and 5xx are retried a bounded number of times, honouring `Retry-After`.
- Suspended sources and sources without credentials refuse to run (exit code 3).
- Usernames are stored only as salted hashes. The raw page cache still has usernames and stays local.

**Output** (`data/raw/<source>/`): `crawl.sqlite` (state and parsed records), `raw/` (gzip page cache, so
parsers can be fixed without crawling again), `images/`, and `export/`.

**Adding a source:** subclass `Source` (or `XenForoSource`) in `sources/<name>.py`, set `name`, `label` and
`base_url`, implement `discover()`, `fetch_thread()` and `probe_targets()`, register the class in
`sources/__init__.py`, and run `probe` before crawling.

## Building the dataset (`src.process`)

```bash
.venv/bin/python -m src.process build --dry-run     # filter and print statistics only
.venv/bin/python -m src.process build               # add new samples to data/samples
.venv/bin/python -m src.process prune --source S     # delete raw files of decided candidates
.venv/bin/python -m src.process cycle --source realedit --max-minutes 60   # streaming collection
.venv/bin/python -m src.process normalize           # normalize every image (see below)
.venv/bin/python -m src.process pack                # zip the data for a private Kaggle dataset
```

1. **Adapters** (`process/adapters/`) turn each kind of raw data into candidates with the same fields: request
   text, original image(s), and edits (image and note). `crawled.py` reads every crawl database, `gier.py` reads
   GIER, and `mineo.py` pairs requests and edits in the shared mineo thread (`raw/mineo/pairing.csv`, corrected
   by hand).
2. **Filter** (`process/build.py`), the same for every source:
   - the request is not empty;
   - exactly one original, present and readable;
   - every image has a shorter side of at least 256 px (as downloaded);
   - edits identical to the original, and duplicate edits, are dropped (SHA-256);
   - at least 2 edits remain;
   - an original found in several sources is kept once.
3. **Writer:** `data/samples/<id>/` and `data/samples.jsonl`. Sample IDs are stable hashes of the source key.
   Builds are incremental: every decision goes to the ledger, so raw files can be deleted after a build (`prune`);
   the dataset holds hard links to them.

**Streaming collection (`cycle`)** was used for RealEdit with little disk space: each round downloads for
`--round-minutes`, builds, and prunes the raw files, until `--max-minutes` or `--min-free-gb` is reached.

### Image normalization (`normalize`)

Community images come in many formats and sizes (JPEG, PNG, WebP, GIF; up to 9,360 px; CMYK, colour profiles,
EXIF rotation, transparency). `normalize` rewrites every image of `data/samples` in place so that all of them are:

- the first frame only (GIF, animated WebP);
- rotated by their EXIF orientation;
- converted to sRGB when they embed a colour profile, then to 8-bit RGB, with transparent areas on white;
- at most 1,024 px on the longest side (`--max-side`; never enlarged);
- JPEG quality 92 (`--quality`) with no metadata: EXIF (including GPS), colour profile, XMP and comments are removed.

A file keeps its name with the `.jpg` extension (`edits/02.png` becomes `edits/02.jpg`), and `sample.json`,
`samples.jsonl` and the provenance index are updated. Edit keys (the file stems used by the ranking) do not change.
Images that already meet every rule are left untouched, so the command can be stopped and run again.

Run on 2026-10-13 (about 1 hour with 3 processes): about 54,700 images converted (all PNG, WebP and
GIF files and most JPEGs), the others already met the rules; no errors; the samples went from 47 GB to 9.5 GB. After resizing,
112 very wide or tall images have a shorter side below 256 px (the build filter applies to the downloaded size).

## Team ranking (`src.rank`, `apps/ranking/`)

Each sample is ranked by 3 of the 9 team members, from best to worst edit, with no ties. Samples that cannot be
ranked fairly are skipped with a reason.

```bash
.venv/bin/python -m src.rank plan --version v1      # freeze a version: chunks and assignments
.venv/bin/python -m src.rank zip --version v1       # chunk zips -> data/ranking/dist/
.venv/bin/python -m src.rank app                    # data/ranking/dist/ranking_app_v2.zip
.venv/bin/python -m src.rank results --name Hai     # zip one annotator's results
.venv/bin/python -m src.rank merge --version v1     # check results, agreement -> v1/labels/
.venv/bin/python -m src.rank finalize --version v1  # final rankings -> v1/final/
```

**Workflow.** Map the slots R1-R9 to team members and send each person the app zip and the 13 chunk zips of
their slot, only within the team. Annotators work in pass order (all pass-1 chunks first, so every sample gets one
ranking early). Collect `results/<chunk>__<name>.json` from everyone into `data/ranking/<version>/results/<slot>/`,
then run `merge` and `finalize`.

**Design.**
- Chunks: samples are sorted by number of edits and dealt in snake order, so every chunk has a similar workload.
- Assignment: each chunk goes to 3 annotators; every annotator gets 13 of the 39 chunks, and every pair of
  annotators shares 3-4 chunks, so agreement can be measured for every pair. Each chunk's annotators are ordered
  as pass 1, 2 and 3.
- Versions: v1 holds the first 7,621 samples (39 chunks of 195-196). v2 holds the other 10,134 samples (39 chunks
  of 259-260, same assignment as v1) and has not been ranked yet. A new version only takes samples that no earlier
  manifest contains.

**The app** (`apps/ranking/`) runs on each annotator's computer: unzip, put the chunk zips into `chunks/`, and
double-click `run_windows.bat` or `run_mac.command` (the first start installs Gradio into `.venv/`). Clicking an
edit gives it the next rank, clicking it again removes the rank; each rank has its own frame colour. Work is saved
after every sample to `results/<chunk>__<name>.json`:

```json
{"app_version": "1.3", "version": "v1", "chunk": "v1_c07", "annotator": "Hai", "updated_at": "...",
 "n_samples": 195, "n_ranked": 180, "n_skipped": 15,
 "records": {"<sample id>": {"ranking": ["02", "01", "03"], "skipped": false, "skip_reason": null,
                             "seconds": 21.4, "saved_at": "..."}}}
```

`ranking` lists edit keys from best to worst; a skipped sample has `"ranking": null` and a reason.

### Final ranking

`finalize` turns the 3 rankings of a sample (and, for RealEdit, the Reddit upvote order) into one ranking.

- **Filter:** a sample that at least one member skipped is dropped.
- **Voters:** every annotator slot is one voter; the time spent on a sample plays no role.
- **Reliability:** one-coin model. Each voter picks the better edit of a pair with its own probability p,
  independently of the others, so two voters agree with probability p_a p_b + (1 - p_a)(1 - p_b). The p values are
  fitted to the observed agreement of every pair of voters. The vote weight is log(p / (1 - p)), so a voter at chance
  level gets no weight.
- **Upvote reference:** for RealEdit samples, the upvote order is one more vote with a fixed weight of half the
  median member weight (`--community-weight 0.5`). It is not estimated from the data, because the app shows edit 01
  (the most upvoted) first: in GIER, where 01 has no meaning, members still put 01 first in 56% of rankings against
  44% by chance.
- **Pairs:** P(x better than y) = sigmoid(sum of the weights of voters preferring x minus those preferring y).
- **Final order:** the order with the highest sum of log P over all its pairs, found exactly by dynamic programming
  over subsets of edits. Ties go to the upvote order, then to the edit key.

Outputs in `data/ranking/v1/final/`: `rankings.jsonl` (per sample: votes, upvote order, final ranking,
`confidence` = mean P of adjacent pairs, status), `pairs.jsonl` (per pair: `p_a` and the winner; soft or hard
labels for a scorer) and `summary.json`.

**v1 result (2026-10-13):** 6,996 samples ranked (RealEdit 5,749, GIER 1,229, others 18), 625 dropped because
someone skipped them. Raw agreement between two members is 57.8% of pairs (Krippendorff's alpha 0.14). Estimated
voter accuracy is 0.57-0.73, except one slot at chance level (weight 0). The upvote order agrees with the team on
61% of pairs and changes the final ranking of 865 RealEdit samples (mostly 1-1 splits). Known limitation: edits
were shown in key order, so rankings carry a position bias; shuffle the display order before ranking v2.

## Filtering (`src.filters`)

Two tasks remove images before training: **sensitive content** (`nsfw`) and **watermarks** (`watermark`). For each
task several detectors run, and the evaluation decides whether a vote of several models beats the single best
model enough to pay for the extra time.

```bash
.venv/bin/python -m src.filters list                                  # registered detectors
.venv/bin/python -m src.filters run --task nsfw --model NAME          # on Kaggle; --limit 20 for a local test
.venv/bin/python -m src.filters vote --task nsfw                      # -> data/filters/nsfw/decisions.jsonl
.venv/bin/python -m src.filters evaluate --task nsfw                  # -> data/filters/nsfw/evaluation.json
```

- **A detector** subclasses `Detector` (`src/filters/base.py`): `load(device)` and `score(images)`, returning the
  probability that each image is positive. Put it in its own file in `nsfw/` or `watermark/` and register it in that
  folder's `__init__.py` (`DETECTORS`).
- **Scores** go to `data/filters/<task>/<model>.jsonl` (`{"image_id", "score"}`), written after every batch so a run
  can resume; `<model>.meta.json` records the device and the time per image.
- **Voting** (`configs/filters.toml`): `mean`, `weighted`, `max` (any model), or `majority` (share of models above
  their own threshold). Scores are combined without running the models again.
- **Evaluation** needs hand labels in `data/labels/<task>.jsonl` (`{"image_id", "label"}`): a few hundred images,
  mostly where the detectors disagree plus a random share. It reports precision, recall and F1 at the configured
  thresholds, the best F1 over thresholds, and the time per image.

## Request analysis (`src.instructions`)

Follows RealEdit (Sushko et al., CVPR 2025) directly. For each request, **one call to an open VLM** gets the request
text, the original image, and RealEdit's two prompts merged into one (`prompts/analyze_v<k>.txt`, both kept word
for word except for the JSON answer format; v2 adds that each category is one listed operation plus one listed
subject):

- **Simplified instruction** (RealEdit Fig. 19): URLs, typos, irrelevant details and thanks removed, the main task
  summarised; humorous or ambiguous requests become `humorous`.
- **Categories** (RealEdit Fig. 12): one or more "operation subject" pairs, one per task in the request.
  Operations: Add, Change-Color, Enhance, Image-Formatting, Open-Ended, Remove, Replace, Restore, Other. Subjects:
  Animal, Background, Body-Part, Clothing/Accessories, Drawing, Food, Logo, Manmade-Structure, Media, Nature, Person,
  Text/Patterns, Vehicle, Other. RealEdit's short names ("Add Clothing", "Change Color") are mapped to these.

```bash
.venv/bin/python -m src.instructions list                                   # clients and prompt versions
.venv/bin/python -m src.instructions run --client qwen2.5-vl-7b --sample 100  # on Kaggle: fixed subset for tuning
.venv/bin/python -m src.instructions run --client qwen2.5-vl-7b --max-hours 11   # on Kaggle: all requests
.venv/bin/python -m src.instructions run --client qwen2.5-vl-3b --shard 0/2   # one shard per GPU (CUDA_VISIBLE_DEVICES)
.venv/bin/python -m src.instructions stats --run qwen2.5-vl-7b:analyze_v2   # vs RealEdit's distribution
.venv/bin/python -m src.instructions agree --run A:analyze_v2 --run B:analyze_v2   # two models or prompts
```

- **Clients** (`clients/`): open VLMs with one interface (`generate_batch`); Qwen2.5-VL 3B and 7B are registered
  (bfloat16 on GPUs that support it, else float16). An answer from a batch that is not JSON (float16 can give
  "!!!!") is asked again alone. Start from `notebooks/instructions_qwen_vl.ipynb` on Kaggle (`STEP` picks the run).
- **Long runs:** `--shard I/N` runs every N-th request, so each GPU takes one shard (`<prompt>.part-I-of-N.jsonl`);
  `--max-hours` stops cleanly before Kaggle's 12 h limit, and the next session resumes from the copied results.
  Measured: qwen2.5-vl-3b, about 2.3 s per request on one T4, so all 17,755 requests take about 6 h on 2 x T4.
- **Results:** `data/instructions/<client>/<prompt>[.part-I-of-N].jsonl`, one row per request: `instruction`, `categories`
  (`[{"operation", "subject"}]`), the model's raw answer, and parsing `problems`. Runs resume where they stopped.
- **Prompts** are versioned files; a version used for a run is never edited (add `analyze_v2.txt` instead). The
  `--sample N` subset is fixed, so prompt versions and models are compared on the same requests.
- The raw request is always kept; simplified instructions are never used as a training signal.

## Scorer (`src.scorer`)

`src/scorer/data.py` loads the preference pairs of the final rankings (`load_pairs`), drops samples whose source
forbids training, and assigns every sample to train (70%), dev (15%) or test (15%) by a hash of its ID, so the
split never changes. v1 gives 15,287 pairs (train 10,598, dev 2,239, test 2,450).

The architecture (survey and proposal in `docs/scorer-survey.md`): frozen encoders (CLIP ViT-L/14, PickScore,
SigLIP, DINOv2) embed the original, the edit and the request once; a small head (linear or MLP) on features of
these embeddings gives one score per edit, trained on pairs of edits of the same request (Bradley-Terry on the soft
labels `p_a`, or an HPSv3-style uncertainty loss).

```bash
.venv/bin/python -m src.scorer embed --backbone clip-l14   # on Kaggle: cache embeddings of the ranked samples
.venv/bin/python -m src.scorer baselines                   # CLIP-score and other zero-shot baselines, human ceiling
.venv/bin/python -m src.scorer train                       # heads of configs/scorer.toml (minutes on a CPU)
.venv/bin/python -m src.scorer report                      # data/scorer/report.md: one table of everything
```

- **Runs** are listed in `configs/scorer.toml` (backbones, inputs, head, loss); each writes
  `data/scorer/runs/<name>/` (`metrics.json`, `scores.jsonl` per edit, `head.pt`). Model selection uses dev; test
  is reported once.
- **Metrics:** pairwise accuracy with a 95% bootstrap interval over samples, accuracy weighted by |2 p_a - 1| and on
  confident pairs, top-1 (best edit found), by source. The `edit-key-order` baseline (01 first) measures the
  position and upvote bias of the labels; it is not a valid scorer.
- Start from `notebooks/scorer_pilot.ipynb` on Kaggle (`STEP` picks the run).

## Data rules

- Respect robots.txt and each site's terms, and rate-limit every request. Never bypass a block.
- Reddit: every API use needs prior approval (Reddit for Researchers). Do not use RSS, old.reddit or other
  workarounds; use published datasets (RealEdit, PSR) instead.
- PhotoshopGurus: do not crawl, logged in or as a guest, until a forum admin gives explicit permission.
- Do not crawl zhopped.com (the domain now hosts a gambling site), Facebook or Threads (their terms forbid it).
- VOZ data must not be used to train or fine-tune models (`trainable: false` in the provenance index).
- Store image URLs instead of image files where possible, and never republish identifiable images of people.
- Keep API keys and cookies in `.env` (or Kaggle secrets), never in the code or in commits.
- Heavy model runs happen on Kaggle, never on a laptop, and only after the coordinator approves them. Data uploaded
  to Kaggle stays in private datasets shared with the team.

## License

The code is released under the MIT License (see `LICENSE`). The data is not distributed: source datasets and
community posts keep their own licences and terms.
