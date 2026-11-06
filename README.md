# EditJudge

**Learning to judge photo edits from the crowd.** On photo-editing communities, people post a photo with an edit
request and other members answer with their edits. EditJudge collects these requests and edits, has a team rank the
edits of every request, and uses the rankings to learn what makes an edit good (Introduction to Data Science,
IT4142E, class 176080).

The work is split into stages (the plan follows the course project proposal):

| Stage | Folder | Status |
|---|---|---|
| 1. Data preparation: crawl, build and normalize the dataset, team ranking | `src/crawl`, `src/process`, `src/rank`, `apps/ranking` | Done (v1 and v2 ranked, final dataset exported) |
| 2. Filtering: sensitive images and watermarks, each by several detectors and a vote | `src/filters` | 6 NSFW and 3 watermark detectors benchmarked on public sets (`docs/filters-benchmark.md`); not yet run on the dataset |
| 2. Request analysis with open VLMs (RealEdit's prompts): simplify and categorise the requests | `src/instructions` | Tested on Kaggle (100 requests, 3B and 7B) |
| 3. Features and the learned scorer | `src/scorer` | Architecture proposed (`docs/scorer-survey.md`); pilot on v1: 60.4% test pair accuracy vs 50.6% for CLIP-score |
| 4. Editing models per operation, and their benchmark | (no code yet) | Planned |
| 5. The editing assistant | (no code yet) | Planned |

## Repository layout

```
src/                        Python package (run as `python -m src.<module>`)
  paths.py                  every data location, in one place (EJ_DATA_DIR / EJ_WORK_DIR / EJ_OUTPUT_DIR override it)
  common/                   shared helpers: JSONL files, dataset images, device and the no-GPU guard, .env
  crawl/                    crawler framework: polite HTTP, SQLite state, one module per source
  process/                  dataset build (adapters, filter, writer), streaming collection, image normalization
  rank/                     ranking chunks and assignments, result checking, final rankings, export of data/
  filters/                  detector interface and runner, public benchmark sets, voting, evaluation
    nsfw/                   sensitive-content detectors, one file each (6)
    watermark/              watermark detectors, one file each (3), and the safe checkpoint loader
  instructions/             request analysis (RealEdit): label sets, prompts/ (versioned), clients/ (VLMs), runner
  scorer/                   preference pairs and splits (data.py), backbones, embeddings, heads, training, evaluation
apps/ranking/               Gradio app the annotators run on their own computers (handed out as a zip)
configs/                    experiment settings (TOML), e.g. filters.toml, scorer.toml
docs/                       design notes, surveys and results, e.g. scorer-survey.md, filters-benchmark.md
notebooks/                  Kaggle notebooks; kaggle_template.ipynb to start from
tests/                      unit tests on a tiny synthetic dataset (never on data/)
data/                       local only, git-ignored: the final dataset (images/, pairs.jsonl); see "Data"
work/                       local only, git-ignored: everything used to build data/, and stage outputs
requirements.txt            dependencies of stage 1
requirements-ml.txt         model libraries for the later stages
requirements-filters.txt    extra packages of the image filter detectors (installed one model at a time on Kaggle)
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
  `python -m src.process pack` writes `work/exports/editjudge-data.zip` (9.1 GB: `data/` as it is, `images/` and
  `pairs.jsonl`). Upload it as a new version of the private dataset `cryandrrich/editjudge`; the notebooks find
  the folder that holds `pairs.jsonl` among their inputs.
- Start from `notebooks/kaggle_template.ipynb`: it clones the code from GitHub (the branch you set, so push first),
  sets `EJ_DATA_DIR` to the read-only input and `EJ_OUTPUT_DIR` to `/kaggle/working/out`, so the code runs unchanged
  and every result lands in one output folder. Turn on the GPU and internet in the notebook settings.
- Bring the results back into `work/` (same sub-folders) and record the run (model, version, time) in the pull
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
- Never commit `data/`, `work/`, model weights, notebook outputs, or `.env`.

## Data

Both folders are local only: they contain photos of real people, text written by real users, and team members'
names. Do not commit, upload or share them outside the team.

**`data/` holds the final dataset and nothing else** (since 2026-11-06), written by `python -m src.rank export`:

```
data/
  images/<sample id>/original.jpg, 01.jpg, 02.jpg, ...    the file of image_id "<sample id>/<key>"
  pairs.jsonl                                             one line per pair of edits of a ranked sample
```

```json
{"a": "00039bc7754c/01", "b": "00039bc7754c/02", "p": 0.6437, "request": "Can someone fix the exposure of this?"}
```

- `a`, `b`: image_ids of two edits of the same sample; the original is "<sample id>/original".
- `p`: probability that `a` is better than `b` (soft label), always at least 0.5, so `a` is the preferred edit (hard
  label). Lines are sorted by sample ID, then by the final ranking; the rank of an edit within its sample is the
  number of pairs it wins (116 of 36,198 pairs, all between non-adjacent edits, disagree with the full ranking).
- `request`: the edit request as written (title and first post for forum threads), not translated or cleaned.

**`work/` holds everything else:**

```
work/
  samples.jsonl                     every built sample (17,755): id, request, original, edits [{image, note}]
  unranked/<id>/                    samples without a final ranking: original.jpg, edits/<key>.jpg (moved back by
                                    `export` if ranked later)
  filtered/<id>/<key>.jpg           images removed by the image filters (flagged, or of a dropped sample)
  index/provenance.jsonl            sample ID -> source, source keys, raw paths, "trainable"
  index/ledger.jsonl                every build decision (kept or rejected, and why)
  index/image_cache.json            image header and hash cache of the build
  ranking/v1/, v2/                  ranking versions: manifest, assignments, results, labels, final rankings
  ranking/dist/                     zip files handed out to annotators, and collected result zips
  labels/<task>.jsonl               hand labels used to evaluate a task (e.g. nsfw, watermark)
  filters/, instructions/, features/, scorer/, editors/
                                    results of the later stages, one folder per task (EJ_OUTPUT_DIR)
  models/                           downloaded model weights
  exports/editjudge-data.zip        the data packed for Kaggle (`python -m src.process pack`)
  logs/                             run logs
```

- `note` is the text that came with an edit (the editor's comment, or a description of the change); may be null.
- Samples carry no source information. `index/provenance.jsonl` maps them back to their source, for licences,
  takedown requests, de-duplication and the VOZ no-training rule (`trainable: false`; the final dataset has no VOZ
  sample).
- The raw crawl outputs and the username salt were deleted (crawling is closed). Paths stored in the ledger and the
  provenance index (`raw/<source>/...`) are kept only as a record.

**Final dataset (2026-11-06):** 16,110 ranked samples (v1 6,996, v2 9,114), 56,568 images (40,458 edits), 36,198
pairs, 8.6 GB; packed as `work/exports/editjudge-data.zip` (9.1 GB).
The other 1,645 samples (skipped by a member, or without a stand-in for a missing member) are in `work/unranked/`.

**Built dataset (frozen on 2026-10-07):** 17,755 samples and 44,800 edits (62,555 images). RealEdit 16,422,
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

**Output** (`work/raw/<source>/`, deleted after the build): `crawl.sqlite` (state and parsed records), `raw/` (gzip page cache, so
parsers can be fixed without crawling again), `images/`, and `export/`.

**Adding a source:** subclass `Source` (or `XenForoSource`) in `sources/<name>.py`, set `name`, `label` and
`base_url`, implement `discover()`, `fetch_thread()` and `probe_targets()`, register the class in
`sources/__init__.py`, and run `probe` before crawling.

## Building the dataset (`src.process`)

Building and normalization are closed: these commands worked on the layout before 2026-11-06 (`data/samples/<id>/`
with `edits/`, `data/samples.jsonl`) and are kept as a record. Only `pack` is still used.

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
3. **Writer:** `data/samples/<id>/` and `data/samples.jsonl` (the layout before 2026-11-06; the list is now
   `work/samples.jsonl`, and `export` rewrote the images into `data/images/`). Sample IDs are stable hashes of the source key.
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
.venv/bin/python -m src.rank zip --version v1       # chunk zips -> work/ranking/dist/
.venv/bin/python -m src.rank app                    # work/ranking/dist/ranking_app_v2.zip
.venv/bin/python -m src.rank results --name Hai     # zip one annotator's results
.venv/bin/python -m src.rank merge --version v1     # check results, agreement -> v1/labels/
.venv/bin/python -m src.rank finalize --version v1 --stand-in upvote --min-accuracy 0.6 --hard-margin 0.1
.venv/bin/python -m src.rank finalize --version v2 --stand-in upvote --min-accuracy 0.6 --hard-margin 0.1
                                                    # final rankings -> <version>/final/ (cleaning: see below)
.venv/bin/python -m src.rank export                 # data/pairs.jsonl and data/images/ from every final/ and the filters
```

**Workflow.** Map the slots R1-R9 to team members and send each person the app zip and the 13 chunk zips of
their slot, only within the team. Annotators work in pass order (all pass-1 chunks first, so every sample gets one
ranking early). Collect `results/<chunk>__<name>.json` from everyone into `work/ranking/<version>/results/<slot>/`,
then run `merge`, `finalize` and `export`. (All ranking paths below are under `work/`.)

**Design.**
- Chunks: samples are sorted by number of edits and dealt in snake order, so every chunk has a similar workload.
- Assignment: each chunk goes to 3 annotators; every annotator gets 13 of the 39 chunks, and every pair of
  annotators shares 3-4 chunks, so agreement can be measured for every pair. Each chunk's annotators are ordered
  as pass 1, 2 and 3.
- Versions: v1 holds the first 7,621 samples (39 chunks of 195-196). v2 holds the other 10,134 samples (39 chunks
  of 259-260, same assignment as v1) was ranked in October-November 2026. A new version only takes samples that no earlier
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

Outputs in `work/ranking/<version>/final/`: `rankings.jsonl` (per sample: votes, upvote order, final ranking,
`confidence` = mean P of adjacent pairs, status), `pairs.jsonl` (per pair: `p_a` and the winner; soft or hard
labels for a scorer) and `summary.json`.

**v1 result (2026-10-13):** 6,996 samples ranked (RealEdit 5,749, GIER 1,229, others 18), 625 dropped because
someone skipped them. Raw agreement between two members is 57.8% of pairs (Krippendorff's alpha 0.14). Estimated
voter accuracy is 0.57-0.73, except one slot at chance level (weight 0). The upvote order agrees with the team on
61% of pairs and changes the final ranking of 865 RealEdit samples (mostly 1-1 splits). Known limitation: edits
were shown in key order, so rankings carry a position bias; shuffle the display order before ranking v2.

**Stand-ins and cleaning (`--stand-in upvote`, used for v1 and v2 since 2026-11-06).** Labels were needed before
every member had sent v2 results, and noisy labels should not reach the scorer. Where the team's labels are
missing or too noisy, the RealEdit upvote order is used instead:

- missing member: for a RealEdit sample, each assigned slot without a result is replaced by the upvote order with
  the median member weight; a sample that is not from RealEdit has no stand-in and is dropped (status "missing");
- noisy member (`--min-accuracy 0.6`): a member whose estimated accuracy is below 0.6 is treated as missing
  (v1: R3 0.50 and R8 0.57; v2: R9 0.545). A RealEdit sample stays ranked even when all its members are replaced;
- hard pair (`--hard-margin 0.1`): in a RealEdit sample, a pair whose probability is between 0.4 and 0.6 takes the
  upvote order with the confidence of one typical member (P = sigmoid(median weight), about 0.68-0.70);
- the half-weight reference vote stays, and a sample that a member skipped is still dropped.

When the missing results arrive, put them in `work/ranking/v2/results/<slot>/` and run `merge`, `finalize` and
`export` again.

**After cleaning (2026-11-06):** v1 6,996 ranked samples, 932 of 15,287 pairs replaced as hard; v2 9,114 ranked
samples, 2,158 of 20,911 pairs replaced. Ranked v2 samples by real members used: 3 (673), 2 (5,270), 1 (2,394),
0 (777, labels from the upvote order only). The final ranking equals the upvote order in 78.1% of RealEdit samples
in v1 and 94.7% in v2.

**v2 team labels (2026-11-06):** results from 6 of 9 slots (R3, R4 and R6 missing). 9,114 samples ranked, 1,009
dropped because someone skipped them, 11 dropped as "missing". Raw agreement between two members is 56.5% of pairs
(Krippendorff's alpha 0.11); estimated voter accuracy 0.55-0.72. The app still showed edits in key order in v2.

## Filtering (`src.filters`)

Two tasks remove images before training: **sensitive content** (`nsfw`) and **watermarks** (`watermark`). For each
task several detectors run, first on a public set with known labels, then on the dataset; the evaluation decides
whether a vote of several models beats the single best model enough to pay for the extra time. Results so far:
`docs/filters-benchmark.md`. Runs on Kaggle (2 x T4): `notebooks/filters_nsfw_benchmark.ipynb` and
`filters_watermark_benchmark.ipynb` (every model on the public sets: accuracy, precision, recall, F1, latency;
votes of every subset), then `filters_apply.ipynb` (the chosen models on the dataset).

```bash
.venv/bin/python -m src.filters list                                  # detectors and their extra pip packages
.venv/bin/python -m src.filters prepare --task nsfw                   # public benchmark sets (Kaggle only)
.venv/bin/python -m src.filters multirun --task nsfw --on benchmark --latency 100   # every model, both GPUs
.venv/bin/python -m src.filters run --task nsfw --model NAME --on benchmark   # on Kaggle
.venv/bin/python -m src.filters evaluate --task nsfw --on benchmark   # -> work/filters/nsfw/benchmark/evaluation.json
.venv/bin/python -m src.filters run --task nsfw --model NAME          # the dataset; on Kaggle
.venv/bin/python -m src.filters vote --task nsfw [--models a,b --method mean]   # -> work/filters/nsfw/decisions.jsonl
.venv/bin/python -m src.filters evaluate --task nsfw                  # vs hand labels -> work/filters/nsfw/evaluation.json
```

| Task | Detectors (`--model`) |
|---|---|
| nsfw | `falconsai` (ViT), `owen` (timm), `nudenet` (ONNX), `opennsfw2` (TensorFlow; RealEdit's filter), `mobilenet` (GantMan MobileNetV2, TensorFlow), `nemo` (LAION's CLIP-based NSFW detector through NeMo Curator), `clip` (CLIP zero-shot) |
| watermark | `convnext` (boomb0om ConvNeXt-tiny), `efficientnet` (qwertyforce EfficientNetV2-B0), `yolo` (qfisch YOLOv8n, AGPL-3.0), `laion` (LAION-5B watermark classifier), `clip` (CLIP zero-shot), `ocr` (EasyOCR text detection) |

- **A detector** subclasses `Detector` (`src/filters/base.py`): `load(device)` and `score(images)`, returning the
  probability that each image is positive. Model libraries are imported inside `load`, and packages beyond
  `requirements-ml.txt` are listed in `install` and recorded in `requirements-filters.txt` (they conflict, so the
  notebook installs them right before the model runs, each model in its own process). Put it in its own file in `nsfw/` or `watermark/` and register it in that folder's
  `__init__.py` (`DETECTORS`). Watermark checkpoints are pinned Hugging Face revisions, loaded with
  `torch.load(weights_only=True)` and, for pickled whole models, an explicit list of allowed classes.
- **Benchmark sets** (`benchmark.py`, community sets pinned to a commit, at most 1,000 images per class):
  NSFW `deepghs` (deepghs/nsfw_detect, the same 2,000 images as the first benchmark), `nsfw_xl`
  (markandey14/nsfw_detection_xl), `unsafebench` (yiting/UnsafeBench, real LAION-5B images; gated, skipped without
  access); watermark `boomb0om` (boomb0om/watermarks-validation, 122 images), `scenery`
  (qwertyforce/scenery_watermarks, real marks; the training set of `efficientnet`), `wm_or_not`
  (prithivMLmods/Watermark-or-Not-20K, synthetic marks). Images and labels go to `work/benchmarks/<task>/` (on
  Kaggle in `/tmp`, never in the saved output); results are reported per set and pooled.
- **Two GPUs** (`parallel.py`, `multirun`): one worker per GPU takes jobs from a queue; every model runs in its own
  process and its own environment (`envs.py`: a venv per group of conflicting packages, on top of Kaggle's ones).
  On the benchmark the models run side by side; on the dataset every model is split into one shard per GPU. The
  runner decodes the next batches on CPU threads while the GPU works, and `--latency N` times N images one by one
  (batch 1, after a warm-up).
- **Scores** go to `work/filters/<task>/[benchmark/]<model>.jsonl` (`{"image_id", "score"}`), written after every
  batch so a run can resume; `<model>.meta.json` records the device and the time per image.
- **Voting** (`configs/filters.toml`): `mean`, `weighted`, `max` (any model), or `majority` (share of models above
  their own threshold). Scores are combined without running the models again.
- **Evaluation**: on the benchmark, every scored model and every voting method; on the dataset, the models of the
  config against hand labels in `work/labels/<task>.jsonl` (`{"image_id", "label"}`: a few hundred images, mostly
  where the detectors disagree plus a random share). It reports accuracy, precision, recall, F1, the false
  positive rate and confusion counts at the thresholds, ROC-AUC, average precision, the best F1 over thresholds,
  the share flagged per class (benchmark), and the time per image. Votes are tried over every subset of 2+ models;
  `choice` gives the best single model, the best vote, the F1 gain and the time ratio.
- **Applying the filters:** bring `decisions.jsonl` of each task back into `work/filters/<task>/` and run
  `python -m src.rank export`: a flagged edit loses its pairs, a sample is dropped when its original is flagged or
  fewer than 2 edits are left; the images move to `work/filtered/` (and back if a later decision changes). Then
  `python -m src.process pack` for a new Kaggle dataset.

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
.venv/bin/python -m src.instructions reparse --run qwen2.5-vl-7b:analyze_v2  # parse stored answers again
.venv/bin/python -m src.instructions complete --client qwen2.5-vl-7b --max-hours 10   # on Kaggle: whole dataset
.venv/bin/python -m src.instructions merge --client qwen2.5-vl-7b           # final.jsonl from the run files
```

**The whole final dataset** (`complete.py`, `notebooks/instructions_complete.ipynb`). RealEdit publishes its
simplified instructions (the `instruction` column of its CSV files, written by the authors' LLM with the Fig. 19
prompt) but not its categories. So the 14,861 RealEdit samples with a published instruction keep it and the model
labels only their categories (`prompts/categorize_v1.txt`: Fig. 12 word for word, categories-only answer), while
the 1,249 other samples (GIER, forums, one empty RealEdit instruction) get both from the model (`analyze_v2`). Our
sample ID of a RealEdit post is sha1("realedit:<post id>")[:12], so the join needs no provenance (`realedit.py`).
Both passes share one loaded model and one time budget (`--max-hours`); a stopped run goes on where it stopped.
Output: `work/instructions/<client>/final.jsonl` (`id`, `instruction`, `instruction_source` realedit or qwen,
`categories` as "operation subject" strings, `problems`) and `final.summary.json` (done, left, hours left). With
qwen2.5-vl-7b on 2 x T4 the 16,020 requests left take about 13-14 h: two sessions of the notebook.

- **Clients** (`clients/`): open VLMs with one interface (`generate_batch`); Qwen2.5-VL 3B and 7B are registered
  (bfloat16 on GPUs that support it, else float16). An answer from a batch that is not JSON (float16 can give
  "!!!!") is asked again alone. Start from `notebooks/instructions_qwen_vl.ipynb` on Kaggle (`STEP` picks the run).
- **Long runs:** `--shard I/N` runs every N-th request, so each GPU takes one shard (`<prompt>.part-I-of-N.jsonl`);
  `--max-hours` stops cleanly before Kaggle's 12 h limit, and the next session resumes from the copied results.
  Measured on 2 x T4 (prompt v2): qwen2.5-vl-3b 1.9 s and qwen2.5-vl-7b 3.2 s per request, so all 17,755 requests
  take about 5 h (3B, one shard per GPU) or about 16 h (7B over both GPUs: two sessions).
- **Parsing** maps answers to RealEdit's labels and records every problem. It also reads three common slips: a
  category split into two list items (`["Remove", "Background"]`), extra words after the subject (`"Remove
  Background Person"`, first subject kept), and a bare subject after a category (`["Remove Background",
  "Person"]`: the second subject of the same operation, `Remove Person`, noted as a problem). `reparse --run CLIENT:PROMPT` applies a newer parser to stored answers.
- **Results:** `work/instructions/<client>/<prompt>[.part-I-of-N].jsonl`, one row per request: `instruction`, `categories`
  (`[{"operation", "subject"}]`), the model's raw answer, and parsing `problems`. Runs resume where they stopped.
- **Prompts** are versioned files; a version used for a run is never edited (add `analyze_v2.txt` instead). The
  `--sample N` subset is fixed, so prompt versions and models are compared on the same requests.
- The raw request is always kept; simplified instructions are never used as a training signal.

## Scorer (`src.scorer`)

`src/scorer/data.py` loads the preference pairs of the final dataset (`data/pairs.jsonl`, `load_pairs`; `a` is the
preferred edit, `p_a` the soft label), drops samples whose source forbids training, and assigns every sample to
train (70%), dev (15%) or test (15%) by a hash of its ID, so the split never changes. The final dataset gives
36,198 pairs (train 25,341, dev 5,388, test 5,469); v1 alone gave 15,287 (train 10,598, dev 2,239, test 2,450).
`version = "all"` in `configs/scorer.toml` uses every pair; `v1` or `v2`, the source of a sample (results by
source) and each annotator's ranking (the human ceiling) come from `work/`, so they work only on the coordinator's
computer.

The architecture (survey and proposal in `docs/scorer-survey.md`): frozen encoders (CLIP ViT-L/14, PickScore,
SigLIP, DINOv2) embed the original, the edit and the request once; a small head (linear or MLP) on features of
these embeddings gives one score per edit, trained on pairs of edits of the same request (Bradley-Terry on the soft
labels `p_a`, or an HPSv3-style uncertainty loss).

```bash
.venv/bin/python -m src.scorer embed --backbone clip-l14   # on Kaggle: cache embeddings of the ranked samples
.venv/bin/python -m src.scorer baselines                   # CLIP-score and other zero-shot baselines, human ceiling
.venv/bin/python -m src.scorer train                       # heads of configs/scorer.toml (minutes on a CPU)
.venv/bin/python -m src.scorer report                      # work/scorer/report.md: one table of everything
```

- **Runs** are listed in `configs/scorer.toml` (backbones, inputs, head, loss); each writes
  `work/scorer/runs/<name>/` (`metrics.json`, `scores.jsonl` per edit, `head.pt`). Model selection uses dev; test
  is reported once.
- **Metrics:** pairwise accuracy with a 95% bootstrap interval over samples, accuracy weighted by |2 p_a - 1| and on
  confident pairs, top-1 (best edit found), by source. The `edit-key-order` baseline (01 first) measures the
  position and upvote bias of the labels; it is not a valid scorer.
- Start from `notebooks/scorer_pilot.ipynb` on Kaggle (`STEP` picks the run). Pilot results (v1, 2026-10-22):
  60.4% test pair accuracy for an MLP head on SigLIP vs 50.6% for CLIP-score.

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
