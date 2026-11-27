# EditJudge

**Learning to judge photo edits from the crowd.** On photo-editing communities, people post a photo with an edit
request and other members answer with their edits. EditJudge collects these requests and edits, has a team rank the
edits of every request, and uses the rankings to learn what makes an edit good (Introduction to Data Science,
IT4142E, class 176080).

The work is split into stages (the plan follows the course project proposal):

| Stage | Folder | Status |
|---|---|---|
| 1. Data preparation: crawl, build and normalize the dataset, team ranking | `src/crawl`, `src/process`, `src/rank`, `apps/ranking` | Done (v1 and v2 ranked, final dataset exported) |
| 2. Filtering: sensitive images and watermarks, each by several detectors and a vote | `src/filters` | 7 NSFW and 6 watermark detectors benchmarked on public sets; applied to the dataset on 2026-11-07 (`../docs/filters-benchmark.md`) |
| 2. Request analysis with open VLMs (RealEdit's prompts): simplify and categorise the requests | `src/instructions` | Done (2026-11-07): every sample of the final dataset has an instruction and categories in `data/pairs.jsonl` |
| 3. The learned scorer and its baselines | `src/scorer` | Final dataset (2026-11-07): `mlp-siglip` 56.7% test pair accuracy (human ceiling 67.3%), +4.9 to +8.5 points over CLIP-score, PickScore, SigLIP and HPSv2.1 (paired intervals above 0), 0.2 s per request on a T4; reward-model and VLM-judge baselines running on Kaggle (`../docs/scorer.md`) |
| 4. Editing models per operation, and their benchmark | `src/editors` | All 7 lists of `../docs/editors.md` (23 editors, 7 tasks). Lists 2 and 3 (8 models) ran on 30 dev samples per task (2026-11-21); the quick test of lists 1, 4 to 7 (2 samples per task, 2026-11-24) ran 14 models; 4 slow ones and VAREdit 2B were dropped; next, every model on the first 80 samples of each task, scored (`SESSION = "small"`, about 9-10 h) |
| 5. The editing assistant | (no code yet) | Planned |

## Repository layout

```
src/                        Python package (run as `python -m src.<module>`)
  paths.py                  every data location, in one place (EJ_DATA_DIR / EJ_WORK_DIR / EJ_OUTPUT_DIR override it)
  common/                   shared helpers: JSONL files, dataset images and texts, device and the no-GPU guard,
                            per-model Python environments (envs.py), runs over all GPUs (gpus.py), .env
  crawl/                    crawler framework: polite HTTP, SQLite state, one module per source
  process/                  dataset build (adapters, filter, writer), streaming collection, image normalization
  rank/                     ranking chunks and assignments, result checking, final rankings, export of data/
  filters/                  detector interface and runner, public benchmark sets, voting, evaluation
    nsfw/                   sensitive-content detectors, one file each
    watermark/              watermark detectors, one file each, and the safe checkpoint loader
  instructions/             request analysis (RealEdit): label sets, prompts/ (versioned), clients/ (VLMs), runner
  scorer/                   preference pairs and splits, encoders and embeddings, heads, training, evaluation, report
    rewards/                reward models and VLM judges used as baselines (ImageReward, EditReward, EditScore, ...)
  editors/                  editing models (`../docs/editors.md`): sample sets, VLM edit plans, grounding masks, one
                            wrapper per model in models/, runner, scores and report
apps/ranking/               Gradio app the annotators run on their own computers (handed out as a zip)
apps/web/                   EditJudge Studio, the assistant's web app (FastAPI); see "Web app"
configs/                    experiment settings (TOML), e.g. filters.toml, scorer.toml
notebooks/                  Kaggle notebooks, named by stage: filters_*, instructions_*, scorer_*, editors_run,
                            ranking_review (built by python -m src.rank.review.notebook)
data/                       local only, git-ignored: the final dataset (images/, pairs.jsonl); see "Data"
work/                       local only, git-ignored: everything used to build data/, and stage outputs
requirements/               dependencies, one file per use (each includes the ones it builds on with -r):
  base.txt                  stage 1 (crawl, dataset, ranking): the laptop
  ml.txt                    model libraries for the later stages (base.txt +)
  filters.txt               extra packages of the image filter detectors (installed one model at a time on Kaggle)
  scorer.txt                extra packages of the scorer's reward-model baselines (one environment per model)
  editors.txt               extra packages of the editing models (ml.txt +)
  web.txt                   the web app alone (mock models), e.g. on a laptop
  ranking.txt               the annotators' ranking app (shipped in its zip as requirements.txt)
  server.txt                the GPU server of the live demo: editors.txt + web.txt (see "GPU server")
.env.example                settings and secrets; copy to .env (git-ignored)
```

Design notes, surveys and results are not in this repository: they live in the project's `docs/` folder next to
it (`../docs/`: `PROJECT.md`, `filters-benchmark.md`, `scorer.md`, `editors.md` (the model list), `ranking-review.md`,
...).

## Setup

Python 3.10 or newer.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements/base.txt
cp .env.example .env        # then fill in what you need; never commit .env
```

All commands run from the repository root, e.g. `.venv/bin/python -m src.rank --help`.

Model libraries (`requirements/ml.txt`) are for Kaggle, where most of them are preinstalled. Locally they are only
for small smoke tests; on an Intel Mac use Python 3.11 with `torch==2.2.2`, `torchvision==0.17.2`, `numpy<2` and
`transformers<5` (newer versions have no wheels there).

## Running on Kaggle

Heavy work (detectors over the whole dataset, LLM/VLM inference, features, editing models) runs on Kaggle with a
GPU, never on a laptop. Without a GPU, model runners refuse more than 20 items (`src/common/device.py`; override
with `--allow-cpu` only for an approved run).

- The data goes to Kaggle as a **private** dataset shared only with the team (it contains photos of real people):
  `python -m src.process pack` writes `work/exports/editjudge-data.zip` (7.6 GB: `data/` as it is, `images/` and
  `pairs.jsonl`). Upload it as a new version of the private dataset `cryandrrich/editjudge`; the notebooks find
  the folder that holds `pairs.jsonl` among their inputs.
- Every notebook clones the code from GitHub (branch `curr`, so push first), sets `EJ_DATA_DIR` to the read-only
  input and `EJ_OUTPUT_DIR` to `/kaggle/working/out`, so the code runs unchanged and every result lands in one
  output folder. Settings: GPU T4 x2, internet on. Long runs stop by themselves after `HOURS` and go on in the
  next session when the notebook's previous output is added as input.
- The notebooks:
  - `filters_nsfw_benchmark`, `filters_watermark_benchmark`: every detector on the public benchmark sets;
  - `filters_apply`: the chosen detectors on the dataset, decisions for `python -m src.rank export`;
  - `instructions_complete`: the simplified instruction and categories of every request (Qwen2.5-VL-7B);
  - `scorer_train`: embeddings, similarity baselines, trained heads and the report;
  - `scorer_baselines`: reward-model and VLM-judge baselines on the test split.
  - `editors_run`: the editing models of every list on 5 (smoke) or 30 dev samples per task.
- Bring the results back into `work/` (same sub-folders) and record the run (model, version, time) in the pull
  request. Do not commit outputs.

## Working as a team

Every task has its own folder, so members can work in parallel and merge with few conflicts.

- Branch from `curr`, one branch per task and person: `<task>/<name>`, e.g. `filters-nsfw/minh`.
- Work inside the task's folder (for example `src/filters/nsfw/` plus its entry in `nsfw/__init__.py`). Changes to
  shared files (`src/paths.py`, `src/common/`, `requirements/`, `README.md`) go in a separate small commit and
  are announced to the coordinator.
- Keep the shared formats: results keyed by `image_id` ("<sample id>/<key>") or sample ID, JSONL files under the
  task's data folder, settings in `configs/`.
- Unit tests (synthetic data only) are kept on the coordinator's computer, not in the repository (`tests/` is
  git-ignored).
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
  number of pairs it wins (25 of 29,584 pairs disagree with the full ranking).
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
  ranking/dist/                     the ranking app zip handed out to annotators (results are in ranking/vN/results)
  filters/, instructions/, scorer/, editors/
                                    results of the later stages, one folder per task (EJ_OUTPUT_DIR)
  models/                           downloaded model weights
  exports/                          zips for Kaggle: editjudge-data.zip (`python -m src.process pack`, rebuilt when
                                    needed), editjudge-scorer.zip (the mlp-siglip head for editors_run)
```

- `note` is the text that came with an edit (the editor's comment, or a description of the change); may be null.
- Samples carry no source information. `index/provenance.jsonl` maps them back to their source, for licences,
  takedown requests, de-duplication and the VOZ no-training rule (`trainable: false`; the final dataset has no VOZ
  sample).
- The raw crawl outputs and the username salt were deleted (crawling is closed). Paths stored in the ledger and the
  provenance index (`raw/<source>/...`) are kept only as a record.

**Final dataset (2026-11-07, after the image filters):** 13,899 samples, 48,226 images (34,327 edits), 29,584 pairs,
7.2 GB; packed as `work/exports/editjudge-data.zip` (7.6 GB). Of the 16,110 ranked samples (v1 6,996, v2 9,114), the
filters (48 NSFW and 5,096 watermark images flagged; `../docs/filters-benchmark.md`) dropped 3,957 edits, 1,185 samples
with a flagged original and 1,026 samples left with fewer than 2 edits; their 8,342 images are in `work/filtered/`.
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
| RealEdit (CVPR 2025, CC BY 4.0): r/PhotoshopRequest and r/estoration posts as image URLs | `sites/realedit.py` | Done: all 22,221 posts with at least 2 edits. Edits are ordered by upvotes |
| dprevived.com "Edit Me An Image" (Misago) | `sites/dprevived.py` | Done |
| mineo (Japan) | `sites/mineo.py` | Done; requests and edits share one thread and are paired by rules |
| fotokto.ru (Russia, phpBB) | `sites/fotokto.py` | Done (small yield) |
| VOZ (Vietnam, XenForo) | `sites/voz.py` | Done; robots.txt says `ai-train=no`: never used for training |
| photoshopcontest.com | `sites/photoshopcontest.py` | Tried on 2026-10-07 (39 samples), stopped by decision |
| b3ta.com image challenges | `sites/b3ta.py` | Tried; no usable samples (composites, dead image hosts) |
| Zhopped via the Wayback Machine | `sites/zhopped.py` | Mostly thumbnails; archive.org rate limits |
| Reddit API | `sites/reddit.py` | Blocked: API use needs prior approval since 2025-11-11. Refuses to run without credentials |
| PhotoshopGurus (XenForo) | `sites/photoshopgurus.py` | **Suspended**: the team account was banned after a logged-in crawl. Refuses to run unless `PSG_PERMISSION=granted`, which may only be set after a forum admin allows it |

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

**Adding a source:** subclass `Source` (or `XenForoSource`) in `sites/<name>.py`, set `name`, `label` and
`base_url`, implement `discover()`, `fetch_thread()` and `probe_targets()`, register the class in
`sites/__init__.py`, and run `probe` before crawling.

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
`../docs/filters-benchmark.md`. Runs on Kaggle (2 x T4): `notebooks/filters_nsfw_benchmark.ipynb` and
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
  `requirements/ml.txt` are listed in `install` and recorded in `requirements/filters.txt` (they conflict, so the
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
  process and its own environment (`src/common/envs.py`: a venv per group of conflicting packages, on top of
  Kaggle's ones).
  On the benchmark the models run side by side; on the dataset every model is split into one shard per GPU. The
  runner decodes the next batches on CPU threads while the GPU works, and `--latency N` times N images one by one
  (batch 1, after a warm-up).
- **Scores** go to `work/filters/<task>/[benchmark/]<model>.jsonl` (`{"image_id", "score"}`), written after every
  batch so a run can resume; `<model>.meta.json` records the device and the time per image.
- **Voting** (`configs/filters.toml`): `mean`, `weighted`, `max` (any model), or `majority` (share of models above
  their own threshold). Scores are combined without running the models again.
- **Evaluation** (benchmark only; the dataset has no labels): every scored model and every voting method. It
  reports accuracy, precision, recall, F1, the false
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

Result (2026-11-07, 2 sessions: 2.7 s per categorize request, 3.2 s per full analysis). On the 13,899 samples of
the final dataset: 12,736 RealEdit instructions, 1,072 written by the model, 91 corrected by hand, and every
sample has at least one valid category (1 for 11,206, 2 or more for 2,693). Two fixes after the run:
- the model sometimes answered labels outside the taxonomy ("Move Person", "Change-Color Helmet", a bare
  "Enhance"); the parser maps them to the closest taxonomy name (`loose_category`, word lists in `schema.py`,
  recorded as "mapped ..." problems), and `reparse` applied it to the stored answers without the model;
- for about 90 vague requests ("Make the photo more attractive") the model wrote a category name ("Enhance
  Nature") or one word ("touchup") as the instruction; they were rewritten by hand following Fig. 19 in
  `work/instructions/qwen2.5-vl-7b/manual.jsonl`, which `merge` applies (`instruction_source` "manual").

- **Clients** (`clients/`): open VLMs with one interface (`generate_batch`); Qwen2.5-VL 3B and 7B are registered
  (bfloat16 on GPUs that support it, else float16). An answer from a batch that is not JSON (float16 can give
  "!!!!") is asked again alone. The 3B/7B comparison on 100 requests used the `run`, `stats` and `agree`
  commands; the whole dataset runs in `notebooks/instructions_complete.ipynb`.
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

Design, survey, baselines and results: `../docs/scorer.md`. In short:
- **Data:** the preference pairs of the final dataset (`data/pairs.jsonl`; `a` is the preferred edit, `p` the soft
  label), split by a hash of the sample ID: 29,584 pairs (train 20,588, dev 4,477, test 4,519). The text is the
  simplified instruction of the request (the request as written when there is none; `text` in
  `configs/scorer.toml`).
- **Main model:** five frozen encoders (CLIP ViT-L/14, PickScore, SigLIP, HPSv2.1, DINOv2) embed every image and
  text once; a small head (linear or MLP) on features of these embeddings gives one score per edit, trained on
  pairs of edits of the same request (Bradley-Terry on the soft labels, or an HPSv3-style uncertainty loss), one
  head per seed, the epoch with the lowest dev loss. Images are padded to a square (no centre crop).
- **Baselines** on the test split: CLIP-score, PickScore, SigLIP and HPSv2.1 (cosine of edit and instruction);
  ImageReward, EditReward-7B, Qwen2.5-VL-7B zero-shot and EditScore-7B (`src/scorer/rewards`, each in its own
  Python environment, pinned upstream code).
- **Evaluation:** pairwise accuracy with a 95% bootstrap interval over samples, paired differences between
  scorers on the same pairs, top-1, the human ceiling; one run selected on dev; the latency of one request
  (two edits, batch 1) of every scorer and baseline.

```bash
python -m src.scorer embed --gpus 2              # Kaggle: embeddings, every image decoded once, one process per GPU
python -m src.scorer baselines                   # similarity baselines and the human ceiling
python -m src.scorer train                       # every run of configs/scorer.toml, 5 seeds each
python -m src.scorer latency                     # Kaggle: time of one request (2 edits) per run and similarity baseline
python -m src.scorer rewards --max-hours 10      # Kaggle: reward models and VLM judges (timed too), resumable;
                                                 # --shard i/n: one part of the edits per session or member
python -m src.scorer report                      # work/scorer/report.md (small files only: runs on a laptop)
```

Outputs in `work/scorer/`: `embeddings/<encoder>/`, `runs/<name>/` (`metrics.json`, `scores.jsonl`, `head.pt`),
`baselines/<name>.jsonl` (a score per edit) and `<name>.latency.json`, `latency.json`, `report.md`. Notebooks: `scorer_train.ipynb`, `scorer_baselines.ipynb`.

## Editing models (`src.editors`)

The models of `../docs/editors.md` (all 7 lists, 23 editors: `python -m src.editors list`), run on a small fixed set of
dev samples per task and judged by the scorer:

| Task | Ops | Specialists (list) |
|---|---|---|
| remove | remove | `lama`, `migan`, `rorem`, `objectclear`, `osor` (2) |
| add_replace | add, replace, background | `sdxl-inpaint`; mask-free `paint-by-inpaint`, `erasedraw` (add only) (3) |
| enhance | light, enhance | `realesrgan`, `hvi-cidnet` (light only), `instructir`, `vosr` (4) |
| restore | restore | `vosr`, `codeformer`, `gfpgan` (5) |
| change_color | colorize, recolor | `ddcolor` (colorize), `sdxl-inpaint` (recolor) (6) |
| formatting | cutout, format | `ben2`, `birefnet` (cutout); `plain-code`, `sdxl-outpaint`, `realesrgan` (format actions) (7) |
| open_ended | open | none: the generalists only |

The generalists of list 1 (`instruct-pix2pix`, `magicbrush`, `realedit`, `flux2-klein-4b`) run on
every task, with the instruction as their prompt (InstantIR, OmniGen2, VIBE and FLUX 3 Image Klein were dropped on
2026-11-24 as too slow for live editing, VAREdit 2B on 2026-11-26: `../docs/editors.md`, "Left out"). The inpainting models need a
mask for every sample; the mask-free ones get the instruction instead. Models whose code pins other packages get
their own environment (`install`, `src/common/envs.py`; the pins are recorded in `requirements/editors.txt`);
models that ship code but no package are downloaded at a pinned commit into `EJ_MODELS_DIR`.

```bash
python -m src.editors samples --task remove          # 30 dev samples with a "Remove <subject>" category (laptop)
python -m src.editors plan --task remove --limit 5   # Kaggle: Qwen2.5-VL-7B says what to find / where / what to paint
python -m src.editors masks --task remove --limit 5  # Kaggle: Grounding DINO + SAM 2.1 (add: the plan's box)
python -m src.editors runall --task remove --limit 5 # Kaggle: every model of the task, each in its own process
python -m src.editors runall --tasks enhance,restore --models vosr   # several tasks in one process: loads once
python -m src.editors score --task remove            # Kaggle: mlp-siglip on the outputs and the human edits
python -m src.editors report --task remove           # work/editors/remove/report.md and contact sheets
python -m src.editors summary                        # work/editors/summary.md: every task and model in one table
```

- **Samples** (`samples.py`): dev samples only, single-category requests first, dealt round-robin over subjects so
  the first 5 (the smoke run) already differ. `add_replace` is half Add, half Replace; Add/Replace Background get
  the op `background`. Left out: "Remove Background" (list 7), samples with a category of the other task, humorous
  requests, and requests one inpainting step cannot answer (cut-outs, crops, retouching and restoration for
  remove; changes of size or shape and whole-image or identity changes for add_replace; word lists in
  `UNSUITABLE`). The other tasks (`GENERIC`, 2026-11-24) take their categories by first word, each with its ops:
  enhance `light` (a keyword test of the instruction) or `enhance`; change_color `colorize` (a greyscale original)
  or `recolor`; formatting half "Image-Formatting" (`format`) and half "Remove Background" (`cutout`); open_ended
  "Open-Ended" and "Other". A sample belongs to one task only (these tasks skip samples of the other tasks' files).
- **Plans** (`plan.py`, `prompts/plan_v<k>.txt`; remove uses v2, add_replace v3): Qwen2.5-VL-7B names each
  target with its own box, what to keep (background; add: what the new thing must not cover), for Add the existing
  thing it goes on (its anchor: a face for a beard) or else a free box, and the inpainting prompt. Its boxes are in pixels of the image it saw, which is read from the
  processor (`image_grid_thw`): the model's config overrides the client's pixel limit, so plan_v1's mapping was
  wrong (smoke run, 2026-11-20). Answers with problems are recorded; a background plan with nothing to keep is
  asked again for the subjects alone (`prompts/keep_v1.txt`). change_color and formatting use plan_v4: the things
  to recolour and their new colour (none when the whole photo's colour changes), or the format action (crop with
  its box, rotate, flip, resize, extend, grayscale, cutout, other).
- **Masks** (`grounding.py`): Grounding DINO's boxes that lie in Qwen's box for the target (so "the letters DB"
  keeps both), Qwen's box when DINO finds none there, then SAM 2.1 and their union; grown by 15 px (remove) or
  8 px (replace); for `background` the kept subjects' mask is inverted; for Add, a rectangle around the anchor
  (or the free box) minus the kept things; an addition worn on the face or head (a beard, glasses, a hat) is
  anchored on the face or head, not the whole person, and an Add rectangle covers at most 30% of the image. A
  sample where nothing is found is `no-mask`. recolor masks are segmented like replace targets (grown by 4 px);
  the other ops get no mask.
  Masks record their plan version, and a mask or an output of an older plan version is made again.
- **Runner** (`base.py`): an editor subclasses `Editor` (`load(device)`, `edit(image, mask, prompt)`; `uses_mask`,
  `ops` or `applies(sample, plan)`, `keep_size`, `settings_version`) in its own file in `models/` and is registered in `EDITORS`; settings follow `../docs/editors.md`. An output
  records the digest of the mask it was made from and the editor's `settings_version`, so only samples whose mask
  or settings changed are run again. Outputs are resized back to
  the original size, and only the masked region is kept (pasted back on the original over a 3 px soft edge), so
  models working at 512 px are not judged on a resampled photo. Edits that change the size on purpose (crop,
  quarter turn, resize, extend: `keep_size = False`) keep it. `runs.jsonl` records status (`ok`, `failed`, `black`, `no-mask`), seconds and peak GPU memory.
- **Scores and report** (`score.py`, `report.py`): every output is scored together with the human edits of its
  sample. The scorer learned from human edits, so AI outputs are another domain: check the contact sheets. With 30
  samples the shares have wide intervals.

Outputs in `work/editors/`: `samples/`, `plans/`, `masks/`, `<model>/` (`<sample id>.png`, `runs.jsonl`),
`<task>/` (`scores.jsonl`, `report.md`, `sheets/`), `summary.md`. Notebook: `editors_run.ipynb`, three sessions: `preflight`
(no GPU: builds every environment, downloads every model and runs the small ones on the CPU), `quick` (every model on
2 samples per task, about 6 h on 2 x T4) and `full`. Models run two at a time, one per T4, while a background thread
downloads the next ones (`python -m src.editors fetch`, sizes and folders from `python -m src.editors info`); each
model's downloads are deleted after its last run.

## Improving ranking labels (T-045)

The review loop is implemented in `src/rank/review/` and `src/scorer/oof.py`; the guide and the commands are in
`../docs/ranking-review.md`. The self-contained [Kaggle notebook](notebooks/ranking_review.ipynb)
imports the original HF result ZIPs and runs the automated stages on Kaggle without downloading the images
to an annotator's laptop. Revised labels require independent reviews and adjudication, or explicit approval
of a group proposal recorded with `review-approve`.
`EJ_PAIRS_FILE` switches between frozen label versions while the images stay in `EJ_DATA_DIR/images`.

## Web app (EditJudge Studio)

**EditJudge Studio** (`apps/web/`) is the live photo-editing assistant of PROJECT.md Part III. A user picks
or uploads a photo and writes a request (English or Vietnamese). The app reads the request, runs the best-ranked
editing models of its task one after another, scores each edit as soon as it is ready, and lets the user compare any
result with the original. Two more tabs show the model ranking (Part II Step 7) and the dataset.

The layout and components are Steven's original app (`studio/static/css/app.css`, merged on 2026-11-25; the rules the page
no longer uses were removed on 2026-11-27). The assistant flow, the tabs, the animations, the Dataset figures and the
dark theme are in `studio/static/css/assistant.css` and `studio/static/js/app.js` (design agreed on 2026-11-27). The clickable
prototype of this design is `../docs/ui-prototype/edit-judge-studio.html`, outside this repository.

### What is real and what is simulated

| Part | On a laptop (now) | On the GPU server (next step) |
|---|---|---|
| Reading the request | keyword router (`studio/router.py`, EN and VI); a Vietnamese request is sent unchanged | Qwen2.5-VL-7B (`src/editors/plan.py`): task, English instruction, target |
| Model choice | real: `assistant.py` (rule below), from `data/rankings.json` | the same |
| Editing models | plain code (PIL) and Real-ESRGAN (spandrel) run; the others are **mocks** that wait about as long as the real model takes | the `src/editors` wrappers |
| Score | proxy heuristic (`scorer.py`), shown as exp(gain) | our scorer mlp-siglip (`work/scorer/runs/mlp-siglip/head.pt`), exp of its score |

Every score is above 0 and higher is better.

### The flow of one request

1. **Read the request** (`assistant.understand`): one of the 7 benchmark tasks (remove, add_replace, enhance,
   restore, change_color, formatting, open_ended), the operations it needs (add, low light, colourise, recolour,
   upscale, crop/rotate/flip/resize/greyscale, cut-out, extend), the English instruction sent to the models and the
   target to find. The request box shows them (task, instruction sent, target, models). A request has at most 100
   characters. The `task` field of `POST /api/jobs` can override the task (the page does not send it).
2. **Pick the models** (`assistant.pick_models`, `../docs/editors.md` "Live demo selection"): the models of the task, best
   mean score first, without FLUX.2 klein 4B and InstructPix2Pix (too slow live). A model limited to one operation
   (EraseDraw to Add, HVI-CIDNet to low light, DDColor to colourising, plain code to geometry, BEN2 and BiRefNet to
   cut-outs, SDXL Outpainting to extending, SDXL Inpainting in Change color to recolouring, Real-ESRGAN in
   Formatting to upscaling) runs only on requests with that operation. **When 6 apply, the first 6 run; otherwise
   the first 3, or all when fewer** (Open-ended has 2).
3. **Run them one after another, fastest first** (`pipeline.Jobs`), so the first result shows within a second or
   two. One job at a time uses the models; another request waits (status "waiting").
4. **Score each output** as soon as it is ready; when all have run, the best one is marked.
5. **Show the result:** the cards move into rank order (failed models last), each with a download button, and the
   best edit opens in the before / after slider, with its model, score, latency and rank under it. Clicking another
   card opens that edit.

#### The progress bar of a model

The expected time is the model's mean seconds per image on a T4 (`data/rankings.json`) times `EJ_WEB_SPEEDUP`
(0.45 for an A30). The bar fills linearly to 85% over 85% of the expected time, then slows down toward 99% and never
reaches 100% before the result; when the result arrives it runs to 100% in 250 ms. Past 115% of the expected time
the card says "slower than usual". The bar at the bottom of the request box sums the expected times of all models.

### How to run (laptop)

Commands from the repository root:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements/web.txt
.venv/bin/python apps/web/fetch_dev_split.py    # 5 public sample photos (Picsum/Unsplash, COCO) into apps/web/dev_split/
.venv/bin/python apps/web/run.py                # http://127.0.0.1:8000, API docs at /docs
```

`apps/web/run.py --port 8010` uses another port, `--reload` restarts on code changes. For the GPU server see "GPU
server" below.

| Variable | Default | Meaning |
|---|---|---|
| `EJ_WEB_SPEEDUP` | `0.45` | T4 seconds times this = the expected seconds on this machine |
| `EJ_WEB_MOCK_DELAY` | `1` | `0`: mocks return at once (no waiting) |
| `EJ_WEB_SCORER` | `proxy` | `proxy`, `imagereward`, `editreward` (`scorer.py`; mlp-siglip comes with the server) |
| `EJ_WEB_REAL_MODELS` | `1` | `0` replaces Real-ESRGAN with a Lanczos mock (no torch needed) |
| `EJ_WEB_ESRGAN_MAX_SIDE` | `512` | Real-ESRGAN input size; the output is 4 times larger |
| `EJ_WEB_KEEP_RUNS` | `40` | job folders kept in `apps/web/studio/outputs/` |
| `EJ_WEB_UPLOADS`, `EJ_WEB_OUTPUTS`, `EJ_WEB_CACHE` | under `apps/web/studio/` | storage folders |

### Files

```
apps/web/
  run.py                  starts uvicorn
  fetch_dev_split.py      downloads the 5 sample photos
  dev_split/              the sample photos (only manifest.json is committed)
  studio/                 the app (Python package)
    main.py               FastAPI routes
    assistant.py          reading the request, picking the models, the Rankings view
    pipeline.py           uploads, samples, jobs (run one after another, score, rank)
    catalog.py            the live models (src/editors registry names), rankings.json and dataset.json
    router.py             keyword intents of a request (EN and VI) and parameters
    model_runner.py       one handler per model: real (plain code, Real-ESRGAN) or mock
    scorer.py             proxy score, ImageReward / EditReward wrappers
    templates/index.html  the page: tabs Assistant, Rankings, Dataset
    static/css/app.css    Steven's design system (layout, components)
    static/css/assistant.css  the assistant's additions and the dark theme
    static/js/app.js      the page logic (no build step)
    data/rankings.json    the scored run of 2026-11-26: per task, every model's mean score and seconds per image
    data/dataset.json     the figures of the Dataset tab (from the two reports in ../docs/report and data/pairs.jsonl)
```
`studio/data/rankings.json` is written from `work/editors/<task>/scores.jsonl` and `runs.jsonl` after a scored run of
`notebooks/editors_run.ipynb`; rewrite it when the benchmark runs again.

### API

| Method | Path | Body / result |
|---|---|---|
| GET | `/` | the app; `#rankings` and `#dataset` open those tabs |
| GET | `/api/health` | status, device, model count, scorer |
| GET | `/api/samples` | sample photos with a request, source, author and licence |
| POST | `/api/upload` | multipart `file` -> `{image_id, url, width, height, filename, bytes}` |
| POST | `/api/jobs` | `{image_id` or `sample_id, instruction, task?}` -> `202` the job |
| GET | `/api/jobs/{job_id}` | the job, polled every 300 ms by the page |
| GET | `/api/rankings` | per task: every benchmarked model with its place, score, T4 `seconds`, `estimate` on the live GPU (seconds x `EJ_WEB_SPEEDUP`) and whether it runs live |
| GET | `/api/dataset` | the figures of `data/dataset.json` and the number of benchmarked models |

A job: `{job_id, status (waiting, running, done, error), understood: {task, task_label, operations, english, target,
vietnamese}, original: {url, width, height}, candidates: [{model, name, benchmark, estimate, status (queued,
running, ok, failed), elapsed, seconds, score, rank, best, image_url, thumb_url, transparent, backend, error}]}`.
Candidates are in running order (fastest first).

Upload rules: JPEG, PNG, WebP, BMP, TIFF and GIF, up to 25 MB and 60 MP; the photo is normalised before it is stored
(EXIF rotation applied, RGB, at most 2048 px). Errors return `{"detail": "..."}`: 400 empty file, 404 unknown image,
sample or job, 413 too large, 415 not a readable image, 422 invalid body (empty or longer than 100 characters, unknown
task).

### GPU server

One machine with a GPU that clones this repository, installs it, downloads every model ahead of time and serves
the live demo. It needs one requirements file, `requirements/server.txt`; the others are records of the Kaggle
stages. The four editors that pin their own packages (`install`: realesrgan, vosr, gfpgan, flux2-klein-4b) get a
small environment on top of the main one (`src/common/envs.py`, the same as on Kaggle), built by `setup`:

```bash
git clone https://github.com/CryAndRRich/edit-judge.git && cd edit-judge
python3 -m venv .venv && .venv/bin/pip install -r requirements/server.txt
export EJ_MODELS_DIR=/data/models EJ_ENVS_DIR=/data/envs   # a large disk: about 80 GB of weights
.venv/bin/python -m src.editors setup                      # environments and downloads of every editor (no GPU)
.venv/bin/python apps/web/run.py --host 0.0.0.0             # the web app
```

`setup --models a,b` prepares only some editors; it resumes, so a second run only checks what is there. The
planner (Qwen2.5-VL-7B), Grounding DINO, SAM 2.1 and the scorer's SigLIP download into the Hugging Face cache on
first use.

### Next steps

- Replace the mocks with the `src/editors` wrappers, one model at a time, keeping the models in RAM and moving one at
  a time to the GPU (peak about 11.5 GB, `../docs/editors.md`).
- Qwen2.5-VL-7B (4-bit) reads the request and writes the English instruction; Grounding DINO and SAM 2.1 make the
  masks of remove, add_replace and change_color.
- Score with mlp-siglip; check free GPU memory before each model.
- Jobs are in memory: a restart loses the running ones.

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
