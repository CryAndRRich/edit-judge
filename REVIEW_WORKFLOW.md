# Running T-045

`src/rank/review/` implements selection, review, versioned labels and round reports.
`src/scorer/oof.py` generates out-of-fold predictions. The self-contained
`notebooks/ranking_review.ipynb` runs the pipeline on Kaggle; regenerate its source bundle with
`python notebooks/build_ranking_review.py` after source changes. The default `prepare` phase imports
small result ZIPs without downloading images or model weights.

## Round 1 results

The 2026-11-24 run completed 25 OOF heads over 29,584 pairs and selected 250 samples. The group proposal
contains 247 rankings and 3 skips; 96 samples were flagged for closer inspection. Nguyễn Đức Minh
approved the proposals and subsequently confirmed checking the review handoff. The approval and
confirmation are stored separately from the frozen proposal in `user-approval.json` and
`manual-review-confirmation.json`.

Applying the approved proposals updated 736 pairs: 344 direction flips and 57 neutral pairs. Three
skipped samples retain their original labels. There are no sample removals. The original dataset and
r0 model are preserved.

| Test label version | r0 accuracy | r1 accuracy | r1 − r0 (95% paired CI), percentage points |
|---|---:|---:|---:|
| Original | 56.72% | 56.65% | -0.07 [-0.65, +0.54] |
| Reviewed | 57.04% | 56.88% | -0.16 [-0.75, +0.48] |

Neither comparison establishes an improvement. Control direction flips are 43/104 (41.35%); conflict
flips are 229/395 (57.97%). The report is complete, but the stopping criteria are not met. These rates
measure disagreement with approved proposals, not independently verified ground-truth errors.
Independent reviewer agreement is unavailable for the group-approval path. Optional extended reward
baselines were not run; coordinator replacement flags were unavailable.

Outputs are in the private HF dataset `IntroDS/RankingResults`, under
`checkpoints/minh-trung-dung/t045-r01-restart-20261124`.
Checkpoint `c14312ebd1403e15c95ee1a2c9f571c74cea0130` records the manual review confirmation.
The label, model and report files were downloaded on Kaggle and checked against SHA-256 manifests;
the pre-review checkpoint was also restored into an empty directory. See `completion-README.md`,
`completion-summary.json`, `changes.jsonl` and `report.json` for the round details.

## Approval of a group proposal

Use this path only after explicit approval:

```bash
python -m src.rank review-approve --round 1 --approver NAME --confirmation TEXT
python -m src.rank review-apply --round 1
```

Approval requires a complete proposal, matching frozen selection/edit mapping and source labels.
It records the approver and confirmation without generating independent votes or chunk assignments.
Directional pairs use p=0.8 as a label-strength setting, not measured reviewer confidence; marked
uncertain pairs use p=0.5. Skipped proposals retain their original pairs. The proposal remains frozen;
corrections after applying a round belong in a later round.

## Inputs and storage

- Attach the final `cryandrrich/editjudge` dataset to Kaggle. It must contain `images/` and `pairs.jsonl`.
- Grant the notebook access to Kaggle Secret `HF_TOKEN`, with permission to read `IntroDS/RankingResults`.
- Original v1/v2 result ZIPs supply annotator names and per-sample rankings/skips. The importer checks final
  sample coverage and rejects contradictory ranking/skip decisions. If two copies both skip a sample but use
  different reason text, all distinct reasons are preserved; this does not change the label. Use `--aliases names.json` to map spelling variants
  to one reviewer identity. Do not merge different people. Inspect `out/review/prepare.json` before selection.
- No 52 GB CrawlData download is needed. Samples and image keys come from the final dataset. Images stay on
  the read-only Kaggle Input; only embeddings, chunks and new labels are written on Kaggle.
- If available, restore `work/ranking/v*/final/` through `EJ_WORK_DIR`; it adds hard-pair and noisy-member
  replacement flags. These cannot be recovered exactly from raw rankings. Missing flags are reported rather
  than invented. Disagreement and label-confidence signals remain usable without those optional flags.
- Save a private Kaggle Version after each phase. Add the preceding saved output as an Input to the next
  session; the notebook restores it without a download to your laptop. Set `RESTORE_FROM` if several versions
  are attached. Dist ZIPs and HF download cache are not copied back; chunk ZIPs can be regenerated.

For automatic durable saving, set `BACKUP_REPO` to a **private** HF dataset with write permission and
`BACKUP_PREFIX` to a unique `checkpoints/...` directory. The round 1 run uses
`IntroDS/RankingResults/checkpoints/minh-trung-dung/t045-r01-restart-20261124`.
The setup verifies write access before GPU work, and restores a pinned snapshot with SHA-256 and size
verification when `RESTORE_BACKUP=True`. A different existing local output is never overwritten.
Each completed command and each completed OOF seed uploads an atomic commit. Frozen selection and
draft maps must also be checkpointed before review. Checkpoint `BACKUP.sync("review-draft")` after saving
review proposals. Embedding work in an interrupted, unfinished embedding command must be repeated;
completed embedding caches are saved and reusable. Model downloads, photos, secrets, HF download caches,
logs and distributable image ZIPs are excluded. No large files download to the laptop.
`BACKUP_VERIFIED` includes the HF commit revision, stage, file count and bytes; a file in `/kaggle/working`
alone does not establish durable saving. Original result ZIP import ignores `checkpoints/`.
Selection iterates split strata in a stable order across Python hash seeds. After dependency installation,
use a fresh Python process for restore verification; the live notebook can retain an older imported HF
library. The completed restore test used HTTP downloads with `HF_HUB_DISABLE_XET=1` in that fresh process.

For CLI runs set `EJ_DATA_DIR` to the final dataset, `EJ_OUTPUT_DIR` to a writable folder, and `HF_TOKEN` in the
environment. `EJ_PAIRS_FILE` optionally points at a versioned pairs file while `EJ_DATA_DIR/images` stays fixed.

## Before reviews

```bash
python -m src.rank review-prepare --hf-results IntroDS/RankingResults
# Or: review-prepare --results /path/to/original-result-zips --aliases /path/to/names.json
python -m src.scorer embed --backbones siglip
python -m src.scorer train --run mlp-siglip --output-name mlp-siglip-r0
python -m src.scorer oof --round 1 --max-hours 9
python -m src.rank review-select --round 1 --size 250
python -m src.rank review-chunks --round 1 --reviewers alice,bob,carol,dave --chunk-size 25
```

Use the real roster from prepare's output, not the example names. Every sample gets exactly two reviewers,
including at least one person outside its original annotators. A roster consisting only of a sample's original
annotators cannot satisfy this rule; use the available team roster. The default also creates 10-sample calibration
chunks for every reviewer. Calibration submissions are reported separately and never become final decisions.

OOF uses five fixed sample-level folds and five seeds per fold. Each held-out fold is excluded from training,
early stopping, feature standardisation and score scaling. An inner 10% sample split supplies early stopping.
Seed logits are checkpointed and reused only when the label, feature and configuration fingerprints match.
Temperature minimises soft-label ensemble log loss. Baseline scores are attached only where both edits exist;
missing coverage is explicitly reported. OOF requires complete image and text caches.

Optional second opinions, on Kaggle, before final selection:

```bash
python -m src.scorer rewards --models imagereward --split all --max-hours 9
python -m src.rank review-shortlist --round 1 --size 1000
# Set EJ_PAIRS_FILE to round-01/baseline-pairs.jsonl for this run only:
python -m src.scorer rewards --models editreward --split all --max-hours 9
# Restore EJ_PAIRS_FILE to the active full label version:
python -m src.scorer oof --round 1 --max-hours 9
```

The final OOF command reuses the trained seed caches and refreshes baseline votes. Freeze selection only after
the desired baseline runs finish. `--split all` scores all active samples, including train/dev.

Random controls are drawn **before** targeted selection, independently of conflict scores, and stratified by
split. Drawing them only after removing the most suspicious samples biases the control downward. Later rounds'
controls estimate the remaining unreviewed pool, not an untouched sample of the original whole dataset.

## Review and meeting

Only give each reviewer their assigned ZIPs and `ranking_app_review.zip` from `round-NN/dist/`.
The app requires the assigned reviewer name. Each sample's edits are randomly permuted and renamed to E01,
E02, etc.; filenames do not reveal their old upvote positions. The decoding map, original identities, split,
bucket, old rankings and scorer opinions stay with the coordinator. Reopening a chunk preserves its order.
Rank according to `IMPROVE_RANKING.md`. Discuss calibration before the main review.

```bash
python -m src.rank review-merge --round 1 --results /path/to/returned-review-jsons-or-zips
```

Identical independent orders are accepted at p=0.9; both reviewers skipping removes the sample. Other cases,
including one skip, remain pending. Copy the discussion entries from `pending.jsonl` into an
`adjudications.jsonl` file in the round folder and complete them in the human meeting:

```json
{"id":"sample-id","reviewers":["alice","carol"],"ranking":["02","01"],"third_reviewer":"bob","reason":"02 completes the request without changing the face","uncertain_pairs":[]}
```

Use ORIGINAL edit keys, as shown in `reviews.jsonl`, not the blinded E01 keys. The third reviewer must be a
different roster member. A meeting decision receives p=0.8; list unresolved original-key pairs under
`uncertain_pairs`, e.g. `[["01","02"]]`, to assign p=0.5 for those pairs. Missing reviews cannot be bypassed by
an adjudication. Rerun merge; `merge-summary.json` must show `complete: true` before applying labels.
Calibration alpha is separate from the independent two-reviewer agreement. Constant-label alpha is undefined
and reported as null.

## Apply, retrain, report

```bash
python -m src.rank review-apply --round 1
# Set EJ_PAIRS_FILE to out/review/round-01/labels-r1/pairs.jsonl:
python -m src.scorer train --run mlp-siglip --output-name mlp-siglip-r1
python -m src.rank review-report --round 1 --before-run mlp-siglip-r0 --after-run mlp-siglip-r1
```

The notebook phases do this environment switching automatically. Original labels are frozen at
`review/labels-r0/pairs.jsonl`. Each round stores its input snapshot, output label version, pair changes,
reviewers and reasons. Images and text fields are preserved. Applied decisions cannot be silently changed;
make the correction in a later round. Switching `EJ_PAIRS_FILE` back to an earlier snapshot undoes the round.
Round 2 must start from round 1's applied labels. Use a distinct output name for every trained label version.

The report measures directional flips, neutral pairs, removals, top-1 changes and reviewer agreement. Neutral
pairs train with p=0.5 but do not count as preferred-direction successes in hard accuracy; top-1 evaluation
omits samples with unresolved pair ties. Both scorers are evaluated on the same original labels and on the
same reviewed labels, with paired bootstrap differences by sample. Training label fingerprints must match
the round inputs/outputs. Counts and coverage are reported; a report without the retrained comparison is
marked incomplete and cannot recommend stopping. The reviewed test set is adaptive, so preserve original-label
comparisons and never select the scorer configuration using test accuracy.

The stop recommendation uses conflict flip <15% and control flip <5%, or two consecutive low-control rounds,
with a five-round hard limit. Inspect control sample counts and removed/neutral labels before accepting a stop;
small controls are uncertain. The hard-limit flag is independent of experiment completeness.

On the coordinator's full original work tree, `python -m src.rank finalize --version v1 --reviews /path/to/review`
(and v2) integrates the same decisions before the existing export. The Kaggle overlay does not require this
full work tree and never calls the image-moving export on read-only Inputs.

## Validation

Coordinator-local synthetic checks live in `tests/test_review.py` (tests remain git-ignored by repository
convention). Run `python -m pytest tests/test_review.py -q`. They cover a full tiny OOF run, cache invalidation,
selection, outsider assignment, blinded chunks, review adjudication, neutral labels, immutable snapshots and
reporting. They download no models, no images and no private data.
