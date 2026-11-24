# Improving the ranking labels

This guide is for the team members who review and fix the final rankings in `data/pairs.jsonl`.

## Why

The final rankings come from 3 members per sample and, for RealEdit samples, the Reddit upvote order. Some of them are wrong:
- members disagreed and the vote was close;
- noisy members and hard pairs were replaced by the upvote order;
- the app showed edit 01 (the most upvoted) first, which biases members towards it.

The scorer learns from these labels, so label noise caps its accuracy. The selected scorer agrees with 56.7% of the test pairs. On test, one member agrees with the other two (when those two agree) on only 67.3% of the pairs.

## The loop

Each round has four steps. Steps 1, 2 and 4 are run by the coordinator. Step 3 is done by the reviewers.

```
Step 1  score every pair (out-of-fold)
   |
Step 2  select about 250 samples to review
   |
Step 3  two reviewers rank each sample again, then disagreements are settled together
   |
Step 4  update the labels, retrain, measure the flip rate
   |
   +--> next round, until the stop rule holds
```

All files of a round go to `work/review/round-NN/` (git-ignored, coordinator only). The original labels are never overwritten. Reviews are applied on top of them, so every round can be undone.

The review tools are implemented in `src/rank/review/` and `src/scorer/oof.py`; see
[REVIEW_WORKFLOW.md](REVIEW_WORKFLOW.md) and `notebooks/ranking_review.ipynb` for commands and storage.

Round 1 completed on 2026-11-24: 25 OOF heads, 29,584 pairs and 250 selected samples. Nguyễn Đức Minh
approved the group proposals and confirmed checking the review handoff. Applying 247 rankings updated
736 pairs (344 direction flips, 57 neutral pairs); 3 skipped samples retain their original labels.
The approval path records one approver and does not supply independent reviewer agreement.

On the same original test labels, r0/r1 accuracy is 56.72%/56.65%; on the same reviewed labels it is
57.04%/56.88%. Paired confidence intervals contain zero, so improvement is not established.
Control/conflict direction flip rates are 41.35%/57.97%; the stopping criteria are not met.
Labels, models, reports and approval records are checkpointed in the private HF dataset
`IntroDS/RankingResults`. Extended reward baselines and coordinator replacement flags were not used.

## Step 1. Score every pair (coordinator)

**Goal:** a scorer opinion on every pair, from a model that never saw that pair.

**Folds:**
- The scorer normally trains on the train split only, so its scores on train pairs are not honest.
- Instead, all samples (train, dev and test) are split into 5 folds by a hash of the sample ID.
- The folds are fixed, so a sample keeps its fold in every round.
- For each fold, the heads are trained on the other 4 folds and score the held-out fold.
- Early stopping uses 10% of the 4 training folds.

**Model:**
- The scorer is the selected configuration of the report: `mlp-siglip`, with 5 seeds per fold.
- Training uses the cached embeddings in `work/scorer/embeddings/`, so no images are read.
- It is light, but every run still needs the user's approval.

**Calibrated probabilities:**
- Raw scores are not probabilities.
- For each pair (a, b), `P_model(a better) = sigmoid((s_a - s_b) / T)`.
- `T` is one temperature fitted on all out-of-fold pairs to minimise the log loss against the labels.
- Only then can model confidence be compared with label confidence.

**Model uncertainty:** `spread` is the standard deviation of `P_model` over the 5 seeds.

**Second opinions (baselines):**
- EditReward, EditScore and ImageReward scores exist only for the test split.
- Plan:
  - ImageReward on every edit, about 2 h on one Kaggle T4;
  - EditReward only on the edits of the top 1,000 candidates from Step 2, a shortlist of about 3,600 edits and about 5 h on 2 T4.
- Both are Kaggle runs and need approval first.

**Command:** `python -m src.scorer oof --round NN` writes `oof_pairs.jsonl`. Each line holds a pair with its label `p`, `P_model`, `spread`, and every baseline that voted.

## Step 2. Select the samples to review (coordinator)

Each pair gets three signals. The label says "a better" when `p >= 0.5`, and `p_label` is the confidence of the label in that direction (`p` itself, at least 0.5).

| Signal | Meaning | Value |
|---|---|---|
| conflict | the model is sure the label is wrong | `P_model(opposite of the label)`, kept only when above 0.5; multiplied by `1 + 0.5 * k`, where k is the number of baselines that also take the opposite side |
| label doubt | the label itself is weak | `1 - 2 * abs(p - 0.5)`, plus 0.5 if the members disagreed on the pair, if the pair was a hard pair replaced by the upvote order, or if a member of the sample was replaced as noisy |
| model doubt | the model cannot decide | `1 - 2 * abs(P_model - 0.5)`, plus `spread` |

**From pairs to samples:**
- A sample's score for each signal is the maximum over its pairs.
- Pairs that involve the current top edit count double, because the assistant uses the top edit.

**One round, about 250 samples:**

| Bucket | Share | How it is filled |
|---|---|---|
| conflict | 50% | highest conflict |
| label doubt | 20% | highest label doubt, not already picked |
| model doubt | 10% | highest model doubt, not already picked |
| random control | 20% | uniform at random from all samples not reviewed yet |

**Rules:**
- **Random control.** It is what makes the round measurable: its flip rate estimates how many labels are wrong in the whole dataset, with no selection bias. Reviewers do not know which bucket a sample came from.
- **Splits.** Train, dev and test samples are all eligible, in proportion to their size (about 70/15/15). The test split is reviewed too, so the reported accuracy is measured on clean labels.
- **No repeats.** A sample reviewed in an earlier round is not picked again, unless the scorer still disagrees strongly with the reviewed label (conflict above 0.8).
- **Not reviewed:** samples that a member skipped, or that the filters removed.

**Command:** `python -m src.rank review-select --round NN --size 250` writes `selected.jsonl` (sample, bucket, signals) and `selection.md` (counts per bucket and split).

## Step 3. Rank again (reviewers)

### Setup

**Chunks:**
- The coordinator makes review chunks of about 25 samples: `python -m src.rank review-chunks --round NN --reviewers <comma-separated-roster>`.
- Every sample goes to **two reviewers**.
- At least one of the two must not be among the sample's original annotators.

**What the app shows in review mode:**
- the edits **in random order**, not by upvotes;
- no earlier ranking;
- no scorer opinion;
- no bucket name.

You rank exactly as in the first ranking: same app, same steps, and results sent back to the coordinator.

### How to judge

Use the same three criteria as the app, in this order:
1. **Does what was asked.** Read the request. An edit that ignores it loses, even if it looks nice.
2. **Looks natural and well made.** No artefacts, halos, smudges, odd colours or distorted faces.
3. **Keeps the rest of the photo.** Parts the request did not mention stay as they were.

Extra rules for the cases that caused the most disagreement:
- **Partial work.** An edit that does the request well loses to one that does all of it acceptably, but beats one that does it badly.
- **Over-editing.** Unrequested changes (filters, crops, extra objects) count against an edit, unless the request invites creativity ("make it funny", "do your best").
- **Joke edits.** On a serious request, a joke edit is ranked last. On a request that asks for humour, judge it as an edit.
- **Near-identical edits.** Still choose. Look at the zoomed image for the small differences: edges, faces, text.
- **Low resolution.** A smaller or blurrier output loses to a sharper one of the same content.
- **Unsure.** Rank anyway, then add the sample number and a short note to the round's shared "unsure" sheet. These samples are discussed in the round meeting. Skip only for the reasons listed in the app.

**Time:** about 1 to 2 minutes per sample. A round is about 500 reviews (250 samples x 2), so about 55 samples per member with 9 members.

### Calibration at the start of each round (15 minutes)

1. Everyone ranks the same 10 samples, taken from the round.
2. The coordinator shows the agreement.
3. The team discusses the samples with the most disagreement.
4. Decisions that set a rule are added to the case log below, so later rounds judge the same way.

### Settling disagreements

The two reviews of a sample are compared:

| Case | Decision | Label confidence |
|---|---|---|
| Same order | accepted | `p = 0.9` for every pair |
| Different order | discussed in the round meeting by the two reviewers and a third member, who decide one order and write a one-line reason | `p = 0.8` for every pair |
| Still undecided after the meeting | pairs the group could not order get `p = 0.5` (no preference) | the rest of the sample keeps the decided order |
| Both reviewers skipped | the sample is removed from the dataset | |

The decided order replaces the sample's earlier ranking.

**Command:** `python -m src.rank review-merge --round NN --results <returned-results-folder>` writes `reviews.jsonl` and `disagreements.md` (the list for the meeting).

## Step 4. Update, retrain, measure (coordinator)

**1. Apply the reviews.**
- `finalize --reviews <review-root>` reads completed `round-*/decisions.jsonl` and uses the decided order for those samples. On Kaggle, `review-apply --round NN` instead writes a versioned `pairs.jsonl` without moving images or needing the original work tree.
- `export` writes a new `data/pairs.jsonl`.
- Each round gets a dataset version: `labels-r0` (original), `labels-r1`, ...
- Every changed pair is logged in `work/review/round-NN/changes.jsonl` with the old order, the new order, the reviewers and the reason.

**2. Measure the round** (`work/review/round-NN/report.md`):
- **Flip rate:** the share of reviewed pairs whose label direction changed, overall and per bucket.
- **Control flip rate:** the flip rate of the random control. It estimates the error rate of the whole dataset.
- **Top-1 change rate:** the share of reviewed samples whose best edit changed.
- **Reviewer agreement:** pairwise agreement between the two reviewers, and Krippendorff's alpha. It should be clearly above the agreement of the original ranking (67.3% on test). If not, the rules need another calibration.
- **Scorer:** dev and test accuracy of the retrained scorer, on the new labels and on the original labels. Two scorers are always compared on the same label version.

**3. Retrain** the scorer on the new labels, then go back to Step 1.

### When to stop

Stop when **both** hold for the last round:
- the flip rate of the conflict bucket is below 10-15%, so the scorer's disagreements are now mostly the scorer's mistakes;
- the control flip rate is below 5%.

A hard limit of 5 rounds keeps the effort bounded. If the conflict flip rate stays high but the control flip rate is low, stop anyway: the remaining errors are rare.

## Case log

Decisions that set a rule are added here, with the round and a sample ID as an example.

| Round | Rule | Example sample |
|---|---|---|
| | | |

## Notes

- **Do not share images outside the team.** Many samples show identifiable people.
- **The test split is reviewed with the same procedure and buckets as the others.**
  - The random control and the "original labels" accuracy keep the comparison honest.
  - Reviewers never see which split a sample is in.
- **The upvote order is no longer used as a stand-in for reviewed samples.** The reviewed order replaces it.
