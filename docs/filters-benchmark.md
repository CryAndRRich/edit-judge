# Image filters: benchmark of NSFW and watermark detectors

First comparison of the detectors for PROJECT.md Part II Step 1b (sensitive images) and Step 1d (watermarks). All
models are pretrained checkpoints, used as they are (no training). The code now lives in `src/filters/`
(`nsfw/`, `watermark/`, `benchmark.py`); the runs below were made with the members' own notebooks and scripts,
which stay on their branches:

| Run | Models | Where | Original code and full outputs |
|---|---|---|---|
| NSFW run A (LMD) | Falconsai, Owen Elliott, NeMo Curator | Kaggle, 1 x T4 | branch `benchmark/nsfw-three-models` (1fdc214), `benchmarks/nsfw/nsfw-image-classification-v2.ipynb`, `Results_LMD.md` |
| NSFW run B (NDM) | NudeNet-320n, OpenNSFW2, MobileNetV2 | Kaggle, 1 x T4 | same branch, `benchmarks/nsfw/Nude.ipynb`, `Results_NDM.md` |
| Watermark (T-024) | YOLOv8n, EfficientNetV2-B0, ConvNeXt-tiny, majority vote | laptop, RTX 4050 | branch `watermark-model` (24f797d), `experiments/T024/` |

To run the models again with the shared code, on these and more community sets (see "Community evaluation sets"):
`notebooks/filters_nsfw_benchmark.ipynb` and `filters_watermark_benchmark.ipynb` (both T4 GPUs; per-image scores
are kept, so every voting method and subset of models gets evaluated).

## NSFW

**Data.** deepghs/nsfw_detect (`nsfw_dataset_v1.zip`), 400 images per class, the same 2,000 images in both runs
(paths sorted, seed 42). Positive (NSFW): `porn`, `hentai`, `sexy` (1,200); negative: `neutral`, `drawings` (800).
Threshold 0.5 for every model.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | AP | ms / image | Run |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| NeMo Curator (CLIP ViT-L/14 + NSFW scorer) | **0.973** | 0.985 | **0.970** | **0.977** | **0.997** | **0.998** | 67.1 (batch 64) | A |
| MobileNetV2 (GantMan 1.2.0) | 0.921 | 0.951 | 0.916 | 0.933 | 0.979 | 0.986 | **8.3** (batch 1) | B |
| Owen Elliott (timm) | 0.887 | 0.897 | 0.916 | 0.906 | 0.949 | 0.959 | 12.4 (batch 32) | A |
| Falconsai (ViT) | 0.893 | **0.991** | 0.828 | 0.902 | 0.976 | 0.986 | 15.6 (batch 32) | A |
| OpenNSFW2 (Yahoo) | 0.823 | 0.960 | 0.735 | 0.833 | 0.943 | 0.963 | 195.5 (batch 1) | B |
| NudeNet-320n | 0.678 | 0.979 | 0.473 | 0.638 | 0.770 | 0.822 | 17.3 (batch 1) | B |

Share of images flagged NSFW, per class (for SFW classes lower is better, for NSFW classes higher is better):

| Class | NeMo | MobileNetV2 | Owen | Falconsai | OpenNSFW2 | NudeNet |
|---|---:|---:|---:|---:|---:|---:|
| neutral (SFW) | 0.015 | 0.078 | 0.063 | 0.003 | 0.028 | 0.028 |
| drawings (SFW) | 0.030 | 0.065 | 0.253 | 0.020 | 0.065 | 0.003 |
| sexy (NSFW) | 0.988 | 0.963 | 0.820 | 0.580 | 0.535 | 0.283 |
| hentai (NSFW) | 0.923 | 0.815 | 0.965 | 0.930 | 0.773 | 0.428 |
| porn (NSFW) | 1.000 | 0.970 | 0.963 | 0.975 | 0.898 | 0.710 |

![NSFW run A: confusion matrices](figures/filters/nsfw_run_a_confusion.png)
![NSFW run A: ROC and precision-recall](figures/filters/nsfw_run_a_roc_pr.png)
![NSFW run B: confusion matrices](figures/filters/nsfw_run_b_confusion.png)
![NSFW run B: ROC and precision-recall](figures/filters/nsfw_run_b_roc_pr.png)

**Findings.**
- NeMo Curator is the most accurate (54 errors of 2,000) but the slowest of the PyTorch models; MobileNetV2 is
  second (F1 0.933) and the fastest.
- `sexy` separates the models most: NudeNet only finds exposed body parts, and Falconsai misses most `sexy`
  images at 0.5 (its F1-optimal threshold is 0.006, which raises its F1 to 0.944, measured on the same images).
- Owen Elliott flags 25% of drawings.
- No vote of several NSFW models was evaluated (the runs kept no per-image scores in the repository).

**Limits.** The label convention (`sexy` and `hentai` are NSFW) favours models trained the same way; NudeNet's
low recall reflects a different definition (exposed parts), not only a weaker model. Speeds come from different
batch sizes and timing setups, so they compare only within a run. No confidence intervals; the set may overlap
the models' training data.

## Watermarks

**Data.** boomb0om/watermarks-validation (pinned revision d2d2270): 122 images, 62 watermark and 60 clean. It is
the evaluation set of the ConvNeXt authors, so it may favour that model. A synthetic set (60 images with text
marks added to 20 clean backgrounds of the same set) was also used; it is not independent and is not ported.
Threshold 0.5; batch 1; FP32.

| Model | Accuracy | Precision | Recall | F1 | FPR | ms / image (p50) | Checkpoint |
|---|---:|---:|---:|---:|---:|---:|---:|
| ConvNeXt-tiny (boomb0om) | **0.934** | **0.982** | **0.887** | **0.932** | **0.017** | **15.5** | 113 MB |
| Majority vote of the three | 0.705 | 0.933 | 0.452 | 0.609 | 0.033 | 77.5 (sum) | 144 MB |
| EfficientNetV2-B0 (qwertyforce) | 0.607 | 0.769 | 0.323 | 0.455 | 0.100 | 26.7 | 24 MB |
| YOLOv8n (qfisch) | 0.582 | 0.824 | 0.226 | 0.354 | 0.050 | 34.7 | 6 MB |

On the synthetic set the order is the same (ConvNeXt F1 0.677, the others 0.14-0.22).

**Findings.** ConvNeXt-tiny is the best and the fastest model. The majority vote is far worse than ConvNeXt
alone (F1 -0.32) and costs about five times the time: the two weak models outvote it.

**Limits.** 122 images, possibly seen by the ConvNeXt authors; EfficientNet's fine-tuning transform is unknown
(the 192 px centre crop from its checkpoint may cut marks near the edges); laptop GPU, not Kaggle.

## Detectors and where they come from

Besides the members' models, the code now has the baselines that PROJECT.md plans. None of these tasks has a
widely used academic benchmark that we know of, and both evaluation sets above are community sets, so this report
says so plainly and adds the detectors that have a paper behind them.

| Task | Detector (`--model`) | Origin | Measured |
|---|---|---|---|
| NSFW | `opennsfw2` | Yahoo Open-NSFW (2016 release, no paper); **the filter of RealEdit (CVPR 2025)**, our main reference | run B |
| NSFW | `nemo` | **LAION's CLIP-based NSFW detector** (an MLP on CLIP ViT-L/14 embeddings), used to filter LAION-5B (NeurIPS 2022), run through NVIDIA NeMo Curator | run A |
| NSFW | `clip` | **CLIP zero-shot** (ICML 2021): NSFW prompts against safe prompts, same ViT-L/14 encoder as `nemo` | not yet |
| NSFW | `falconsai`, `owen`, `mobilenet`, `nudenet` | community checkpoints (Hugging Face, GitHub) | runs A and B |
| Watermark | `laion` | **LAION's watermark classifier** (EfficientNet-B3a), used to filter LAION-5B, MIT licence | not yet |
| Watermark | `clip` | **CLIP zero-shot**: watermarked against clean prompts | not yet |
| Watermark | `ocr` | **OCR text detection**: EasyOCR with the CRAFT detector (CVPR 2019); flags any text, a lower bound | not yet |
| Watermark | `convnext`, `efficientnet`, `yolo` | community checkpoints | T-024 |

## Against the plan (PROJECT.md Step 1b and 1d)

Hand labels of our own images were dropped from the plan on 2026-11-06 (time): models, votes and thresholds are
chosen on the public sets, and on our data only the share of flagged images is reported.

| Plan item | NSFW | Watermark |
|---|---|---|
| Several detectors, including the planned baselines | done in code: 7 (opennsfw2, NudeNet, ViT, CLIP zero-shot as planned) | done in code: 6 (LAION, OCR, CLIP zero-shot as planned) |
| Measured on a benchmark | done: all 7 on 3 sets (2026-11-06) | done: all 6 on 3 sets (2026-11-06) |
| Single models against votes of every subset (mean, weighted, any, majority) | done | done |
| Run on every original and edit of the dataset | next (`filters_apply`: nemo + falconsai) | next (`filters_apply`: convnext) |
| Drop flagged edits and their pairs; the sample if the original is flagged or fewer than 2 edits remain | code ready (`python -m src.rank export`) | same |

## Benchmark with the shared code (2026-11-06)

`notebooks/filters_nsfw_benchmark.ipynb` and `filters_watermark_benchmark.ipynb`, Kaggle 2 x T4, every model on
every image of 3 community sets per task (sets: next section), threshold 0.5. Latency: one image (batch 1) after a
warm-up, decoding excluded; throughput: batched, ms per image. Outputs (per-image scores, `evaluation.json`,
`evaluation.md`) are in `work/filters/<task>/benchmark/`.

**Check of the ported code.** On the sets of the first benchmarks the scores match the members' runs: NSFW on
`deepghs` F1 nemo 0.977 (first run 0.9773), mobilenet 0.933 (0.9329), owen 0.906, falconsai 0.902, opennsfw2
0.833, nudenet 0.638; watermark on `boomb0om` convnext 0.932 (0.9322), efficientnet 0.455 (0.4545), yolo 0.359
(0.354, batch 16 instead of 1).

### NSFW (5,352 images, 2,552 positive)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Latency p50 / p95 (ms) | Throughput (ms/img) |
|---|---:|---:|---:|---:|---:|---:|---:|
| nemo (LAION CLIP NSFW head) | **0.951** | 0.949 | 0.948 | **0.948** | **0.990** | 69.9 / 75.7 | 59.7 |
| mobilenet | 0.901 | 0.915 | 0.872 | 0.893 | 0.959 | 42.2 / 62.0 | 38.1 |
| owen | 0.888 | 0.888 | 0.876 | 0.882 | 0.953 | 19.2 / 33.8 | 14.4 |
| falconsai | 0.877 | 0.970 | 0.766 | 0.856 | 0.955 | 17.2 / 20.7 | 16.3 |
| opennsfw2 (RealEdit's filter) | 0.850 | 0.951 | 0.723 | 0.821 | 0.944 | 291.2 / 315.2 | 54.3 |
| clip (zero-shot) | 0.688 | 0.605 | **0.997** | 0.753 | 0.943 | 77.0 / 106.9 | 59.8 |
| nudenet | 0.746 | **0.980** | 0.476 | 0.641 | 0.777 | 17.0 / 29.9 | 21.7 |

F1 (accuracy) and ROC-AUC by set; false positive rate (FPR) on the clean images of each set:

| Model | deepghs F1 | nsfw_xl F1 | unsafebench F1 | AUC deepghs / nsfw_xl / unsafebench | FPR deepghs / nsfw_xl / unsafebench |
|---|---:|---:|---:|---|---|
| nemo | 0.977 | 0.978 | 0.772 | 0.997 / 0.998 / 0.938 | 0.022 / 0.015 / 0.098 |
| mobilenet | 0.933 | 0.930 | 0.636 | 0.979 / 0.979 / 0.833 | 0.071 / 0.061 / 0.088 |
| owen | 0.906 | 0.932 | 0.622 | 0.949 / 0.982 / 0.844 | 0.158 / 0.083 / 0.073 |
| falconsai | 0.902 | 0.905 | 0.482 | 0.976 / 0.985 / 0.753 | 0.011 / 0.014 / 0.038 |
| opennsfw2 | 0.833 | 0.877 | 0.585 | 0.943 / 0.975 / 0.837 | 0.046 / 0.029 / 0.030 |
| clip | 0.848 | 0.801 | 0.486 | 0.922 / 0.960 / 0.978 | 0.527 / 0.494 / 0.745 |
| nudenet | 0.638 | 0.738 | 0.280 | 0.770 / 0.836 / 0.624 | 0.015 / 0.008 / 0.005 |

Best votes at 0.5 by F1: clip + falconsai + nemo (mean) F1 0.955, nemo + owen (mean) 0.955; F1 gain over nemo
alone +0.007 for 1.2-2.3 times the time.

### Watermark (4,122 images, 2,062 positive)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Latency p50 / p95 (ms) | Throughput (ms/img) |
|---|---:|---:|---:|---:|---:|---:|---:|
| clip (zero-shot) | **0.721** | 0.690 | **0.801** | **0.742** | 0.815 | 64.7 / 68.0 | 48.8 |
| laion | 0.709 | 0.892 | 0.476 | 0.621 | 0.765 | 20.7 / 24.8 | 7.2 |
| convnext | 0.710 | 0.968 | 0.435 | 0.601 | **0.868** | **8.8 / 9.2** | 9.7 |
| ocr | 0.674 | **0.970** | 0.359 | 0.524 | 0.804 | 50.6 / 79.3 | 58.2 |
| efficientnet | 0.564 | 0.855 | 0.155 | 0.262 | 0.602 | 18.1 / 21.2 | 6.3 |
| yolo | 0.572 | 0.954 | 0.152 | 0.262 | 0.677 | 17.7 / 29.0 | 17.8 |

| Model | boomb0om F1 | scenery F1 | wm_or_not F1 | AUC boomb0om / scenery / wm_or_not | FPR boomb0om / scenery / wm_or_not |
|---|---:|---:|---:|---|---|
| clip | 0.662 | 0.760 | 0.729 | 0.746 / 0.835 / 0.810 | 0.750 / 0.342 / 0.355 |
| laion | 0.791 | 0.356 | 0.798 | 0.813 / 0.674 / 0.890 | 0.267 / 0.042 / 0.061 |
| convnext | 0.932 | 0.200 | 0.837 | 0.971 / 0.794 / 0.951 | 0.017 / 0.015 / 0.014 |
| ocr | 0.611 | 0.264 | 0.711 | 0.846 / 0.723 / 0.882 | 0.067 / 0.008 / 0.011 |
| efficientnet | 0.455 | 0.100 | 0.386 | 0.667 / 0.528 / 0.665 | 0.100 / 0.024 / 0.024 |
| yolo | 0.359 | 0.047 | 0.429 | 0.768 / 0.575 / 0.796 | 0.033 / 0.005 / 0.008 |

Best vote at 0.5 by F1: clip + convnext + ocr (max) F1 0.787, +0.046 over clip alone, but its false positive rate
is 0.37.

### Choice for the dataset

The benchmark sets are about half positive; in our data NSFW images and watermarks are rare (RealEdit, 92% of the
samples, was already filtered with opennsfw2). So the false positive rate decides, not F1 at 0.5: every clean
image flagged loses its pairs, and a flagged original loses the whole sample. Candidates at several thresholds
(FPR over all clean benchmark images; the last column is FPR x 56,568, an upper estimate of clean images flagged):

| Task | Rule | Threshold | Precision | Recall | F1 | FPR | FPR, real photos | Clean images flagged (est.) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| NSFW | nemo | 0.5 | 0.949 | 0.948 | 0.948 | 0.047 | 0.098 | ~2,650 |
| NSFW | nemo | 0.9 | 0.977 | 0.859 | 0.914 | 0.019 | 0.045 | ~1,050 |
| NSFW | mean nemo + owen | 0.7 | 0.983 | 0.857 | 0.915 | 0.014 | 0.015 | ~770 |
| NSFW | mean nemo + falconsai | 0.5 | 0.978 | 0.870 | 0.921 | 0.018 | 0.031 | ~1,010 |
| **NSFW** | **mean nemo + falconsai** | **0.7** | **0.992** | **0.759** | **0.860** | **0.006** | **0.011** | **~320** |
| Watermark | clip | 0.5 | 0.690 | 0.801 | 0.742 | 0.360 | 0.342 | ~20,400 |
| Watermark | max convnext + ocr | 0.5 | 0.956 | 0.545 | 0.694 | 0.025 | 0.022 | ~1,430 |
| Watermark | mean convnext + ocr | 0.5 | 0.986 | 0.408 | 0.578 | 0.006 | 0.004 | ~330 |
| **Watermark** | **convnext** | **0.5** | **0.968** | **0.435** | **0.601** | **0.015** | **0.015** | **~830** |

"Real photos" are `unsafebench` (LAION photos) for NSFW and `scenery` (r/EarthPorn photos) for watermarks.

- **NSFW: mean of nemo and falconsai, flagged at 0.7** (`configs/filters.toml`). Precision 0.992 and FPR 0.6% at a
  recall of 0.76; about 76 ms per image together (about 20 min on the dataset with both GPUs). nemo alone at 0.5
  has the best F1 but would flag about 10% of real clean photos.
- **Watermark: convnext alone at 0.5.** The only model good on two sets (AUC 0.95-0.97) with FPR 1.5%, and the
  fastest (8.8 ms). It catches stock-photo and text marks but misses most small photographers' signatures (recall
  0.11 on `scenery`; no model reaches a usable FPR there). OCR is left out despite its low FPR on the benchmark:
  it flags any text, and our edits often contain wanted text (memes, captions, "add text" requests).
- Every model's scores are kept, so after the run on the dataset the threshold can be changed without running the
  models again (`python -m src.filters vote --task T --threshold X`).

Other findings: CLIP zero-shot ranks well (NSFW AUC 0.92-0.98) but its 0.5 threshold is badly calibrated (half of
the clean images flagged); efficientnet is weak even on `scenery`, its own training set (AUC 0.53), which points to
a preprocessing different from its fine-tuning (the card does not state it); NudeNet only finds exposed body
parts, hence its low recall on `sexy` and `unsafebench`.

## Community evaluation sets (added 2026-11-06)

Verified on the Hugging Face Hub and pinned to a commit (`src/filters/benchmark.py`); at most 1,000 images per class.

| Task | Set | Source | Images | Note |
|---|---|---|---|---|
| NSFW | `deepghs` | deepghs/nsfw_detect | 2,000 (5 classes) | the first benchmark's sample; GantMan-style classes, so MobileNetV2 may have seen similar data |
| NSFW | `nsfw_xl` | markandey14/nsfw_detection_xl (MIT) | 2,000 | sources not stated; the nsfw side includes drawn and anime images |
| NSFW | `unsafebench` | yiting/UnsafeBench (gated) | up to 2,000 | real LAION-5B images only, Sexual vs safe; LAION's NSFW tags came from the `nemo` head, so `nemo` may be favoured |
| Watermark | `boomb0om` | boomb0om/watermarks-validation | 122 | the ConvNeXt authors' evaluation set |
| Watermark | `scenery` | qwertyforce/scenery_watermarks (CC BY-NC 4.0) | 2,000 | real photographers' marks on r/EarthPorn photos; the training set of `efficientnet` (not a fair test for it) |
| Watermark | `wm_or_not` | prithivMLmods/Watermark-or-Not-20K (Apache-2.0) | 2,000 | synthetic text marks, some very faint |

Rejected: sets that are the training data of a checkpoint we test (e.g. bastienp/visible-watermark-pita for
`yolo`), sets with one class only, AI-generated sets, and sets needing the owner's manual approval.

## Next runs (on Kaggle, after approval)

1. Done (2026-11-06): both benchmark notebooks, all 13 detectors on 3 sets per task.
2. `notebooks/filters_apply.ipynb` with the choice above (nsfw: nemo + falconsai, mean, 0.7; watermark: convnext,
   0.5) on the 56,568 dataset images; download `filter_decisions.zip` (decisions and every model's scores).
3. On the coordinator's computer: put its folders into `work/filters/`, check the counts (images flagged,
   samples dropped) and, if needed, vote again with another threshold; then `python -m src.rank export` (drops
   flagged edits and samples, reports the counts for the data-loss funnel), `python -m src.process pack`, and
   upload the new version of the Kaggle dataset.
