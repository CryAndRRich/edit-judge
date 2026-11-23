# Enhance benchmark (MODEL_LIST.md, list 4) — progress

Runner: `notebook_runner.py` (self-contained Kaggle script; `kernel-metadata.json`, kernel
`stevenpaulli/enhance-models-benchmark`, input dataset `cryandrrich/editjudge`). Samples: dev split, category
`Enhance*`, fixed hash order, seed 0, inputs shrunk to a long side of 1024 (never enlarged).

## Smoke run (5 samples) — 2026-11-22, Kaggle T4

Run in two versions: v1 (all 5 models) and v2 (VOSR only, after adding `fairscale`). Outputs in `work/editors/smoke/`.

| Model | Status | ok / 5 | s per image (T4) | Peak GPU (GB) | Black / NaN | Notes |
|---|---|---|---|---|---|---|
| Real-ESRGAN x4plus | ok | 5 | 4.7-9.7 (first image 9.7) | 0.8 | none | x4 output shrunk to input size; fp16, tiles of 256 |
| HVI-CIDNet | ok | 1 (4 skipped) | 1.3 | 1.2 | none | only the low-light instruction ("Remove shadows and brighten boots.") ran; brightens strongly (mean 132 -> 188) |
| InstructIR | ok | 5 | 0.4-1.2 | 0.8-1.1 | none | fp32 |
| VOSR 2.0 | ok | 5 | 7.2-25.4 (first image 25.4) | 8.2-8.8 | none | fp32, tile 512; load 103 s, 7.0 GB |
| AceTone | **unsupported** | 0 | | | | released checkpoint (PST-Preview) needs a reference photo, not an instruction |

No out-of-memory errors, no black or non-finite outputs. VOSR v1 failed only on a missing `fairscale` package.

## Open points

## Decisions (2026-11-22, coordinator)
- AceTone removed from the Enhance benchmark and listed under "Left out" in MODEL_LIST.md: `AceTone-3B-PST-Preview`
  needs a reference photo for style transfer and does not take a text instruction.
- Full run approved for the other four models; scorer = ImageReward and EditReward (repository reward models).

## Full run (30 dev-split Enhance samples) — Kaggle T4, kernel version 3
| Model | ok / failed / skipped | black / non-finite | s per image | first image (s) | load (s) | peak GPU (GB) | ImageReward gain [95% CI], win rate | EditReward gain [95% CI], win rate |
|---|---|---|---|---|---|---|---|---|
| Real-ESRGAN x4plus | 30 / 0 / 0 | 0 / 0 | 5.34 | 15.4 | 9 | 0.83 | -0.008 [-0.074, +0.065], 50% (n=30) | +0.029 [-0.038, +0.101], 53% (n=30) |
| HVI-CIDNet | 7 / 0 / 23 | 0 / 0 | 0.33 | 1.8 | 5 | 1.15 | +0.113 [-0.012, +0.275], 86% (n=7) | -0.210 [-0.469, +0.057], 14% (n=7) |
| InstructIR | 30 / 0 / 0 | 0 / 0 | 0.51 | 1.4 | 37 | 1.30 | +0.031 [-0.011, +0.078], 67% (n=30) | -0.100 [-0.177, -0.036], 23% (n=30) |
| VOSR 2.0 | 30 / 0 / 0 | 0 / 0 | 10.10 | 25.7 | 87 | 9.10 | +0.037 [-0.021, +0.107], 60% (n=30) | +0.110 [+0.035, +0.189], 77% (n=30) |

All four models: no failed sample, no black or non-finite output, no out-of-memory error. HVI-CIDNet ran on its 7
low-light samples and skipped 23 by design. Scores: kernel `stevenpaulli/enhance-models-scoring` (127 items: 97 outputs
and 30 unedited inputs as the baseline), no null score. Only VOSR 2.0 (+0.110) and InstructIR (-0.100) on EditReward have
a 95% interval that excludes zero. Details, charts and limits: `ENHANCE_BENCHMARK_REPORT.md`.
