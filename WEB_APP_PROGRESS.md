# Web app: progress, architecture and how to run it

**Edit-Judge Studio** (`web_app/`) is the web front of the editing benchmark. It reads an editing instruction
(English or Vietnamese), routes it to the candidate models of `MODEL_LIST.md`, runs them, scores every edit against
the unedited photo, and shows the winner. The plan it implements is `web_app_development_plan.md`.

Status on 2026-11-23: all five phases of the plan are done, and all tests pass (see [Tests](#tests)).

## What is real and what is simulated

Read this before trusting a number in the UI.

| Part | Status on a laptop (Apple M-series, 8 GB) | On a CUDA machine with 16 GB or more |
|---|---|---|
| Router (7 lists, EN and VI) | real | real |
| Real-ESRGAN x4plus | **real**: spandrel, on MPS, about 4 s for a 512 px input | real (CUDA) |
| Plain code (PIL: crop, resize, rotate, flip, greyscale) | **real** | real |
| The other 21 editing models | **mock**: a Pillow/numpy simulation of what each model does | mock, until their weights are wired in |
| Scorer | **proxy** heuristic by default; **ImageReward** (the repository's wrapper) with `EJ_WEB_SCORER=imagereward` | **EditReward** (the repository's wrapper), picked by `auto` |

- Every candidate in the API and the UI carries `backend: "real" | "mock"` and a badge (REAL / MOCK).
- The scorer in use is shown in the header. When it is not EditReward, the hero banner says so and gives the reason.
- The **proxy is not EditReward**. It checks three things:
  - alignment: whether the requested change happened (brighter, sharper, background gone, rotated...);
  - quality change: exposure, contrast, sharpness, colourfulness, noise and clipping;
  - fidelity: whether the content was kept.

  It is a stand-in so the pipeline runs without a GPU. It must not be cited as a benchmark result.
- The T4 numbers shown for VOSR 2.0, HVI-CIDNet, InstructIR and Real-ESRGAN are real. They come from the Kaggle
  Enhance benchmark (`reports/enhance/stats.json`).

## How to run

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements_web.txt
.venv/bin/python scripts/fetch_dev_split.py      # 5 public photos (Picsum/Unsplash, COCO) into dev_split/
.venv/bin/python run_web.py                      # http://127.0.0.1:8000, API docs at /docs
```

- `run_web.py --port 8010` uses another port, and `--reload` restarts the server when a backend file changes.
- On the development machine, port 8000 was taken by another project, so the tests ran on 8010.
- The Real-ESRGAN weights (64 MB) are downloaded on first use into `web_app/.cache/` (git-ignored).

Settings (environment variables):

| Variable | Default | Meaning |
|---|---|---|
| `EJ_WEB_SCORER` | `auto` | `auto` (EditReward on CUDA with 16 GB or more, else the proxy), `editreward`, `imagereward`, `proxy`. A backend that cannot load falls back to the proxy, and the reason is shown. |
| `EJ_WEB_REAL_MODELS` | `1` | `0` replaces Real-ESRGAN with a Lanczos mock (no torch needed at run time) |
| `EJ_WEB_ESRGAN_MAX_SIDE` | `512` | Real-ESRGAN input size; the output is 4 times larger |
| `EJ_WEB_KEEP_RUNS` | `40` | Run folders kept in `web_app/outputs/` |
| `EJ_WEB_UPLOADS`, `EJ_WEB_OUTPUTS`, `EJ_WEB_CACHE` | under `web_app/` | Storage folders |

ImageReward on a laptop needs `pip install "transformers>=4.56,<5" timm`. On first use it downloads about 1.8 GB.

## Architecture

```
browser (templates/index.html, static/js/app.js, static/css/app.css: vanilla JS + CSS, no build step)
   │  POST /api/upload · POST /api/route (live preview) · POST /api/jobs → poll GET /api/jobs/{id}
   ▼
web_app/app/main.py         FastAPI: CORS, static + media mounts, Jinja2, JSON error handlers, scorer warm-up
web_app/app/pipeline.py     upload validation and storage · route → run → save → score → rank · jobs
   ├── router.py            TaskRouter: instruction → intents → categories → model IDs (EN + VI, accents optional)
   ├── catalog.py           the 23 candidates (MODEL_LIST.md) and the T4 benchmark of the Enhance models
   ├── model_runner.py      EditingModel interface · Real-ESRGAN (spandrel) · PIL · 21 mocks · thread pool
   ├── scorer.py            ImageScorer: EditReward / ImageReward (src/scorer/rewards) or the proxy · rank
   └── schemas.py           Pydantic request and response models (OpenAPI at /docs)
```

### One request, step by step

1. The photo is loaded and limited to 1024 px. All candidates and the scorer see this same image.
2. `TaskRouter.route`:
   - finds the intents (e.g. `brighten`, `remove_bg`, `rotate`);
   - maps them to categories;
   - picks the models: the shared list always runs, plus the specialists of each matched category, with the
     rules of `MODEL_LIST.md` (HVI-CIDNet on lighting requests only, cut-out models for background requests,
     DDColor to colourise only...).
3. `ModelRunner.run_models` runs the candidates in a thread pool:
   - 4 at a time;
   - torch models one at a time, behind a lock;
   - each run is timed;
   - one model failing never stops the others.
4. The outputs are saved:
   - JPEG, or PNG for cut-outs with transparency;
   - a WebP thumbnail of each.
5. `ImageScorer.score_candidates` scores the unedited photo and every edit. **gain = score(edit) − score(original)**,
   the "do nothing" baseline of the benchmark. `rank_candidates` sorts by gain; ties go to the faster model.
6. The response lists, best first, every candidate with:
   - gain, raw score and baseline;
   - the proxy breakdown;
   - latency, VRAM estimate (fp16 weights) and output size;
   - the T4 benchmark row, where there is one.

### Adding a real model

1. Subclass `EditingModel` in `model_runner.py`.
2. Set `backend = "real"` and implement `edit(image, route) -> PIL.Image`.
3. Register the class in `HANDLERS`.
4. Set `backend="real"` on the model in `catalog.py`.

Nothing else changes: routing, scoring, the API and the UI pick it up.

## API

| Method | Path | Body / result |
|---|---|---|
| GET | `/` | the app; `?run=<run_id>` reopens a run; `?sample=<id>&instruction=<text>&autorun=1` |
| GET | `/api/health` | status, device, model count, scorer backend and fallback reason |
| GET | `/api/models` | 7 categories and 23 models: backend, date, VRAM estimate, T4 benchmark |
| GET | `/api/samples` | dev_split photos, with instruction, source, author and licence |
| POST | `/api/route` | `{instruction}` → `{primary, intents, categories, params, models}` |
| POST | `/api/upload` | multipart `file`; see the upload rules below → `{image_id, url, width, height, filename, bytes}` |
| POST | `/api/process` | `{image_id` or `sample_id, instruction, models?}` → `ProcessResponse` (synchronous) |
| POST | `/api/jobs` | the same body → `202 {job_id, models}` |
| GET | `/api/jobs/{job_id}` | `{status: queued/running/scoring/done/error, progress: {model: state}, done, total, result}` |
| GET | `/api/runs/{run_id}` | a finished run (`web_app/outputs/<run_id>/result.json`) |
| GET | `/media/uploads/…`, `/media/outputs/…`, `/media/samples/…` | images |

Upload rules for `POST /api/upload`:
- formats: JPEG, PNG, WebP, BMP, TIFF and GIF, up to 25 MB and 60 MP;
- the photo is normalised before it is stored: EXIF rotation applied, converted to RGB, at most 2048 px.

Errors return `{"detail": "..."}`:

| Status | When |
|---|---|
| 400 | empty file |
| 404 | unknown image, sample, job or run |
| 413 | file too large |
| 415 | not a readable image |
| 422 | invalid body, empty instruction, unknown model |
| 500 | unexpected error (logged with its traceback) |

## Model routing

The shared list always runs. The specialists are added when one of their categories matches.

| Category (MODEL_LIST.md list) | Example triggers (EN / VI, with or without accents) | Models added |
|---|---|---|
| 1 Shared | always | FLUX 3 Image Klein, VAREdit 2B, FLUX.2 [klein] 4B, VIBE, OmniGen2, RealEdit, MagicBrush, InstructPix2Pix |
| 2 Remove | remove / erase / delete / watermark; xóa, loại bỏ, tẩy | ObjectClear, RORem, LaMa |
| 3 Add / Replace | add, put, replace, swap, turn X into, change the background; thêm, chèn, thay, đổi … thành | SDXL Inpainting 0.1 |
| 4 Enhance | brighter, sharpen, denoise, deblur, vivid, contrast, upscale; làm sáng, làm nét, khử nhiễu, độ phân giải | VOSR 2.0, HVI-CIDNet (lighting only), InstructIR, Real-ESRGAN |
| 5 Restore | restore, old photo, scratches, faded, face; phục hồi, ảnh cũ, phai màu, khuôn mặt | VOSR 2.0, InstantIR, CodeFormer and GFPGAN (faces / old photos) |
| 6 Change-Color | make the car red, bluer, colourise, black and white; đổi màu, tô màu, đen trắng | SDXL Inpainting (one region), DDColor (colourise only), PIL (greyscale) |
| 7 Image-Formatting | remove background, transparent, crop, rotate, flip, resize, extend; xóa phông nền, cắt ảnh, xoay, mở rộng | BEN2 and BiRefNet (cut-out), SDXL Inpainting (extend), Real-ESRGAN (upscale), PIL (geometry) |

### Ambiguous wordings

| Instruction | Routed as | Not as |
|---|---|---|
| "remove the noise / blur" | Enhance | Remove |
| "remove scratches" | Restore | Remove |
| "remove the background" | Formatting + Remove (as `MODEL_LIST.md` says for Remove Background) | object removal |
| "blur the background" | shared list only | deblurring |
| "add more contrast" | Enhance | Add |
| "colourise this black-and-white photo" | colourise | greyscale |
| "phai màu" (faded colours) | Restore | Change-Color |
| "90 độ sang phải" | rotation | "độ sáng" (brightness) |

The parameters (target colour, rotation angle and direction, size, scale, aspect ratio) are parsed from the
instruction.

## Tests

| Test | What it checks | Result |
|---|---|---|
| `tests/test_router.py` | 42 English and Vietnamese instructions (with and without accents), the parsed parameters, empty or garbage input | 42/42 |
| `tests/test_pipeline.py` | live server, real dev_split photos (details below) | 9/9 with the proxy (42 s); 9/9 with ImageReward (134 s) |
| `tests/ui_smoke.mjs` | headless Chrome through the DevTools protocol, no npm packages (details below) | 24/24 steps, 0 console or network errors |

`tests/test_pipeline.py` covers:
- an upload and a full run of every photo, and the whole response contract (ranks, sorting, finite gains, every
  image URL served);
- that the expected specialists win (proxy only);
- a model override, the job flow, and reloading a run;
- 16 error cases, including path traversal, a truncated JPEG and an oversized instruction;
- a transparent PNG and a 3000 px upload;
- 3 concurrent jobs.

`tests/ui_smoke.mjs` covers:
- an upload through the file input, a quick tag and the live routing preview;
- the run (live panel, then results), the compare slider, and the lightbox with the arrow keys and Esc;
- the filters and the sort;
- reloading a run from its URL;
- a 390 px phone layout with no horizontal overflow;
- the rejection of a file that is not an image.

```bash
.venv/bin/python tests/test_router.py
EJ_WEB_URL=http://127.0.0.1:8000 .venv/bin/python tests/test_pipeline.py
node tests/ui_smoke.mjs http://127.0.0.1:8000 /tmp/ej-ui
```

`tests/` is git-ignored in this repository (see `.gitignore`), so these files live on the coordinator's computer only.

## Iteration log (build → test → inspect → fix)

| # | Found by | Problem | Fix |
|---|---|---|---|
| 1 | router tests | `coloui?riz` required a "u", so "colorize" was missed | `colou?riz` |
| 2 | exploratory prompts | "90 độ sang phải" matched brightness; "add more contrast" became Add; "bluer" was unknown; "remove the person and the background" missed the cut-out | removed the colliding unaccented keyword, added rules and comparatives; the cases are now in the tests |
| 3 | runner smoke test | 4 models crashed: several threads decoded one lazily loaded PIL image | decode once, before the threads |
| 4 | runner smoke test | `asyncio.Semaphore` was bound to another event loop | a `ThreadPoolExecutor` |
| 5 | visual check of the outputs | the cut-out mask lost the pug's dark face; object removal hit a cat's head; outpainting showed seams | hole filling, centre-surround object finder, feathered composite |
| 6 | scorer calibration | black and white scored 0: the "unchanged" check compared greyscale thumbnails | compare in colour |
| 7 | scorer calibration | cut-outs and "brighter" were penalised for the change they were asked to make | skip the quality terms a request targets; fidelity measured on the subject (alpha or non-background) |
| 8 | scorer calibration | crop and extend checks counted pixels, so they were wrong for models that work at 512 px | scale-invariant checks (match against a centre crop of the original) |
| 9 | E2E timings | Real-ESRGAN took 35-50 s per request after the first one: MPS kept about 3 GB cached on an 8 GB Mac, so the next runs swapped | `torch.mps.empty_cache()` after each run: 3.6-4.6 s, test suite 168 s → 42 s |
| 10 | headless screenshots | thumbnails of cut-outs had no checkerboard (CSS specificity); the hero image was not centred; `?run=` left the controls empty | CSS fixes; the run's photo and instruction are restored |
| 11 | phone screenshot | the header wrapped onto 4 lines | compact header below 640 px |
| 12 | UX review | while models ran, the results column showed the welcome screen | live panel with per-model skeleton cards |
| 13 | ImageReward run | the first request waited about 80 s for the model to load | warm-up at server start; a model that cannot load falls back to the proxy at once |

## Known limits and next steps

- **Wire in the real models** on a GPU machine (Kaggle 2 × T4 or similar), in this order:
  - the cheap ones first: HVI-CIDNet, InstructIR, LaMa, BEN2, BiRefNet, DDColor;
  - then the diffusion editors.

  Their outputs are mocks until then.
- **Masks:**
  - Remove and region recolouring need Grounding DINO + SAM 2.1 masks (MODEL_LIST.md, Grounding).
  - The mocks use a saliency mask and a "most distinct region" guess instead, and say so in their note.
- **Scorer on a GPU:** run with `EJ_WEB_SCORER=auto` (EditReward). The proxy weights were calibrated by eye on the
  5 dev photos and were never fitted to human labels.
- **Jobs are in memory:** a server restart loses running jobs. Finished runs stay on disk (`/api/runs/{id}`).
