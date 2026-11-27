"""FastAPI application of EditJudge Studio, the live photo-editing assistant.

Pages: GET / (the app: tabs Assistant, Rankings, Dataset), GET /docs (OpenAPI).
API:
- GET  /api/health                 status, device, scorer, backend (real or mock) and, on the GPU server, what is
                                   loaded and the free GPU memory
- GET  /api/samples                sample photos (apps/web/dev_split)
- POST /api/upload                 multipart file -> {image_id, url, ...}
- POST /api/jobs                   {image_id | sample_id, instruction, task?} -> the job: the request read, the
                                   models picked (fastest first), each queued
- GET  /api/jobs/{job_id}          the job: per-model status, elapsed seconds, output and score as they come
- GET  /api/rankings               the benchmark per task (../docs/editors.md, scored run) and which models run live
- GET  /api/dataset                the figures of the Dataset tab (data/dataset.json) and the model count
Media: /media/uploads, /media/outputs, /media/samples; static files: /static.
"""

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .assistant import rankings_view
from .catalog import MODELS, dataset_stats
from .pipeline import (MAX_UPLOAD_BYTES, OUTPUT_DIR, SAMPLES_DIR, UPLOAD_DIR, WEB_ROOT, Jobs, PipelineError, samples,
                       save_upload)
from .schemas import JobRequest, UploadResponse

log = logging.getLogger("studio")
VERSION = "2.0.0"


@asynccontextmanager
async def lifespan(app: FastAPI):
    for folder in (UPLOAD_DIR, OUTPUT_DIR):
        folder.mkdir(parents=True, exist_ok=True)
    app.state.jobs = Jobs()
    log.info("scorer: %s", app.state.jobs.scorer.info())
    jobs = app.state.jobs
    if jobs.live is not None:                     # the GPU server: load Qwen, the detectors and the scorer now
        def warm():                               # Qwen, the detectors, the scorer, then the editors
            from .assistant import downloadable_models
            jobs.live.warm()
            backend = jobs.scorer.backend
            if backend is not None and hasattr(backend, "load"):
                try:
                    backend.load()
                except Exception:
                    log.exception("loading the scorer failed")
            jobs.live.warm_editors(downloadable_models())
        app.state.warming = asyncio.get_running_loop().run_in_executor(None, warm)
    yield


app = FastAPI(title="EditJudge Studio", version=VERSION, lifespan=lifespan,
              description="Reads an editing request, runs the best-ranked models of its task one after another, "
                          "and scores every edit.")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=WEB_ROOT / "static"), name="static")
app.mount("/media/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")
app.mount("/media/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")
app.mount("/media/samples", StaticFiles(directory=SAMPLES_DIR), name="samples")
templates = Jinja2Templates(directory=WEB_ROOT / "templates")


@app.exception_handler(PipelineError)
async def pipeline_error(request: Request, error: PipelineError):
    return JSONResponse(status_code=error.status, content={"detail": error.message})


@app.exception_handler(RequestValidationError)
async def validation_error(request: Request, error: RequestValidationError):
    messages = []
    for item in error.errors():
        where = ".".join(str(part) for part in item.get("loc", []) if part != "body")
        messages.append(f"{where}: {item.get('msg')}" if where else item.get("msg"))
    return JSONResponse(status_code=422, content={"detail": "; ".join(messages)})


@app.exception_handler(Exception)
async def unexpected_error(request: Request, error: Exception):
    log.exception("unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(status_code=500, content={"detail": f"internal error: {type(error).__name__}"})


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def index(request: Request):
    return templates.TemplateResponse(request, "index.html", {"version": VERSION})


@app.get("/api/health")
async def health(request: Request):
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    except ImportError:
        device = "cpu (no torch)"
    jobs = request.app.state.jobs
    return {"status": "ok", "version": VERSION, "device": device, "models": len(MODELS), "samples": len(samples()),
            "scorer": jobs.scorer.info(), "backend": "real" if jobs.live is not None else "mock",
            "live": jobs.live.info() if jobs.live is not None else None}


@app.get("/api/samples")
async def list_samples():
    return {"samples": samples()}


@app.post("/api/upload", response_model=UploadResponse)
async def upload(file: UploadFile = File(...)):
    data = await file.read(MAX_UPLOAD_BYTES + 1)
    return await asyncio.to_thread(save_upload, data, file.filename)


@app.post("/api/jobs", status_code=202)
async def create_job(body: JobRequest, request: Request):
    return request.app.state.jobs.create(body.image_id, body.sample_id, body.instruction, body.task)


@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str, request: Request):
    return request.app.state.jobs.get(job_id)


@app.get("/api/rankings")
async def get_rankings(request: Request):
    live_models = request.app.state.jobs.live
    # on the GPU server: the wait for one edit now, as measured here (the edit, plus the load when the editor is
    # in RAM or on disk); elsewhere the benchmark's seconds times EJ_WEB_SPEEDUP
    measured = {} if live_models is None else {
        name: live_models.estimate(name) for name in live_models.editors.stats.rows if live_models.estimate(name)}
    return rankings_view(measured)


@app.get("/api/dataset")
async def get_dataset():
    return dataset_stats()
