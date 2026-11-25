"""FastAPI application of the AI Image Editing Benchmark & Auto-Selection web app.

Pages: GET / (the app), GET /docs (OpenAPI).
API:
- GET  /api/health                 status, scorer backend, device
- GET  /api/models                 the 23 candidates and 7 categories
- GET  /api/samples                dev_split photos with their instructions
- POST /api/route                  {instruction} -> categories, intents, models (live preview)
- POST /api/upload                 multipart file -> {image_id, url, ...}
- POST /api/process                {image_id | sample_id, instruction, models?} -> ranked candidates (synchronous)
- POST /api/jobs, GET /api/jobs/{job_id}   the same as a background job with per-model progress
- GET  /api/runs/{run_id}          a finished run (shareable)
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

from .catalog import CATEGORIES, MODELS
from .pipeline import (MAX_UPLOAD_BYTES, OUTPUT_DIR, SAMPLES_DIR, UPLOAD_DIR, WEB_ROOT, Jobs, Pipeline,
                       PipelineError, load_run, samples, save_upload)
from .router import ROUTER
from .schemas import (JobCreated, JobStatus, ProcessRequest, ProcessResponse, RouteRequest, UploadResponse)

log = logging.getLogger("web_app")
VERSION = "1.0.0"


@asynccontextmanager
async def lifespan(app: FastAPI):
    for folder in (UPLOAD_DIR, OUTPUT_DIR):
        folder.mkdir(parents=True, exist_ok=True)
    app.state.pipeline = Pipeline()
    app.state.jobs = Jobs(app.state.pipeline)
    log.info("scorer: %s", app.state.pipeline.scorer.info())
    warmup = asyncio.create_task(asyncio.to_thread(app.state.pipeline.scorer.warmup))   # reward model, if any
    yield
    warmup.cancel()


app = FastAPI(title="Edit-Judge Studio", version=VERSION, lifespan=lifespan,
              description="Routes an editing instruction to candidate models, runs them, scores every edit "
                          "against the unedited input and picks the best.")
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
    return templates.TemplateResponse(request, "index.html", {"version": VERSION,
                                                              "scorer": request.app.state.pipeline.scorer.info()})


@app.get("/api/health")
async def health(request: Request):
    import torch
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    return {"status": "ok", "version": VERSION, "device": device, "models": len(MODELS),
            "samples": len(samples()), "scorer": request.app.state.pipeline.scorer.info()}


@app.get("/api/models")
async def models():
    return {"categories": [{"id": key, **value, "models": [m.id for m in MODELS if key in m.categories]}
                           for key, value in CATEGORIES.items()],
            "models": [model.to_dict() for model in MODELS]}


@app.get("/api/samples")
async def list_samples():
    return {"samples": samples()}


@app.post("/api/route")
async def route(body: RouteRequest):
    return ROUTER.route(body.instruction).to_dict()


@app.post("/api/upload", response_model=UploadResponse)
async def upload(file: UploadFile = File(...)):
    data = await file.read(MAX_UPLOAD_BYTES + 1)
    return await asyncio.to_thread(save_upload, data, file.filename)


@app.post("/api/process", response_model=ProcessResponse)
async def process(body: ProcessRequest, request: Request):
    return await request.app.state.pipeline.process(body.image_id, body.sample_id, body.instruction, body.models)


@app.post("/api/jobs", response_model=JobCreated, status_code=202)
async def create_job(body: ProcessRequest, request: Request):
    return request.app.state.jobs.create(body.image_id, body.sample_id, body.instruction, body.models)


@app.get("/api/jobs/{job_id}", response_model=JobStatus)
async def get_job(job_id: str, request: Request):
    return request.app.state.jobs.get(job_id)


@app.get("/api/runs/{run_id}", response_model=ProcessResponse)
async def get_run(run_id: str):
    return load_run(run_id)
