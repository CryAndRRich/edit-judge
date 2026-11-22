"""The request pipeline: image storage, upload validation, route -> run models -> save -> score -> rank, and
background jobs with per-model progress.

Storage (git-ignored, local only):
- web_app/uploads/<image id>.jpg: uploaded photos, normalised (EXIF rotation applied, RGB, at most 2048 px);
- web_app/outputs/<run id>/: original.jpg (the input at the working size), <model>.jpg|png, <model>.thumb.webp,
  result.json. Only the newest KEEP_RUNS runs are kept.
"""

import asyncio
import io
import json
import logging
import os
import re
import shutil
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError

from .catalog import MODELS_BY_ID, REPO_ROOT
from .model_runner import WORK_MAX_SIDE, ModelRunner, limit_size
from .router import ROUTER
from .scorer import ImageScorer, rank_candidates

log = logging.getLogger("web_app.pipeline")

WEB_ROOT = REPO_ROOT / "web_app"
UPLOAD_DIR = Path(os.environ.get("EJ_WEB_UPLOADS") or WEB_ROOT / "uploads")
OUTPUT_DIR = Path(os.environ.get("EJ_WEB_OUTPUTS") or WEB_ROOT / "outputs")
SAMPLES_DIR = REPO_ROOT / "dev_split"
MAX_UPLOAD_BYTES = 25 * 1024 * 1024
MAX_UPLOAD_PIXELS = 60_000_000
STORE_MAX_SIDE = 2048
KEEP_RUNS = int(os.environ.get("EJ_WEB_KEEP_RUNS", "40"))
ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP", "BMP", "TIFF", "GIF", "MPO"}
ID_RE = re.compile(r"^[0-9a-f]{32}$")

Image.MAX_IMAGE_PIXELS = MAX_UPLOAD_PIXELS


class PipelineError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status, self.message = status, message


# ---------------------------------------------------------------- images in and out

def save_upload(data: bytes, filename: str | None) -> dict:
    if not data:
        raise PipelineError(400, "the file is empty")
    if len(data) > MAX_UPLOAD_BYTES:
        raise PipelineError(413, f"the file is larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB")
    try:
        with Image.open(io.BytesIO(data)) as probe:
            fmt = probe.format
            probe.verify()                        # truncated or corrupt files fail here
        image = Image.open(io.BytesIO(data))
        if fmt not in ALLOWED_FORMATS:
            raise PipelineError(415, f"unsupported image format {fmt}; use JPEG, PNG, WebP, BMP, TIFF or GIF")
        image.seek(0)
        image = ImageOps.exif_transpose(image)
        if image.mode in ("RGBA", "LA", "P"):
            rgba = image.convert("RGBA")
            background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
            image = Image.alpha_composite(background, rgba)
        image = limit_size(image.convert("RGB"), STORE_MAX_SIDE)
    except PipelineError:
        raise
    except Image.DecompressionBombError:
        raise PipelineError(413, "the image has too many pixels")
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as error:
        raise PipelineError(415, f"not a readable image ({type(error).__name__})")
    if min(image.size) < 16:
        raise PipelineError(422, "the image is too small (at least 16 x 16 pixels)")
    image_id = uuid.uuid4().hex
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    path = UPLOAD_DIR / f"{image_id}.jpg"
    image.save(path, quality=94)
    safe_name = re.sub(r"[^\w.\- ]+", "_", Path(filename or "image").name)[:120]
    meta = {"image_id": image_id, "filename": safe_name, "width": image.width, "height": image.height,
            "bytes": len(data), "url": f"/media/uploads/{image_id}.jpg"}
    (UPLOAD_DIR / f"{image_id}.json").write_text(json.dumps(meta))
    return meta


def samples() -> list[dict]:
    manifest = SAMPLES_DIR / "manifest.json"
    if not manifest.exists():
        return []
    rows = []
    for row in json.loads(manifest.read_text()):
        if (SAMPLES_DIR / row["file"]).exists():
            rows.append({**row, "url": f"/media/samples/{row['file']}"})
    return rows


def resolve_image(image_id: str | None, sample_id: str | None) -> tuple[Path, str]:
    if image_id:
        if not ID_RE.match(image_id):
            raise PipelineError(404, "unknown image_id")
        path = UPLOAD_DIR / f"{image_id}.jpg"
        if not path.exists():
            raise PipelineError(404, "unknown image_id (upload the image again)")
        return path, f"/media/uploads/{image_id}.jpg"
    for row in samples():
        if row["id"] == sample_id:
            return SAMPLES_DIR / row["file"], row["url"]
    raise PipelineError(404, f"unknown sample_id {sample_id!r}")


def save_output(image: Image.Image, folder: Path, stem: str) -> tuple[str, str, bool]:
    transparent = image.mode == "RGBA"
    name = f"{stem}.png" if transparent else f"{stem}.jpg"
    if transparent:
        image.save(folder / name, optimize=True)
    else:
        image.convert("RGB").save(folder / name, quality=92)
    thumb = limit_size(image, 640)
    thumb.save(folder / f"{stem}.thumb.webp", quality=82)
    return name, f"{stem}.thumb.webp", transparent


def prune_runs() -> None:
    if not OUTPUT_DIR.exists():
        return
    runs = sorted((p for p in OUTPUT_DIR.iterdir() if p.is_dir()), key=lambda p: p.stat().st_mtime, reverse=True)
    for old in runs[KEEP_RUNS:]:
        shutil.rmtree(old, ignore_errors=True)


# ---------------------------------------------------------------- the pipeline

class Pipeline:
    def __init__(self, scorer: ImageScorer | None = None, runner: ModelRunner | None = None):
        self.scorer = scorer or ImageScorer()
        self.runner = runner or ModelRunner()

    def plan(self, instruction: str, models: list[str] | None) -> tuple:
        route = ROUTER.route(instruction)
        if models:
            unknown = [m for m in models if m not in MODELS_BY_ID]
            if unknown:
                raise PipelineError(422, f"unknown model IDs: {unknown}; see GET /api/models")
            model_ids = list(dict.fromkeys(models))
        else:
            model_ids = route.models
        return route, model_ids

    async def process(self, image_id: str | None, sample_id: str | None, instruction: str,
                      models: list[str] | None = None, progress=None, on_stage=None) -> dict:
        started = time.perf_counter()
        route, model_ids = self.plan(instruction, models)
        path, source_url = resolve_image(image_id, sample_id)

        def load():
            with Image.open(path) as image:
                return limit_size(ImageOps.exif_transpose(image).convert("RGB"), WORK_MAX_SIDE)

        original = await asyncio.to_thread(load)
        run_id = uuid.uuid4().hex
        folder = OUTPUT_DIR / run_id
        folder.mkdir(parents=True, exist_ok=True)
        await asyncio.to_thread(original.save, folder / "original.jpg", quality=94)
        loaded = time.perf_counter()

        results = await self.runner.run_models(original, route, model_ids, progress)
        ran = time.perf_counter()

        candidates = []
        for result in results:
            info = MODELS_BY_ID[result.model_id]
            candidate = {"model_id": info.id, "name": info.name, "date": info.date,
                         "categories": list(info.categories), "backend": result.backend, "status": result.status,
                         "elapsed_ms": round(result.elapsed_ms, 1), "vram_gb_estimate": info.weights_gb,
                         "note": result.note, "error": result.error, "benchmark": info.to_dict()["benchmark"],
                         "image": result.image}
            if result.image is not None:
                name, thumb, transparent = await asyncio.to_thread(save_output, result.image, folder, info.id)
                candidate.update(image_url=f"/media/outputs/{run_id}/{name}",
                                 thumb_url=f"/media/outputs/{run_id}/{thumb}", width=result.image.width,
                                 height=result.image.height, transparent=transparent)
            candidates.append(candidate)

        if on_stage:
            on_stage("scoring")
        await asyncio.to_thread(self.scorer.score_candidates, original, route, candidates)
        best, ranked = rank_candidates(candidates)
        scored = time.perf_counter()
        for candidate in ranked:
            candidate.pop("image", None)

        response = {
            "run_id": run_id, "instruction": instruction, "route": route.to_dict(),
            "original": {"url": f"/media/outputs/{run_id}/original.jpg", "source_url": source_url,
                         "width": original.width, "height": original.height,
                         "image_id": image_id, "sample_id": sample_id},
            "scorer": self.scorer.info(), "best": best, "candidates": ranked,
            "counts": {"models": len(ranked), "ok": sum(c["status"] == "ok" for c in ranked),
                       "failed": sum(c["status"] != "ok" for c in ranked),
                       "real": sum(c["backend"] == "real" for c in ranked),
                       "mock": sum(c["backend"] == "mock" for c in ranked),
                       "beat_original": sum((c.get("score") or {}).get("gain") is not None
                                            and c["score"]["gain"] > 0 for c in ranked)},
            "timings_ms": {"load": round((loaded - started) * 1000, 1), "models": round((ran - loaded) * 1000, 1),
                           "scoring": round((scored - ran) * 1000, 1), "total": round((scored - started) * 1000, 1)},
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        (folder / "result.json").write_text(json.dumps(response, ensure_ascii=False, indent=1))
        await asyncio.to_thread(prune_runs)
        log.info("run %s: %r, %d models (%d ok), winner %s, %.1f s", run_id, instruction[:60], len(ranked),
                 response["counts"]["ok"], best["model_id"] if best else None, (scored - started))
        return response


def load_run(run_id: str) -> dict:
    if not ID_RE.match(run_id):
        raise PipelineError(404, "unknown run_id")
    path = OUTPUT_DIR / run_id / "result.json"
    if not path.exists():
        raise PipelineError(404, "unknown run_id (runs are pruned after a while)")
    return json.loads(path.read_text())


# ---------------------------------------------------------------- background jobs

class Jobs:
    """In-memory jobs: POST creates one, GET polls its per-model progress and, at the end, the result."""
    MAX_JOBS = 200

    def __init__(self, pipeline: Pipeline):
        self.pipeline = pipeline
        self.jobs: dict[str, dict] = {}
        self.tasks: set[asyncio.Task] = set()

    def create(self, image_id, sample_id, instruction, models) -> dict:
        route, model_ids = self.pipeline.plan(instruction, models)
        resolve_image(image_id, sample_id)                      # 404 now rather than in the background
        job_id = uuid.uuid4().hex
        job = {"job_id": job_id, "status": "queued", "progress": {m: "queued" for m in model_ids},
               "done": 0, "total": len(model_ids), "error": None, "result": None, "created": time.time()}
        self.jobs[job_id] = job
        self._trim()

        def progress(model_id, state):
            job["progress"][model_id] = state
            job["done"] = sum(s in ("ok", "failed") for s in job["progress"].values())

        def stage(name):
            job["status"] = name

        async def work():
            job["status"] = "running"
            try:
                job["result"] = await self.pipeline.process(image_id, sample_id, instruction, model_ids, progress,
                                                            stage)
                job["status"] = "done"
            except PipelineError as error:
                job["status"], job["error"] = "error", error.message
            except Exception as error:
                log.exception("job %s failed", job_id)
                job["status"], job["error"] = "error", f"{type(error).__name__}: {error}"

        task = asyncio.create_task(work())
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)
        return {"job_id": job_id, "models": model_ids}

    def get(self, job_id: str) -> dict:
        job = self.jobs.get(job_id)
        if job is None:
            raise PipelineError(404, "unknown job_id")
        return {k: v for k, v in job.items() if k != "created"}

    def _trim(self):
        if len(self.jobs) > self.MAX_JOBS:
            for job_id in sorted(self.jobs, key=lambda j: self.jobs[j]["created"])[:len(self.jobs) - self.MAX_JOBS]:
                if self.jobs[job_id]["status"] in ("done", "error"):
                    del self.jobs[job_id]
