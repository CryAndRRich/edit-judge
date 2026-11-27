"""The assistant's request flow: image storage and upload validation, then one background job per request:
read the request (assistant.understand), pick the models (assistant.pick_models), run them one after another, fastest
first, and score each output as soon as it is ready. The browser polls GET /api/jobs/{id} and draws the progress.

Storage (git-ignored, local only):
- apps/web/studio/uploads/<image id>.jpg: uploaded photos, normalised (EXIF rotation applied, RGB, at most 2048 px);
- apps/web/studio/outputs/<job id>/: original.jpg (the input at the working size), <model>.jpg|png,
  <model>.thumb.webp, result.json. Only the newest KEEP_RUNS jobs are kept.

On the GPU server (live.py, CUDA) the job first reads the request with Qwen2.5-VL-7B and plans and masks the edit
(the job's "stage": reading, then nothing), then the real editors run; on a laptop the mocks of model_runner.py run.

Score: our scorer mlp-siglip on the GPU server, the proxy heuristic of scorer.py on a laptop (EJ_WEB_SCORER);
shown as exp(gain), so it is above 0 and higher is better.
"""

import asyncio
import io
import json
import logging
import os
import re
import math
import shutil
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError

from . import live
from .assistant import pick_models, understand
from .catalog import WEB_DIR
from .model_runner import WORK_MAX_SIDE, ModelRunner, limit_size
from .router import ROUTER
from .scorer import ImageScorer

log = logging.getLogger("studio.pipeline")

WEB_ROOT = WEB_DIR / "studio"
UPLOAD_DIR = Path(os.environ.get("EJ_WEB_UPLOADS") or WEB_ROOT / "uploads")
OUTPUT_DIR = Path(os.environ.get("EJ_WEB_OUTPUTS") or WEB_ROOT / "outputs")
SAMPLES_DIR = WEB_DIR / "dev_split"
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


# ---------------------------------------------------------------- jobs

def display_score(candidate: dict) -> float | None:
    gain = (candidate.get("score") or {}).get("gain")
    return round(math.exp(gain), 2) if gain is not None and math.isfinite(gain) else None


class Jobs:
    """In-memory jobs: POST creates one, GET polls it. A job is {job_id, status (running, done, error),
    understood, original, candidates: [{model, name, status (queued, running, ok, failed), estimate, elapsed, ...}]}.
    A job waits ("waiting") while another one uses the models, then runs ("running")."""
    MAX_JOBS = 200

    def __init__(self, scorer: ImageScorer | None = None, runner: ModelRunner | None = None):
        # auto: mlp-siglip on the GPU server, the proxy elsewhere; EJ_WEB_SCORER=imagereward|editreward still work
        self.live = (runner.live if runner is not None else
                     live.LiveModels() if live.use_real() else None)
        self.scorer = scorer or ImageScorer(os.environ.get("EJ_WEB_SCORER", "auto"))
        self.runner = runner or ModelRunner(self.live)
        self.results = live.ResultCache() if self.live is not None and live.CACHE_RESULTS else None
        self.jobs: dict[str, dict] = {}
        self.tasks: set[asyncio.Task] = set()
        self.gpu = asyncio.Lock()               # one job on the models at a time: the server has one GPU for them

    def create(self, image_id: str | None, sample_id: str | None, instruction: str, task: str | None = None) -> dict:
        path, source_url = resolve_image(image_id, sample_id)          # 404 now rather than in the background
        understood = understand(instruction, task)
        picked = pick_models(understood.task, understood.operations)
        if self.live is not None:                   # the times measured on this GPU, once there are some
            for row in picked:
                row["estimate"] = self.live.estimate(row["model"]) or row["estimate"]
        job_id = uuid.uuid4().hex
        job = {"job_id": job_id, "status": "waiting", "understood": understood.to_dict(),
               "original": {"source_url": source_url, "image_id": image_id, "sample_id": sample_id},
               "candidates": [{"model": row["model"], "name": row["name"], "benchmark": row["score"],
                               "estimate": row["estimate"], "status": "queued", "started": None, "elapsed": None}
                              for row in sorted(picked, key=lambda row: row["estimate"])],      # fastest first
               "scorer": self.scorer.info(), "stage": None, "error": None, "created": time.time(),
               "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        self.jobs[job_id] = job
        self._trim()
        task_handle = asyncio.create_task(self._work(job, path, understood))
        self.tasks.add(task_handle)
        task_handle.add_done_callback(self.tasks.discard)
        return self.get(job_id)

    async def _work(self, job: dict, path: Path, understood) -> None:
        try:
            def load():
                with Image.open(path) as image:
                    return limit_size(ImageOps.exif_transpose(image).convert("RGB"), WORK_MAX_SIDE)

            original = await asyncio.to_thread(load)
            folder = OUTPUT_DIR / job["job_id"]
            folder.mkdir(parents=True, exist_ok=True)
            await asyncio.to_thread(original.save, folder / "original.jpg", quality=94)
            job["original"].update(url=f"/media/outputs/{job['job_id']}/original.jpg", width=original.width,
                                   height=original.height)
            route = ROUTER.route(understood.english)
            async with self.gpu:
                job["status"] = "running"
                context = key = None
                if self.live is not None:
                    job["stage"] = "reading"
                    if self.results is not None:
                        key = live.ResultCache.key(folder / "original.jpg", understood.instruction, understood.task)
                        context = await asyncio.to_thread(self.results.load_context, key)
                        if context is not None:     # answered before: replay the reading as long as it took
                            await asyncio.sleep(sum(context.seconds.values()))
                    if context is None:
                        try:
                            context = await asyncio.to_thread(self.live.prepare, folder / "original.jpg", original,
                                                              understood)
                            if key:
                                await asyncio.to_thread(self.results.save_context, key, context)
                        except Exception:       # the editors without a region can still run on the request
                            log.exception("job %s: reading the request failed", job["job_id"])
                            context = live.EditContext(instruction=understood.english,
                                                       op=live.sample_op(understood.task, understood.operations))
                    job["understood"].update(english=context.instruction, target=context.target(),
                                             prepare_seconds=context.seconds)
                    job["stage"] = None
                for candidate in job["candidates"]:
                    await self._run(job, candidate, original, route, folder, context, key)
            self._rank(job, final=True)
            job["status"] = "done"
            (folder / "result.json").write_text(json.dumps(self.public(job), ensure_ascii=False, indent=1))
            await asyncio.to_thread(prune_runs)
            log.info("job %s: %r -> %s, %d models, best %s", job["job_id"], understood.instruction[:60],
                     understood.task, len(job["candidates"]), next((c["model"] for c in job["candidates"]
                                                                     if c.get("best")), None))
        except Exception as error:
            log.exception("job %s failed", job["job_id"])
            job["status"], job["error"] = "error", f"{type(error).__name__}: {error}"

    async def _run(self, job: dict, candidate: dict, original, route, folder: Path, context=None,
                   key: str | None = None) -> None:
        """One model: its output from the result cache when this request was answered before, else a run and a
        score (then cached)."""
        candidate["status"], candidate["started"] = "running", time.time()
        model = candidate["model"]
        tag = live.output_tag(model, self.scorer.info()["backend"]) if key else None
        cached = await asyncio.to_thread(self.results.load_output, key, model, tag) if key else None
        if cached is not None:                       # answered before: no model runs, the bar runs as long as the
            image, meta = cached                     # model took, then the cached edit shows
            candidate.update(estimate=meta["seconds"], cached=True)
            await asyncio.sleep(meta["seconds"])
            candidate.update(seconds=meta["seconds"], backend="real", error=None)
            score = meta["score"]
        else:
            result = await asyncio.to_thread(self.runner.run_one, model, original, route, candidate["estimate"],
                                             context)
            candidate.update(seconds=round(result.elapsed_ms / 1000, 2), backend=result.backend,
                             error=result.error)
            image, score = (result.image if result.status == "ok" else None), None
            if image is not None:
                scored = {"image": image}
                await asyncio.to_thread(self.scorer.score_candidates, original, route, [scored],
                                        context.instruction if context else None)
                score = display_score(scored)
                if key and score is not None:
                    await asyncio.to_thread(self.results.save_output, key, model, tag, image,
                                            {"seconds": candidate["seconds"], "score": score})
        if image is not None:
            name, thumb, transparent = await asyncio.to_thread(save_output, image, folder, model)
            candidate.update(image_url=f"/media/outputs/{job['job_id']}/{name}",
                             thumb_url=f"/media/outputs/{job['job_id']}/{thumb}", width=image.width,
                             height=image.height, transparent=transparent, score=score)
        candidate["status"] = "ok" if image is not None and candidate.get("score") is not None else "failed"
        self._rank(job)

    @staticmethod
    def _rank(job: dict, final: bool = False) -> None:
        """Places by score among the finished models (ties: the faster first); `best` once every model ran."""
        done = sorted((c for c in job["candidates"] if c["status"] == "ok"), key=lambda c: (-c["score"], c["seconds"]))
        for place, candidate in enumerate(done, start=1):
            candidate["rank"] = place
            if final and place == 1:
                candidate["best"] = True

    def public(self, job: dict) -> dict:
        now = time.time()
        data = {k: v for k, v in job.items() if k != "created"}
        data["candidates"] = []
        for candidate in job["candidates"]:
            row = {k: v for k, v in candidate.items() if k != "started"}
            if candidate["status"] == "running" and candidate["started"]:
                row["elapsed"] = round(now - candidate["started"], 2)
            data["candidates"].append(row)
        return data

    def get(self, job_id: str) -> dict:
        job = self.jobs.get(job_id)
        if job is None:
            raise PipelineError(404, "unknown job_id")
        return self.public(job)

    def _trim(self):
        if len(self.jobs) > self.MAX_JOBS:
            for job_id in sorted(self.jobs, key=lambda j: self.jobs[j]["created"])[:len(self.jobs) - self.MAX_JOBS]:
                if self.jobs[job_id]["status"] in ("done", "error"):
                    del self.jobs[job_id]
