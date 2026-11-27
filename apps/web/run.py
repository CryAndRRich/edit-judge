"""Starts the web app (apps/web/studio/) with Uvicorn on http://127.0.0.1:8000.

Usage: python apps/web/run.py [--host 127.0.0.1] [--port 8000] [--reload]
       python apps/web/run.py --check [--models a,b]   # GPU server: every live editor once on a sample photo
       python apps/web/run.py --precompute             # GPU server: answer the 5 sample requests ahead of time
Settings (environment): EJ_WEB_BACKEND=auto|real|mock (real: the models on CUDA, live.py), EJ_WEB_SCORER=auto|
mlp-siglip|editreward|imagereward|proxy, EJ_WEB_EDITORS_GB=24, EJ_WEB_SPEEDUP, EJ_WEB_KEEP_RUNS=40; on a laptop
also EJ_WEB_REAL_MODELS=0 (mock Real-ESRGAN too) and EJ_WEB_ESRGAN_MAX_SIDE=512.
Requirements: pip install -r requirements/web.txt (laptop) or requirements/server.txt (GPU server; README).
"""

import argparse
import logging
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description="EditJudge Studio web app")
    parser.add_argument("--host", default=os.environ.get("EJ_WEB_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("EJ_WEB_PORT", "8000")))
    parser.add_argument("--reload", action="store_true", help="restart when a backend file changes (development)")
    parser.add_argument("--log-level", default="info")
    parser.add_argument("--check", action="store_true",
                        help="GPU server: load every live editor once, edit a sample photo, print time and memory")
    parser.add_argument("--models", help="--check: comma-separated editors (default: every live editor)")
    parser.add_argument("--precompute", action="store_true",
                        help="GPU server: run the sample photos' requests now, so the demo shows them at once")
    args = parser.parse_args()

    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))
    logging.basicConfig(level=args.log_level.upper(), format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
                        datefmt="%H:%M:%S")
    from studio.live import server_defaults
    server_defaults()                     # GPU server: caches in work/hf_cache of the repository
    if args.check:
        check(args.models.split(",") if args.models else None)
        return
    if args.precompute:
        precompute()
        return
    import uvicorn
    print(f"EditJudge Studio on http://{args.host}:{args.port}  (API docs: /docs)", flush=True)
    uvicorn.run("studio.main:app", host=args.host, port=args.port, reload=args.reload,
                reload_dirs=[str(ROOT / "studio")] if args.reload else None, log_level=args.log_level)


def check(models: list[str] | None) -> None:
    import json

    from PIL import Image

    from studio.assistant import downloadable_models
    from studio.live import check_editors, cuda_available
    from studio.pipeline import OUTPUT_DIR

    if not cuda_available():
        raise SystemExit("--check runs the editors on the GPU server (CUDA)")
    photos = sorted((ROOT / "dev_split").glob("*.jpg"))
    if not photos:
        raise SystemExit("no sample photo in apps/web/dev_split/ (python apps/web/fetch_dev_split.py)")
    with Image.open(photos[0]) as image:
        image = image.convert("RGB")
        image.thumbnail((1024, 1024))
    names = models or downloadable_models()
    print(f"checking {len(names)} editors on {photos[0].name} ({image.width} x {image.height})", flush=True)
    results = check_editors(names, image)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "check.json").write_text(json.dumps(results, indent=1))
    failed = [row["model"] for row in results if row["status"] != "ok"]
    print(f"{len(results) - len(failed)}/{len(results)} ok; failed: {', '.join(failed) or 'none'}; "
          f"details in {OUTPUT_DIR / 'check.json'}")


def precompute() -> None:
    """Every sample photo with its request (dev_split/manifest.json) through the whole pipeline once: the answers
    go to the result cache (live.ResultCache), so on the demo they come back at once."""
    import asyncio
    import time

    from studio.pipeline import Jobs, samples

    async def run_all() -> None:
        jobs = Jobs()
        if jobs.live is None:
            raise SystemExit("--precompute runs the real models on the GPU server (CUDA)")
        jobs.live.warm()
        for sample in samples():
            started = time.monotonic()
            job = jobs.create(None, sample["id"], sample["instruction"])
            while job["status"] not in ("done", "error"):
                await asyncio.sleep(0.5)
                job = jobs.get(job["job_id"])
            best = next((c for c in job["candidates"] if c.get("best")), None)
            print(f"{sample['id']:24} {job['status']:5} {time.monotonic() - started:6.1f} s  "
                  f"{len(job['candidates'])} models, best {best['model'] if best else '-'}  "
                  f"({job['understood']['english']})", flush=True)

    asyncio.run(run_all())


if __name__ == "__main__":
    main()
