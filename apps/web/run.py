"""Starts the web app (apps/web/web_app/) with Uvicorn on http://127.0.0.1:8000.

Usage: python apps/web/run.py [--host 127.0.0.1] [--port 8000] [--reload]
Settings (environment): EJ_WEB_SCORER=auto|editreward|imagereward|proxy, EJ_WEB_REAL_MODELS=0 (mock Real-ESRGAN
too), EJ_WEB_ESRGAN_MAX_SIDE=512, EJ_WEB_KEEP_RUNS=40. Requirements: pip install -r requirements/web.txt
"""

import argparse
import logging
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description="Edit-Judge Studio web app")
    parser.add_argument("--host", default=os.environ.get("EJ_WEB_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("EJ_WEB_PORT", "8000")))
    parser.add_argument("--reload", action="store_true", help="restart when a backend file changes (development)")
    parser.add_argument("--log-level", default="info")
    args = parser.parse_args()

    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))
    logging.basicConfig(level=args.log_level.upper(), format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
                        datefmt="%H:%M:%S")
    import uvicorn
    print(f"Edit-Judge Studio on http://{args.host}:{args.port}  (API docs: /docs)", flush=True)
    uvicorn.run("web_app.app.main:app", host=args.host, port=args.port, reload=args.reload,
                reload_dirs=[str(ROOT / "web_app" / "app")] if args.reload else None, log_level=args.log_level)


if __name__ == "__main__":
    main()
