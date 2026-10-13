"""Settings and secrets from the .env file at the repository root (never committed; see .env.example).
Process environment variables take precedence, so Kaggle secrets can be passed as environment variables."""

import os
from pathlib import Path

from ..paths import ENV_FILE

PREFIXES = ("CRAWLER_", "PSG_", "REDDIT_", "EJ_", "HF_", "OPENAI_", "GEMINI_", "GOOGLE_", "ANTHROPIC_")


def load_env(path: Path = ENV_FILE) -> dict[str, str]:
    """Parse a minimal KEY=VALUE file; environment variables with a known prefix override it."""
    values: dict[str, str] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            values[key.strip()] = value.strip().strip('"').strip("'")
    values.update({k: v for k, v in os.environ.items() if k in values or k.startswith(PREFIXES)})
    return values


ENV = load_env()
