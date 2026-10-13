"""Prompt templates, one file per version: <name>_v<k>.txt in this folder. Never edit a version that has been used
for a run; add the next version instead, so every output can be traced to its exact prompt."""

from pathlib import Path

HERE = Path(__file__).resolve().parent


def load(name: str, version: int) -> str:
    return (HERE / f"{name}_v{version}.txt").read_text(encoding="utf-8")


def versions(name: str) -> list[int]:
    return sorted(int(p.stem.rsplit("_v", 1)[1]) for p in HERE.glob(f"{name}_v*.txt"))
