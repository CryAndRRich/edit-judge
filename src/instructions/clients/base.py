"""The interface every LLM/VLM client implements."""

import json
import re
from pathlib import Path


class Client:
    name: str = ""          # short unique name; also the output folder in data/instructions/
    model_id: str = ""      # Hugging Face ID or API model name, with version

    def load(self, device: str) -> None:
        """Load a local model, or check the API key. Called once before generate()."""

    def generate(self, prompt: str, images: list[Path] | None = None, max_tokens: int = 512) -> str:
        """The model's text answer to `prompt`, optionally looking at `images`."""
        raise NotImplementedError


def parse_json(text: str) -> dict:
    """The first JSON object in a model answer (models often wrap it in prose or code fences)."""
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        raise ValueError("no JSON object in the answer")
    return json.loads(match.group(0))
