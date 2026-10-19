"""The interface every VLM client implements."""

from pathlib import Path


class Client:
    name: str = ""          # short unique name; also the output folder in data/instructions/
    model_id: str = ""      # Hugging Face ID, with revision when pinned
    batch_size: int = 1

    def load(self, device: str) -> None:
        """Load the model onto `device`. Called once before generating."""
        raise NotImplementedError

    def generate(self, prompt: str, image: Path | None = None, max_tokens: int = 256) -> str:
        """The model's answer to `prompt`, looking at `image` (greedy decoding, so runs are repeatable)."""
        raise NotImplementedError

    def generate_batch(self, prompts: list[str], images: list[Path | None], max_tokens: int = 256) -> list[str]:
        """Answers for several requests at once; clients that can batch override this."""
        return [self.generate(p, i, max_tokens) for p, i in zip(prompts, images)]
