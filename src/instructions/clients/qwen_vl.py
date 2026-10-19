"""Qwen2.5-VL (open weights, Apache 2.0 for the 3B and 7B models) through Hugging Face transformers.

Runs on a Kaggle GPU: bfloat16 where the GPU supports it, else float16 (T4, P100). The 7B model needs about 17 GB,
so on Kaggle it is spread over 2 x T4 (device_map="auto"); the 3B model fits one T4, so 2 x T4 run two shards.
Images are limited to MAX_PIXELS so that one request stays a few hundred visual tokens.
"""

from pathlib import Path

from .base import Client

MAX_PIXELS = 512 * 28 * 28      # about 640 x 640 px; Qwen2.5-VL splits images into 28 x 28 patches
MIN_PIXELS = 64 * 28 * 28


class QwenVL(Client):
    model_id = ""
    batch_size = 8

    def load(self, device: str) -> None:
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        if device == "cuda":
            dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        else:
            dtype = torch.float32
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_id, dtype=dtype, device_map="auto" if device == "cuda" else None).eval()
        if device != "cuda":
            self.model.to(device)
        # pixel limits as "size" (min_pixels= and max_pixels= are deprecated)
        self.processor = AutoProcessor.from_pretrained(
            self.model_id, size={"shortest_edge": MIN_PIXELS, "longest_edge": MAX_PIXELS})
        self.processor.tokenizer.padding_side = "left"      # needed for batched generation

    def generate(self, prompt: str, image: Path | None = None, max_tokens: int = 256) -> str:
        return self.generate_batch([prompt], [image], max_tokens)[0]

    def generate_batch(self, prompts: list[str], images: list[Path | None], max_tokens: int = 256) -> list[str]:
        import torch
        from PIL import Image

        texts, pixels = [], []
        for prompt, image in zip(prompts, images):
            content = ([{"type": "image"}] if image else []) + [{"type": "text", "text": prompt}]
            texts.append(self.processor.apply_chat_template([{"role": "user", "content": content}],
                                                            tokenize=False, add_generation_prompt=True))
            if image:
                with Image.open(image) as im:
                    pixels.append(im.convert("RGB"))
        inputs = self.processor(text=texts, images=pixels or None, padding=True, return_tensors="pt")
        inputs = inputs.to(self.model.device)
        with torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
        out = out[:, inputs["input_ids"].shape[1]:]
        return self.processor.batch_decode(out, skip_special_tokens=True)


class Qwen25VL3B(QwenVL):
    name = "qwen2.5-vl-3b"
    model_id = "Qwen/Qwen2.5-VL-3B-Instruct"


class Qwen25VL7B(QwenVL):
    name = "qwen2.5-vl-7b"
    model_id = "Qwen/Qwen2.5-VL-7B-Instruct"
    batch_size = 4
