"""VIBE (Alekseenko et al., arXiv 2601.02242, 2026; code Apache 2.0, weights under the SANA licence): instruction
editing with a Sana1.5 1.6B diffusion transformer conditioned on Qwen3-VL-2B (the image and the instruction) through
learned meta-query tokens, and the DC-AE f32c32 VAE (MODEL_LIST.md list 1).

The `vibe` package (github.com/ai-forever/VIBE at COMMIT) has no dependencies in its pyproject; its requirements pin
torch 2.6, diffusers 0.33.1 and transformers 4.57.1. Qwen3-VL needs transformers >= 4.57 (Kaggle's main environment
may have 4.56) and the pipeline subclasses diffusers 0.33's SanaPipeline, so it gets its own environment that keeps
the machine's torch and overrides only diffusers and transformers; the package is installed with --no-deps.
The weights (iitolstykh/VIBE-Image-Edit at REVISION, 9.3 GB) name the transformer class by its module path
(`vibe.transformer.vibe_sana_editing`), so the package must be importable when the pipeline loads.

Settings of the authors' README and scripts/inference.py: 20 steps, guidance 4.5, image guidance 1.2 (three-way
classifier-free guidance), the default DPM-Solver++ flow scheduler; the prompt is the instruction (the pipeline wraps
it as "what will this image be like if {instruction}").

Deviations:
- fp16 instead of bf16 (T4): upstream `ImageEditor.__init__` hard-codes `self.weight_dtype = torch.bfloat16`
  (vibe/editor.py line 54) and passes it to `VIBESanaEditingPipeline.from_pretrained` in `get_generation_pipe`.
  This wrapper loads the pipeline itself with float16 instead of using ImageEditor (the same call, other dtype).
- Size: the pipeline works at the Sana aspect-ratio bin closest to the photo (the 512 bins up to 589,824 px, the
  1,024 bins above; a 1,024 x 768 photo runs at 1,152 x 832) and stretches the photo to it. ImageEditor then crops
  the padding of a letterbox the pipeline never applied and resizes back, which shifts the content by up to a few
  percent; here the output is returned at the bin size, as generated, and the runner resizes it back to the
  original size, which undoes the stretch exactly.
"""

from .. import TASKS
from ..base import Editor

REPO = "ai-forever/VIBE"
COMMIT = "7f0f01f9a6f66d55aa0fec2bf2562c332bba262b"
MODEL = "iitolstykh/VIBE-Image-Edit"
REVISION = "d670fc5220d9806dd95e3d27508aaa66e7415c45"
STEPS, GUIDANCE, IMAGE_GUIDANCE = 20, 4.5, 1.2
WEIGHTS = ["model_index.json", "scheduler/*", "text_encoder/*", "tokenizer/*", "transformer/*", "vae/*"]


class VIBE(Editor):
    name = "vibe"
    model_id = f"{MODEL}@{REVISION[:7]} (code {COMMIT[:7]})"
    tasks = TASKS
    uses_mask = False
    install = ("diffusers==0.33.1 transformers==4.57.1 loguru==0.7.3",
               f"--no-deps https://github.com/{REPO}/archive/{COMMIT}.tar.gz")
    settings = {"steps": STEPS, "guidance": GUIDANCE, "image_guidance": IMAGE_GUIDANCE, "size": "sana bin",
                "dtype": "float16", "prompt": "instruction"}

    def load(self, device: str) -> None:
        import torch
        from huggingface_hub import snapshot_download
        from vibe.generative_pipeline import VIBESanaEditingPipeline

        self.torch, self.device = torch, device
        weights = snapshot_download(MODEL, revision=REVISION, allow_patterns=WEIGHTS)   # not the example images
        self.pipe = VIBESanaEditingPipeline.from_pretrained(
            weights, torch_dtype=torch.float16 if device == "cuda" else torch.float32).to(device)
        self.pipe.set_progress_bar_config(disable=True)

    def edit(self, image, mask, prompt: str, seed: int = 0):
        height, width = self.pipe.get_closest_size(image.height, image.width)
        generator = self.torch.Generator(device=self.device).manual_seed(seed)
        with self.torch.inference_mode():
            return self.pipe(conditioning_image=image, prompt=prompt, height=height, width=width,
                             num_inference_steps=STEPS, guidance_scale=GUIDANCE,
                             image_guidance_scale=IMAGE_GUIDANCE, num_images_per_prompt=1, output_type="pil",
                             generator=generator).images[0]
