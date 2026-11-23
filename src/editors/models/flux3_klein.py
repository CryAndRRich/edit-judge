"""FLUX 3 Image Klein (Black Forest Labs, 10/2026): a 5.6B flow-map transformer distilled for instruction editing,
with Qwen3-VL-4B-Instruct as its text encoder (MODEL_LIST.md list 1).

Source: hf:black-forest-labs/flux-3-image-klein at REVISION (gated: accept the FLUX Kommunity License on the page;
the token comes from the HF_TOKEN environment variable, which huggingface_hub reads by itself). Licence: FLUX
Kommunity License v1.0 (non-commercial use); the text encoder is Apache 2.0. Reference code:
gh:black-forest-labs/flux-3-image-klein (not used here: it pins torch 2.11, transformers 5.8.1 and vLLM, and
autocasts to bf16). Diffusers support (modular pipeline, `Flux3DistilledAutoBlocks`) first shipped in v0.42.0.

Download: only `transformer/` (11.10 GB), `text_encoder/` (8.88 GB, with its tokenizer) and `vae/` (0.34 GB) plus
`modular_model_index.json`; the repository also holds the same transformer and VAE again in the reference-code
layout (`model.safetensors`, `ae.safetensors`) and FP8 / NVFP4 variants (43 GB in all).

Settings (model card table and the distilled blocks of diffusers): 16 steps, no classifier-free guidance (the
distilled checkpoint samples a flow map; there is no guidance argument). Timestep shift: the scheduler config of
the pipeline (diffusers default 6.93; the reference code uses alpha 16.0: see the open points). The output has the
input's size (the pipeline reflect-pads the reference to a multiple of 16 px and crops the result back).

Deviations, forced by 2 x T4 (16 GB, no fast bf16):
- fp16 instead of bf16 for the transformer and the text encoder (bf16-trained: an overflow shows up as a black or
  NaN image, which the runner records as "black"; non-finite text embeddings fail the sample);
- split over the 2 GPUs: the text encoder on cuda:0, the transformer and the VAE (fp32, as the card loads it) on
  cuda:1. The text encoder is loaded outside the pipeline and its embeddings are passed as `prompt_embeds`,
  because a modular pipeline runs every block on one execution device;
- no prompt upsampler. The card says to always run it (a 9.4B Qwen3.5 model, 18.8 GB in bf16, in
  hf:black-forest-labs/flux-3-image-klein-pup) and that a short instruction "sent as typed ... reframes the shot".
  It does not fit next to the editor on 2 T4s, so the prompt is the simplified instruction as typed.
Mask-free: runs on every task, gets the instruction and no mask; its output is not pasted back.
"""

import json
from pathlib import Path

from .. import TASKS
from ..base import Editor

REPO = "black-forest-labs/flux-3-image-klein"
REVISION = "3b767ff9b5261d71645e2ec49d87d298d955238d"   # HF main on 2026-11-24 (last modified 2026-11-19)
FILES = ("modular_model_index.json", "transformer/*", "text_encoder/*", "vae/*")
DEFAULT_BLOCKS = "Flux3DistilledAutoBlocks"
TEXT_DEVICE, DENOISE_DEVICE = "cuda:0", "cuda:1"
STEPS = 16
LONG_SIDE = 1024        # the dataset's own limit; lower it first if cuda:1 runs out of memory


class Flux3Klein(Editor):
    name = "flux3-klein"
    model_id = f"{REPO}@{REVISION[:7]}"
    tasks = TASKS
    uses_mask = False
    install = ("diffusers==0.43.0 transformers==5.8.1",)   # FLUX.3 blocks: diffusers >= 0.42, transformers >= 5.2

    def __init__(self):
        self.settings = {"steps": STEPS, "guidance": None, "long_side": LONG_SIDE, "dtype": "float16",
                         "vae_dtype": "float32", "prompt_upsampler": False, "prompt": "instruction",
                         "devices": {"text_encoder": TEXT_DEVICE, "transformer": DENOISE_DEVICE}}

    def load(self, device: str) -> None:
        import diffusers
        import torch
        from diffusers import AutoencoderKLFlux2, Flux3Transformer2DModel, ModularPipeline
        from diffusers.modular_pipelines.flux3.encoders import Flux3DistilledTextEncoderStep
        from huggingface_hub import snapshot_download
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        if torch.cuda.device_count() < 2:
            raise RuntimeError("FLUX 3 Image Klein is split over 2 GPUs (text encoder and transformer)")
        self.torch = torch
        weights = snapshot_download(REPO, revision=REVISION, allow_patterns=list(FILES))
        index = json.loads((Path(weights) / "modular_model_index.json").read_text())
        blocks_name = index.get("_blocks_class_name") or DEFAULT_BLOCKS
        blocks = getattr(diffusers, blocks_name)() if hasattr(diffusers, blocks_name) else \
            ModularPipeline.from_pretrained(weights).blocks
        for step in ("prompt_upsampler", "text_encoder"):
            blocks.sub_blocks.pop(step, None)
        self.pipe = blocks.init_pipeline(weights)
        self.pipe.update_components(
            transformer=Flux3Transformer2DModel.from_pretrained(
                weights, subfolder="transformer", torch_dtype=torch.float16).to(DENOISE_DEVICE),
            vae=AutoencoderKLFlux2.from_pretrained(
                weights, subfolder="vae", torch_dtype=torch.float32).to(DENOISE_DEVICE))
        self.pipe.set_progress_bar_config(disable=True)

        self.text_encoder = Qwen3VLForConditionalGeneration.from_pretrained(
            weights, subfolder="text_encoder", dtype=torch.float16, attn_implementation="sdpa").to(TEXT_DEVICE).eval()
        self.processor = AutoProcessor.from_pretrained(weights, subfolder="text_encoder")
        self.encode_prompt = Flux3DistilledTextEncoderStep._get_qwen3_vl_prompt_embeds
        self.settings = {**self.settings, "blocks": type(blocks).__name__, "pipeline_class": index.get("_class_name"),
                         "shift": self.pipe.scheduler.config.get("shift")}

    def edit(self, image, mask, prompt: str, seed: int = 0):
        from PIL import Image

        if max(image.size) > LONG_SIDE:
            scale = LONG_SIDE / max(image.size)
            image = image.resize((round(image.width * scale), round(image.height * scale)), Image.Resampling.LANCZOS)
        with self.torch.no_grad():
            embeds = self.encode_prompt(text_encoder=self.text_encoder, tokenizer=self.processor, prompt=prompt,
                                        device=TEXT_DEVICE)
        if not self.torch.isfinite(embeds).all():
            raise RuntimeError("non-finite text embeddings (fp16 overflow in the text encoder)")
        generator = self.torch.Generator().manual_seed(seed)
        return self.pipe(image=image, prompt_embeds=embeds.to(DENOISE_DEVICE), num_inference_steps=STEPS,
                         generator=generator, output="images")[0]
