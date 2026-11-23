"""InstantIR (Huang et al., arXiv 2410.06551, 10/2024; Apache 2.0): blind image restoration on SDXL. A DINOv2-large
image encoder and an adapter (LQ-Adapter) read the degraded photo; at every step a LoRA "previewer" (one LCM step)
makes a restored preview, and an "aggregator" (a copy of the SDXL UNet encoder) feeds the photo and the preview to
the UNet as ControlNet-style residuals.

The repository has no package to install: its code is downloaded at COMMIT into EJ_MODELS_DIR and put on the path
(as for ObjectClear). Its requirements pin diffusers 0.28.1, transformers 4.36.2, peft 0.10.0 and torch 2.0.1, but
the code it imports at inference (pipelines/sdxl_instantir.py, module/aggregator.py, module/ip_adapter/,
schedulers/) only uses diffusers names that still exist in 0.35-0.39, and the public HF Space fffiloni/InstantIR runs
the unchanged code on diffusers 0.33.1, transformers 4.52.3, torch 2.8: so it runs in the main environment.

Weights: SDXL base 1.0 (fp16 variant) pinned to SDXL_REVISION, facebook/dinov2-large pinned to DINO_REVISION (the
repository's default `--vision_encoder_path` is a private path), and the three InstantIR files models/adapter.pt,
models/aggregator.pt, models/previewer_lora_weights.bin pinned to REVISION.

Settings of infer.py (the authors' script): the loading order adapter, previewer LoRA, scheduler DDPM with an LCM
single-step previewer scheduler, aggregator weights; 30 steps, guidance 7.0, preview_start 0.0, creative_start
(control_guidance_end) 1.0, the default timesteps (denoising_start 1000), batch size 1, and its resize_img: the short
side raised to at least 768, the long side capped at 1,024, both sides rounded down to a multiple of 64 (a 1024 x 683
photo is run at 1024 x 640). The prompt is the script's default positive and negative prompt (used when `--prompt`
is empty): InstantIR's prompt describes the content for "creative restoration", which our instruction ("Restore this
old photo") does not, so the instruction is ignored. fp16 on CUDA, as the script; the SDXL VAE is upcast to fp32 by
the pipeline itself (force_upcast). Deviations: the output is resized back by the runner (Lanczos) instead of the
script's bilinear resize, and the invisible watermark is turned off (the repository's requirements do not install
it, so the script runs without it too).
"""

import sys

from ..base import Editor
from .fetch import repository_code

REPO = "instantX-research/InstantIR"
COMMIT = "183ccf559446f472e4a10a276f0ecf908df4adb7"
MODEL = "InstantX/InstantIR"
REVISION = "7cbe3320352691cfa8c094811ebab32d3c5673ad"
SDXL = "stabilityai/stable-diffusion-xl-base-1.0"
SDXL_REVISION = "462165984030d82259a11f4367a4eed129e94a7b"
DINO = "facebook/dinov2-large"
DINO_REVISION = "47b73eefe95e8d44ec3623f8890bd894b6ea2d6c"
STEPS = 30
GUIDANCE = 7.0
# infer.py's default prompts (the same text as gradio_demo/app.py's PROMPT and NEG_PROMPT)
PROMPT = ("Photorealistic, highly detailed, hyper detailed photo - realistic maximum detail, 32k, ultra HD, "
          "extreme meticulous detailing, skin pore detailing, hyper sharpness, perfect without deformations, "
          "taken using a Canon EOS R camera, Cinematic, High Contrast, Color Grading. ")
NEGATIVE_PROMPT = ("blurry, out of focus, unclear, depth of field, over-smooth, sketch, oil painting, cartoon, "
                   "CG Style, 3D render, unreal engine, dirty, messy, worst quality, low quality, frames, painting, "
                   "illustration, drawing, art, watermark, signature, jpeg artifacts, deformed, lowres")


class InstantIR(Editor):
    name = "instantir"
    model_id = f"{MODEL}@{REVISION[:7]} on {SDXL}@{SDXL_REVISION[:7]}, {DINO}@{DINO_REVISION[:7]} (code {COMMIT[:7]})"
    tasks = ("restore",)
    uses_mask = False
    settings = {"size": "infer.py resize_img (short side >= 768, long side <= 1024, multiples of 64)",
                "steps": STEPS, "guidance": GUIDANCE, "preview_start": 0.0, "creative_start": 1.0,
                "scheduler": "DDPM, LCM single-step previewer", "prompt": "infer.py default"}

    def load(self, device: str) -> None:
        import torch
        from huggingface_hub import snapshot_download

        sys.path.insert(0, str(repository_code(REPO, COMMIT)))
        from diffusers import DDPMScheduler
        from infer import resize_img
        from module.ip_adapter.utils import load_adapter_to_pipe
        from pipelines.sdxl_instantir import InstantIRPipeline
        from schedulers.lcm_single_step_scheduler import LCMSingleStepScheduler

        self.torch, self.device, self.resize = torch, device, resize_img
        dtype = torch.float16 if device == "cuda" else torch.float32
        weights = snapshot_download(MODEL, revision=REVISION, allow_patterns=["models/*"])
        encoder = snapshot_download(DINO, revision=DINO_REVISION,
                                    allow_patterns=["config.json", "preprocessor_config.json", "model.safetensors"])
        # the fp16 files also on CPU (upcast on load): the fp32 ones are twice the download
        self.pipe = InstantIRPipeline.from_pretrained(SDXL, revision=SDXL_REVISION, variant="fp16",
                                                      torch_dtype=dtype, add_watermarker=False)
        load_adapter_to_pipe(self.pipe, f"{weights}/models/adapter.pt", encoder)
        self.pipe.prepare_previewers(f"{weights}/models")
        self.pipe.to(device=device, dtype=dtype)
        self.pipe.scheduler = DDPMScheduler.from_pretrained(SDXL, revision=SDXL_REVISION, subfolder="scheduler")
        self.previewer_scheduler = LCMSingleStepScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.aggregator.load_state_dict(torch.load(f"{weights}/models/aggregator.pt", map_location="cpu"))
        self.pipe.aggregator.to(device, dtype=dtype)
        self.pipe.set_progress_bar_config(disable=True)

    def edit(self, image, mask, prompt: str, seed: int = 0):
        working, _ = self.resize(image)
        generator = self.torch.Generator(device=self.device).manual_seed(seed)
        return self.pipe(prompt=[PROMPT], image=[working], num_inference_steps=STEPS, generator=generator,
                         negative_prompt=[NEGATIVE_PROMPT], guidance_scale=GUIDANCE,
                         previewer_scheduler=self.previewer_scheduler, preview_start=0.0,
                         control_guidance_end=1.0).images[0]
