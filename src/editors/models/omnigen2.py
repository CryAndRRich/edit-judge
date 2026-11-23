"""OmniGen2 (Wu et al., 2025; Apache 2.0): a unified generation and editing model. A Qwen2.5-VL-3B MLLM encodes the
instruction, a 4B diffusion transformer (Lumina-2 style, flow matching) generates from it and from the VAE latents
of the input image. MODEL_LIST.md list 1 (Shared): it runs on every task, mask-free, with the instruction as prompt.

The repository has no package to install and diffusers support is not finished, so its code is downloaded at COMMIT
into EJ_MODELS_DIR and put on the path (as ObjectClear). Only torch, diffusers, transformers, accelerate and einops
are needed on the inference path; the torch 2.6 / transformers 4.51.3 pins of its requirements.txt are for training
and are not installed: the model runs with the machine's packages (einops is the only one that may be missing).

Built as in the repository's inference.py: `OmniGen2Pipeline.from_pretrained` with the transformer loaded by the
repository's `OmniGen2Transformer2DModel` class. Two deviations in loading, not in the computation: the weights are
fetched once with `snapshot_download` at REVISION (the pipeline loader cannot pin a revision of the custom modules),
and the repository's scheduler class (`omnigen2.schedulers`, dynamic time shift by token count) is passed in, so no
code of the Hugging Face repository is run (no `trust_remote_code`).

Settings of example_edit.sh and inference.py: 50 steps, text guidance 5.0, image guidance 2.0 (the README
recommends 1.2-2.0 for editing), CFG over the whole range (0, 1), inference.py's default negative prompt,
`max_sequence_length=1024`, seed via a CUDA generator. The output takes the input's size (`align_res`, one input
image): the pipeline's defaults `max_pixels` 1024 * 1024 and `max_input_image_side_length` 1024 keep our images (at
most 1,024 px) at their size, rounded down to a multiple of 16; the runner resizes it back.

T4 deviations: fp16 instead of bf16 (the authors trained in bf16 and say fp16 "is not expected to perform well",
issue #70; NaN outputs are caught as "black"). No flash-attn: the repository falls back to its SDPA processor and
an unfused SwiGLU by itself. The weights are stored in fp32 (31 GB download; about 15.6 GB in fp16: transformer
7.9, Qwen2.5-VL 7.5), more than one T4 holds, so the pipeline uses diffusers' model CPU offload (mllm, then
transformer, then VAE on the GPU), which the README describes as halving memory "with a negligible impact on speed".
"""

import sys

from .. import TASKS
from ..base import Editor
from .fetch import repository_code

REPO = "VectorSpaceLab/OmniGen2"
COMMIT = "18e6f9d5271b517fcb32e999f10df943ae9b8f20"
MODEL = "OmniGen2/OmniGen2"
REVISION = "df5dca8a981d74e6c3af214c145f5c735fe72367"
WEIGHT_FOLDERS = ["model_index.json", "mllm/*", "processor/*", "scheduler/*", "transformer/*", "vae/*"]
STEPS = 50
TEXT_GUIDANCE = 5.0
IMAGE_GUIDANCE = 2.0
CFG_RANGE = (0.0, 1.0)
MAX_SEQUENCE_LENGTH = 1024
NEGATIVE_PROMPT = ("(((deformed))), blurry, over saturation, bad anatomy, disfigured, poorly drawn face, mutation, "
                   "mutated, (extra_limb), (ugly), (poorly drawn hands), fused fingers, messy drawing, broken legs "
                   "censor, censored, censor_bar")


class OmniGen2(Editor):
    name = "omnigen2"
    repos = (MODEL,)
    local_dirs = (f"OmniGen2-{COMMIT}",)
    download_gb = 31.0
    model_id = f"{MODEL}@{REVISION[:7]} (code {COMMIT[:7]})"
    tasks = TASKS
    uses_mask = False
    install = ("einops",)       # the rest of the inference path is the machine's torch, diffusers, transformers
    settings = {"steps": STEPS, "text_guidance": TEXT_GUIDANCE, "image_guidance": IMAGE_GUIDANCE,
                "cfg_range": list(CFG_RANGE), "negative_prompt": "inference.py default", "max_pixels": 1024 * 1024,
                "dtype": "fp16", "offload": "model_cpu", "prompt": "instruction"}

    def fetch(self) -> None:
        from huggingface_hub import snapshot_download
        repository_code(REPO, COMMIT)
        snapshot_download(MODEL, revision=REVISION, allow_patterns=WEIGHT_FOLDERS)

    def load(self, device: str) -> None:
        import torch
        from huggingface_hub import snapshot_download

        sys.path.insert(0, str(repository_code(REPO, COMMIT)))
        from omnigen2.models.transformers.transformer_omnigen2 import OmniGen2Transformer2DModel
        from omnigen2.pipelines.omnigen2.pipeline_omnigen2 import OmniGen2Pipeline
        from omnigen2.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler

        self.torch, self.device = torch, device
        dtype = torch.float16 if device == "cuda" else torch.float32
        weights = snapshot_download(MODEL, revision=REVISION, allow_patterns=WEIGHT_FOLDERS)
        transformer = OmniGen2Transformer2DModel.from_pretrained(weights, subfolder="transformer", torch_dtype=dtype)
        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(weights, subfolder="scheduler")
        self.pipe = OmniGen2Pipeline.from_pretrained(weights, transformer=transformer, scheduler=scheduler,
                                                     torch_dtype=dtype)
        if device == "cuda":
            self.pipe.enable_model_cpu_offload(device=device)
        else:
            self.pipe.to(device)
        self.pipe.set_progress_bar_config(disable=True)

    def edit(self, image, mask, prompt: str, seed: int = 0):
        generator = self.torch.Generator(device=self.device).manual_seed(seed)
        return self.pipe(prompt=prompt, input_images=[image], num_inference_steps=STEPS,
                         max_sequence_length=MAX_SEQUENCE_LENGTH, text_guidance_scale=TEXT_GUIDANCE,
                         image_guidance_scale=IMAGE_GUIDANCE, cfg_range=CFG_RANGE, negative_prompt=NEGATIVE_PROMPT,
                         generator=generator, output_type="pil").images[0]
