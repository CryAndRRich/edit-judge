"""FLUX.2 [klein] 4B (Black Forest Labs, 01/2026; Apache 2.0): a 4B rectified-flow transformer that does
text-to-image and (multi-)reference image editing in one model, step- and guidance-distilled
(../docs/editors.md list 1).

Weights: hf:black-forest-labs/FLUX.2-klein-4B at REVISION, in diffusers format (Flux2KleinPipeline: transformer
Flux2Transformer2DModel 7.75 GB, text encoder Qwen3ForCausalLM (Qwen3-4B, 36 layers) 8.04 GB, VAE
AutoencoderKLFlux2 0.17 GB, all bf16 on disk). The root file flux-2-klein-4b.safetensors (BFL format) is not
downloaded. Flux2KleinPipeline first appeared in diffusers 0.37.0; this editor runs in its own environment with
diffusers 0.37.1 (the last release without the torch >= 2.6 requirement), on top of the machine's transformers
(Qwen3 needs >= 4.51; < 5 is fine) and huggingface-hub (0.37.1 asks >= 0.34, < 2).

Editing: the photo is passed as the pipeline's `image` (one reference image); the output has the reference's size
when height and width are not given. Settings of the model card: 4 steps, guidance 1.0 (distilled: higher values
are ignored), FlowMatchEuler with the pipeline's default sigmas. The prompt is the simplified instruction.

Deviations, all for the 2 x T4 (fp16 only, 16 GB each):
- fp16 instead of the card's bf16 for the text encoder and the transformer; the VAE runs in fp32 (its config sets
  force_upcast; 84M parameters). The pipeline would decode fp16 latents with the fp32 VAE and fail on the dtype, so
  it returns latents and the VAE decodes them here.
- the two big parts sit on different GPUs: the text encoder on cuda:0, transformer and VAE on cuda:1. The prompt is
  encoded here with the pipeline's own Qwen3 encoder function (chat template, hidden states of layers 9, 18, 27,
  512 tokens) and handed to the pipeline as prompt_embeds; the pipeline is loaded without its text encoder.
- the pipeline center-crops the reference to a multiple of 16 px (the BFL reference code does the same); here the
  photo is resized to a multiple of 16 instead, so the output stays aligned with the original after the runner
  resizes it back. Inputs are at most 1,024 px on the long side, below the pipeline's 1024 x 1024 area cap, so
  they are edited at their own size.
"""

from .. import TASKS
from ..base import Editor

REPO = "black-forest-labs/FLUX.2-klein-4B"
REVISION = "e7b7dc27f91deacad38e78976d1f2b499d76a294"
STEPS = 4
GUIDANCE = 1.0
MULTIPLE = 16               # VAE factor 8 times the 2 x 2 latent patches
HIDDEN_LAYERS = (9, 18, 27)
MAX_TOKENS = 512


def working_size(width: int, height: int) -> tuple[int, int]:
    return max(MULTIPLE, round(width / MULTIPLE) * MULTIPLE), max(MULTIPLE, round(height / MULTIPLE) * MULTIPLE)


class Flux2Klein4B(Editor):
    name = "flux2-klein-4b"
    repos = (REPO,)
    download_gb = 16.0
    gpus = 2
    model_id = f"{REPO}@{REVISION[:7]}"
    tasks = TASKS
    uses_mask = False
    install = ("diffusers==0.37.1",)
    settings = {"steps": STEPS, "guidance": GUIDANCE, "size": f"input, multiple of {MULTIPLE}",
                "dtype": "fp16, VAE fp32", "prompt": "instruction"}

    def fetch(self) -> None:
        from huggingface_hub import snapshot_download
        snapshot_download(REPO, revision=REVISION, allow_patterns=[
            "model_index.json", "scheduler/*", "transformer/*", "text_encoder/*", "tokenizer/*", "vae/*"])

    def load(self, device: str) -> None:
        import torch
        from diffusers import AutoencoderKLFlux2, Flux2KleinPipeline
        from transformers import Qwen2TokenizerFast, Qwen3ForCausalLM

        self.torch = torch
        if device == "cuda" and torch.cuda.device_count() >= 2:
            self.text_device, self.image_device = "cuda:0", "cuda:1"
        else:
            self.text_device = self.image_device = device
        half = torch.float16 if device == "cuda" else torch.float32
        self.tokenizer = Qwen2TokenizerFast.from_pretrained(REPO, subfolder="tokenizer", revision=REVISION)
        self.text_encoder = Qwen3ForCausalLM.from_pretrained(
            REPO, subfolder="text_encoder", revision=REVISION, torch_dtype=half,
            attn_implementation="sdpa").to(self.text_device).eval()
        vae = AutoencoderKLFlux2.from_pretrained(REPO, subfolder="vae", revision=REVISION, torch_dtype=torch.float32)
        self.pipe = Flux2KleinPipeline.from_pretrained(
            REPO, revision=REVISION, vae=vae, text_encoder=None, tokenizer=None, torch_dtype=half)
        self.pipe.to(self.image_device)
        self.pipe.set_progress_bar_config(disable=True)
        self.encode = Flux2KleinPipeline._get_qwen3_prompt_embeds

    def edit(self, image, mask, prompt: str, seed: int = 0):
        from PIL import Image

        torch = self.torch
        with torch.inference_mode():
            embeddings = self.encode(text_encoder=self.text_encoder, tokenizer=self.tokenizer, prompt=prompt,
                                     device=self.text_device, max_sequence_length=MAX_TOKENS,
                                     hidden_states_layers=HIDDEN_LAYERS).to(self.image_device)
            reference = image.resize(working_size(*image.size), Image.Resampling.BICUBIC)
            generator = torch.Generator(device=self.image_device).manual_seed(seed)
            latents = self.pipe(image=reference, prompt_embeds=embeddings, num_inference_steps=STEPS,
                                guidance_scale=GUIDANCE, generator=generator, output_type="latent").images
            decoded = self.pipe.vae.decode(latents.to(torch.float32), return_dict=False)[0]
        return self.pipe.image_processor.postprocess(decoded, output_type="pil")[0]
