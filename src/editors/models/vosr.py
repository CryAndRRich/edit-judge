"""VOSR 2.0 (Wu et al., CVPR 2026; Apache 2.0): one-step generative super-resolution with a 1.4B LightningDiT on
the Qwen-Image 2D VAE latent, conditioned on DINOv2-L features. Used at scale 1 as a sharpener and restorer.

The repository has no package to install: its code is downloaded at COMMIT into EJ_MODELS_DIR and put on the path;
the weights of CSWRY/VOSR at REVISION (VOSR2/, Qwen-Image-vae-2d/, torch_cache/ with DINOv2-L and its hub code)
go into the repository's preset/ckpts, where the script expects them. Its models/rmsnorm.py imports `fairscale`,
which Kaggle lacks (Steven's first run failed on it), so the model gets its own environment with the version of
the repository's requirements.txt; everything else it imports (diffusers, timm, einops, opencv, safetensors) is
already there.

As in Steven's Enhance run of 2026-11-22 (his standalone script, class VOSR2), which reuses the functions of
inference_vosr_onestep.py with `-c preset/ckpts/VOSR2 -u 1 --tile_size 512`: upscale 1 (the output has the input
size), process_size 512, latent tiles of 512 px with a 32 px overlap, full-image VAE, deterministic VAE latent
(posterior mode), 1 step, wavelet colour fix, fp32 (the script has no dtype flag; peak 9.1 GB on a T4). The input
is reflect-padded to a multiple of 16 and the output cropped back. The seed is the runner's (the script's default
is 42); the prompt is ignored and no mask is used.

The script's paths are relative to the repository root (`torch.hub.set_dir('preset/ckpts/torch_cache')` at import,
the VAE folder, `torch.hub.load('facebookresearch/dinov2', ...)`), so load() works inside the repository folder and
returns to the previous folder afterwards; the hub folder is then set to its absolute path. Nothing relative is
read during inference.
"""

import sys

from ..base import Editor
from .fetch import repository_code

REPO = "cswry/VOSR"
COMMIT = "516f292b99cf23c76fdc33351e86dc4f97711fe8"
MODEL = "CSWRY/VOSR"
REVISION = "f24b3061b7f350b81e1907bdcfc27f71fb4ff3f3"
WEIGHT_PATTERNS = ["VOSR2/*", "Qwen-Image-vae-2d/*", "torch_cache/*"]
PAD_MULTIPLE = 16
OVERRIDES = {"upscale": 1, "process_size": 512, "align_method": "wavelet", "tile_size": 512, "tile_overlap": 32,
             "vae_tile_size": None, "vae_tile_overlap": None, "posterior_mode": True, "infer_steps": 1}


def pad_to_multiple(array, factor: int):
    """Reflect-pad an HxWxC array so both sides are multiples of `factor`."""
    import numpy as np
    height, width = array.shape[:2]
    pad_height, pad_width = (-height) % factor, (-width) % factor
    if pad_height or pad_width:
        array = np.pad(array, ((0, pad_height), (0, pad_width), (0, 0)), mode="reflect")
    return array


class VOSR(Editor):
    name = "vosr"
    local_dirs = (f"VOSR-{COMMIT}",)
    download_gb = 8.0
    model_id = f"{MODEL}@{REVISION[:7]} (code {COMMIT[:7]})"
    tasks = ("enhance", "restore")
    uses_mask = False
    install = ("fairscale==0.4.13",)
    settings = {"weights": "CSWRY/VOSR (VOSR2, Qwen-Image-vae-2d, torch_cache)", "upscale": 1,
                "process_size": 512, "tile_size": 512, "tile_overlap": 32, "infer_steps": 1,
                "align_method": "wavelet", "posterior_mode": True, "dtype": "fp32",
                "padding": f"reflect to a multiple of {PAD_MULTIPLE}", "prompt": "ignored"}

    def fetch(self) -> None:
        from huggingface_hub import snapshot_download
        repository = repository_code(REPO, COMMIT)
        snapshot_download(MODEL, revision=REVISION, local_dir=str(repository / "preset" / "ckpts"),
                          allow_patterns=WEIGHT_PATTERNS)

    def load(self, device: str) -> None:
        import argparse
        import json
        import os

        import torch
        from huggingface_hub import snapshot_download
        from safetensors.torch import load_file

        repository = repository_code(REPO, COMMIT)
        checkpoints = repository / "preset" / "ckpts"
        snapshot_download(MODEL, revision=REVISION, local_dir=str(checkpoints), allow_patterns=WEIGHT_PATTERNS)
        sys.path.insert(0, str(repository))     # before the import: the repository's top-level `models` package
        previous_folder = os.getcwd()
        os.chdir(repository)                    # the script's paths (preset/ckpts/...) are relative
        try:
            import inference_vosr_onestep as script
            from models.lightningdit import LightningDiT
            from models.qwenimage_vae2d import AutoencoderKLQwenImage2D

            torch.hub.set_dir(str(checkpoints / "torch_cache"))     # absolute, so it stays valid after the chdir
            weights_folder = checkpoints / "VOSR2"
            with open(weights_folder / "args.json") as handle:
                args = argparse.Namespace(**json.load(handle))
            for key, value in OVERRIDES.items():
                setattr(args, key, value)
            # as the script: a missing key means True; VOSR2's args.json has False
            args.auxiliary_time_cond = bool(getattr(args, "auxiliary_time_cond", None) is not False)
            if args.ae_type != "qwen":
                raise RuntimeError(f"VOSR2 expected the qwen VAE, got {args.ae_type}")
            self.vae = AutoencoderKLQwenImage2D.from_pretrained(str(checkpoints / "Qwen-Image-vae-2d"))
            self.vae.to(device).eval()
            self.encoder = script.load_dinov2(args, device)
            model = LightningDiT(
                input_size=args.resolution // 8, patch_size=args.patch_size, in_channels=2 * 16, out_channels=16,
                hidden_size=args.dim, depth=args.depth, num_heads=args.num_heads, mlp_ratio=args.mlp_ratio,
                z_dims=args.enc_dim, encdim_ratio=args.encdim_ratio, auxiliary_time_cond=args.auxiliary_time_cond,
                use_qknorm=args.use_qknorm, use_swiglu=args.use_swiglu, use_rope=args.use_rope,
                use_rmsnorm=args.use_rmsnorm, wo_shift=args.wo_shift,
                num_fused_layers=len(args.layer_dinov2b_list))
            weights = next(path for path in (weights_folder / "clean_weights" / "ema_model.safetensors",
                                             weights_folder / "checkpoints" / "ema_model.safetensors",
                                             weights_folder / "ema_model.safetensors") if path.is_file())
            print(model.load_state_dict(load_file(str(weights)), strict=False), flush=True)
            model.to(device).eval()
            model.forward = model.forward_flexible    # variable input sizes (RoPE built per call), as the script
        finally:
            os.chdir(previous_folder)
        self.model, self.args, self.script, self.torch, self.device = model, args, script, torch, device

    def edit(self, image, mask, prompt: str, seed: int = 0):
        import numpy as np
        from PIL import Image

        torch = self.torch
        width, height = image.size
        padded = Image.fromarray(pad_to_multiple(np.asarray(image), PAD_MULTIPLE))
        pixels = torch.from_numpy(np.asarray(padded, dtype=np.float32) / 255.0).permute(2, 0, 1)[None]
        pixels = pixels.to(self.device) * 2.0 - 1.0
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        with torch.no_grad():
            result = self.script.tiled_latent_inference(self.model, self.vae, self.encoder, pixels, self.args,
                                                        device=self.device, light_decoder=None)
        result = torch.nan_to_num(result[0].float() * 0.5 + 0.5, nan=0.0, posinf=1.0, neginf=0.0).clamp(0, 1)
        array = (result * 255).round().byte().permute(1, 2, 0).cpu().numpy()
        output = self.script.wavelet_color_fix(Image.fromarray(array), padded)
        return output.crop((0, 0, width, height))
