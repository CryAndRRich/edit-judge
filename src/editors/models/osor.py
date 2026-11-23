"""OSOR-SDXL-Inpainting (Zhou et al., ECCV 2026; code MIT, checkpoint OpenRAIL++-M): one-step object removal that
also removes the object's shadows and reflections, on SDXL Inpainting 0.1 with a rank-256 LoRA and a 5-channel
output layer whose fifth channel is a blending alpha.

The authors' inference (zhouqm-git/osor at CODE_COMMIT, osor-sdxlinpainting/scripts/inference_enhance.py) is
re-implemented here rather than imported: its modules import `src.models...`, which collides with this repository's
`src` package. Re-implemented as written there:
- the SDXL-Inpainting UNet with peft LoRA (rank 256, alpha 256) on the listed modules, then conv_out grown from 4 to
  5 channels, then the phase II checkpoint (conv_out + LoRA) loaded non-strictly (frozen weights are not in it;
  every key of the checkpoint must be used);
- the fixed prompt "Remove the instance of object", encoded once with SDXL's two text encoders (cache_prompts.py);
- one step at t = 400: the image latent is noised (DDPM), the UNet gets [noisy latent, max-pooled mask, image
  latent], the predicted x0 is blended with the image latent by the predicted alpha, then decoded;
- shorter side 512, both sides rounded up to a multiple of 16.
Changes for T4: float16 instead of the hard-coded bfloat16 (the SDXL-Inpainting VAE runs in fp16 here, as for
sdxl-inpaint; black outputs are flagged by the runner). The authors' paste-back by alpha is off (the paper setting);
the runner's paste-back by our mask applies, as for every model.
"""

from ..base import Editor

BASE = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
BASE_REVISION = "115134f363124c53c7d878647567d04daf26e41e"
WEIGHTS = "QinmingZhou/OSOR"
WEIGHTS_REVISION = "9288156766b1df44e5f50fb697876a6f2fd8c062"
WEIGHTS_FILE = "osor-sdxlinpainting/weights/sdxlinpainting_phase2.pth"
CODE_COMMIT = "d5b68e822996d61aae6ba612e5e9b5a14f576879"
PROMPT = "Remove the instance of object"
LORA_MODULES = ["to_k", "to_q", "to_v", "to_out.0", "conv", "conv1", "conv2", "conv_shortcut", "proj_in",
                "proj_out", "ff.net.2", "ff.net.0.proj"]
RANK = 256
T = 400
SHORT_SIDE = 512


def working_size(width: int, height: int, short: int = SHORT_SIDE) -> tuple[int, int]:
    """inference_enhance.py:_calculate_resize_dimensions."""
    if min(width, height) < short:
        new_w, new_h = width, height
    elif width < height:
        new_w, new_h = short, int(height * short / width)
    else:
        new_w, new_h = int(width * short / height), short
    return (new_w + 15) // 16 * 16, (new_h + 15) // 16 * 16


class OSOR(Editor):
    name = "osor"
    model_id = f"{WEIGHTS}@{WEIGHTS_REVISION[:7]}:{WEIGHTS_FILE} on {BASE} (code {CODE_COMMIT[:7]})"
    tasks = ("remove",)
    settings = {"short_side": SHORT_SIDE, "steps": 1, "t": T, "lora_rank": RANK, "prompt": PROMPT,
                "dtype": "float16"}

    def load(self, device: str) -> None:
        import torch
        import torch.nn as nn
        from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
        from huggingface_hub import hf_hub_download
        from peft import LoraConfig
        from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

        self.torch, self.device = torch, device
        self.dtype = dtype = torch.float16 if device == "cuda" else torch.float32
        kw = {"revision": BASE_REVISION}
        self.vae = AutoencoderKL.from_pretrained(BASE, subfolder="vae", torch_dtype=dtype, **kw).to(device).eval()
        self.scheduler = DDPMScheduler.from_pretrained(BASE, subfolder="scheduler", **kw)

        class Generator(nn.Module):                   # same module names as the authors' DiffGANGeneratorEnhance
            def __init__(self):
                super().__init__()
                self.unet = UNet2DConditionModel.from_pretrained(BASE, subfolder="unet", torch_dtype=dtype, **kw)
                self.unet.add_adapter(LoraConfig(r=RANK, lora_alpha=RANK, init_lora_weights="gaussian",
                                                 target_modules=LORA_MODULES))
                old = self.unet.conv_out
                new = nn.Conv2d(old.in_channels, 5, old.kernel_size, old.stride, old.padding)
                with torch.no_grad():
                    new.weight[:4] = old.weight
                    new.bias[:4] = old.bias
                self.unet.conv_out = new

        self.generator = Generator()
        path = hf_hub_download(WEIGHTS, WEIGHTS_FILE, revision=WEIGHTS_REVISION)
        state = torch.load(path, map_location="cpu", weights_only=True)
        state = {k.replace("module.", "").replace("_orig_mod.", ""): v for k, v in state.items()}
        keys = self.generator.load_state_dict(state, strict=False)
        if keys.unexpected_keys:
            raise RuntimeError(f"OSOR checkpoint keys not used: {keys.unexpected_keys[:5]} "
                               f"({len(keys.unexpected_keys)} in all): the module layout does not match")
        self.generator.to(device, dtype=dtype).eval()

        # the fixed prompt, encoded once (cache_prompts.py), then the text encoders are freed
        with torch.no_grad():
            embeds = []
            for sub, model_class in (("", CLIPTextModel), ("_2", CLIPTextModelWithProjection)):
                tokenizer = CLIPTokenizer.from_pretrained(BASE, subfolder=f"tokenizer{sub}", **kw)
                encoder = model_class.from_pretrained(BASE, subfolder=f"text_encoder{sub}", torch_dtype=dtype,
                                                      **kw).to(device)
                ids = tokenizer(PROMPT, padding="max_length", max_length=tokenizer.model_max_length,
                                truncation=True, return_tensors="pt").input_ids.to(device)
                output = encoder(ids, output_hidden_states=True)
                embeds.append(output.hidden_states[-2])
                if sub:
                    self.pooled = output.text_embeds
                del encoder
            self.prompt_embeds = torch.cat(embeds, dim=-1)
        if device == "cuda":
            torch.cuda.empty_cache()

    def edit(self, image, mask, prompt: str, seed: int = 0):
        import torch.nn.functional as F
        from PIL import Image
        from torchvision.transforms.functional import to_tensor

        torch = self.torch
        size = working_size(*image.size)
        lq = to_tensor(image.resize(size, Image.Resampling.BICUBIC))[None].to(self.device)
        holes = to_tensor(mask.resize(size, Image.Resampling.NEAREST))[None].to(self.device)
        torch.manual_seed(seed)
        with torch.no_grad():
            z_lq = self.vae.encode((lq * 2 - 1).to(self.dtype)).latent_dist.sample()
            z_lq = z_lq * self.vae.config.scaling_factor
            mask_latent = F.max_pool2d(holes.float(), kernel_size=8, stride=8).to(self.dtype)
            noise = torch.randn_like(z_lq)
            timesteps = torch.full((1,), T, dtype=torch.long, device=self.device)
            z = self.scheduler.add_noise(z_lq, noise, timesteps)
            height, width = lq.shape[-2:]
            time_ids = torch.tensor([[height, width, 0, 0, height, width]], dtype=self.dtype, device=self.device)
            output = self.generator.unet(torch.cat([z, mask_latent, z_lq], dim=1), timesteps,
                                         encoder_hidden_states=self.prompt_embeds,
                                         added_cond_kwargs={"text_embeds": self.pooled, "time_ids": time_ids},
                                         return_dict=False)[0]
            eps, alpha = output[:, :4], torch.sigmoid(output[:, 4:])
            # an int timestep: the scheduler's tables stay on the CPU (a CUDA index tensor cannot index them)
            z_out = self.scheduler.step(eps, T, z).pred_original_sample
            z_out = z_out * alpha + z_lq * (1 - alpha)
            x = self.vae.decode(z_out.to(self.dtype) / self.vae.config.scaling_factor).sample.float()
        x = ((x + 1) / 2).clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy()
        return Image.fromarray((x * 255).astype("uint8"))
