# Editing models to run

These are the image editing models we run for the model ranking (Part II Step 7 in `../docs/PROJECT.md`). Our scorer then judges their outputs.

There are 7 lists:
- one shared list of generalist models, which follow any instruction;
- one list for each of the 6 tasks that have specialist models.

The tasks Open-Ended and Other use the shared list only. A grounding step makes the masks that some specialists need.

Every model below was checked on 2026-11-17 against its Hugging Face API file list, its model card and its GitHub repository. Each model had its own check. Lists 2 and 3 were extended on 2026-11-21 after a second search of CVPR, ICCV, ECCV, NeurIPS and ICLR papers (models marked *added 2026-11-21*; candidates for later are under "Candidates for later"). Lists 2 and 3 run in `src/editors` (status in the table at the end); the other lists have not been run yet.

## Summary

The **Date** column is:
- the conference and year, for a model whose paper a conference accepted;
- otherwise the month and year of its Hugging Face release;
- otherwise "arXiv" and the year.

Conference acceptances come from the authors' README or arXiv page. The conference sites blocked automated reads, so they were not cross-checked.

| Model | Date | Tasks (list numbers) |
|---|---|---|
| FLUX 3 Image Klein | 10/2026 | 1 Shared |
| VAREdit 2B | ICLR 2026 | 1 Shared |
| OSOR-SDXL-Inpainting | ECCV 2026 | 2 Remove (added 2026-11-21) |
| ObjectClear | CVPR 2026 | 2 Remove |
| VOSR 2.0 | CVPR 2026 | 4 Enhance, 5 Restore |
| AceTone | CVPR 2026 | 4 Enhance (also whole-image tone in 6 Change-Color) |
| FLUX.2 [klein] 4B | 01/2026 | 1 Shared |
| VIBE | 12/2025 | 1 Shared |
| OmniGen2 | 06/2025 | 1 Shared |
| RealEdit | CVPR 2025 | 1 Shared (classic) |
| RORem | CVPR 2025 | 2 Remove |
| Paint by Inpaint (add-finetuned-mb) | CVPR 2025 | 3 Add (mask-free; added 2026-11-21) |
| HVI-CIDNet | CVPR 2025 | 4 Enhance |
| BEN2 | 01/2025 | 7 Image-Formatting |
| InstantIR | 10/2024 | 5 Restore |
| InstructIR | ECCV 2024 | 4 Enhance |
| EraseDraw v0.2 | ECCV 2024 | 3 Add (mask-free; added 2026-11-21) |
| BiRefNet | 07/2024 | 7 Image-Formatting |
| MagicBrush | NeurIPS 2023 | 1 Shared (classic) |
| SDXL Inpainting 0.1 | 09/2023 | 3 Add/Replace, 6 Change-Color (one region), 7 Image-Formatting (outpainting) |
| DDColor | ICCV 2023 | 6 Change-Color |
| MI-GAN | ICCV 2023 | 2 Remove (light GAN; added 2026-11-21) |
| InstructPix2Pix | CVPR 2023 | 1 Shared (classic) |
| CodeFormer | NeurIPS 2022 | 5 Restore (classic) |
| LaMa (big-lama) | WACV 2022 | 2 Remove (classic) |
| Real-ESRGAN x4plus | ICCV 2021 (workshop) | 4 Enhance, 7 Image-Formatting (classic) |
| GFPGAN v1.4 | CVPR 2021 | 5 Restore (classic) |
| Grounding DINO base | ECCV 2024 | Masks for lists 2, 3, 6 and 7 |
| SAM 2.1 large | ICLR 2025 | Masks for lists 2, 3, 6 and 7 |

That is 27 editing models and 2 grounding models. List 7 also uses plain code (PIL) for crop, resize and rotate.

## Rules

**Open weights only.** No paid closed-source APIs.

**Hardware: Kaggle, 2 x T4.**
- Each GPU has 16 GB (about 15 GB usable), so 32 GB in total.
- fp16 only: T4 has no fast bf16. Most new weights are stored in bf16 and must be cast to fp16 at load. Some code hard-codes bf16 or flash-attention-2; the notes say where to patch it (fp16, and `sdpa` attention).
- FLUX models can give black or NaN images in fp16. If that happens, keep the VAE in fp32.
- A model is kept when its weights in fp16 fit in one T4, or split over the two:
  - text encoder on GPU 0, main network on GPU 1;
  - or diffusers `device_map="balanced"`.

**Preference.** New, light, and published at a good conference. Each list also keeps one or two classic models as baselines.

**No Stable Diffusion 2.1.** Its official Hugging Face repository was removed, so models built on it are left out.

**Columns of the lists:**
- **Weights (fp16):** the size of the weights the model loads, in GB, computed from the file sizes in the Hugging Face API. It does not count activations.
- **Effort:**
  - **easy:** a standard diffusers or transformers class, or a pip package;
  - **medium:** clone the GitHub repository and use its code, patch it, or pin old packages in their own venv (see `src/common/envs.py`).
- **Source:** `hf:` is a Hugging Face repository, `gh:` a GitHub repository.

**Settings.** Use the steps and guidance given here, from each model card, unless the smoke run shows a problem.

## 1. Shared list (generalists)

Every model here runs on every sample of the benchmark set.

| Model | Date | Source | Weights (fp16) | License | Effort | Settings and notes |
|---|---|---|---|---|---|---|
| FLUX 3 Image Klein | 10/2026 | hf:black-forest-labs/flux-3-image-klein | 20.3 (transformer 11.10, text encoder 8.88, VAE 0.34) | FLUX Kommunity license, gated | medium | Accept the license on the page and use an HF token. The card and the pipeline class are hidden until access is granted, so read them first. Download only `transformer/`, `text_encoder/`, `vae/` (`allow_patterns`): the whole repository is 43 GB. Split over the 2 GPUs. |
| VAREdit 2B | ICLR 2026 | hf:HiDream-ai/VAREdit (`2B-512.pth`, `flan-t5-xl/`), gh:HiDream-ai/VAREdit; VAE `infinity_vae_d32reg.pth` from hf:FoundationVision/Infinity | about 11.7 (model 4.4, flan-t5-xl 5.7, VAE 1.6); the files are fp32: 8.8 + 11.4 + 1.6 GB to download | MIT | medium | Visual autoregressive, not diffusion, so it is fast. 512 x 512 only (the input is padded to a square). `infer.py`: `load_model(..., model_size="2B", image_size=512)`, then `generate_image(...)` with cfg 3.0, tau 0.1 (from the README table; the code defaults differ). The code hard-codes bf16 autocast in several places: patch to fp16. One GPU. |
| FLUX.2 [klein] 4B | 01/2026 | hf:black-forest-labs/FLUX.2-klein-4B | 16.0 (transformer 7.75, text encoder 8.04, VAE 0.17) | Apache 2.0 | easy | `Flux2KleinPipeline`, diffusers from GitHub main; 4 steps, guidance 1.0. Split the text encoder and transformer over the 2 GPUs. The card's example is text-to-image only: check the name of the input-image argument in the pipeline source. |
| VIBE | 12/2025 (arXiv 2601.02242) | hf:iitolstykh/VIBE-Image-Edit, gh:ai-forever/VIBE | 9.3 (Sana1.5 1.6B 3.8, Qwen3-VL-2B 4.3, VAE 1.2) | code Apache 2.0, model under the SANA license | medium | Fits one T4. `vibe` package from GitHub, `ImageEditor(...)`, 20 steps, guidance 4.5, image guidance 1.2. The code hard-codes bf16: patch `weight_dtype` to fp16. Pins torch 2.6, diffusers 0.33.1, transformers 4.57.1, so use its own venv. |
| OmniGen2 | 06/2025 | hf:OmniGen2/OmniGen2, gh:VectorSpaceLab/OmniGen2 | about 15.1 (transformer 7.9, Qwen2.5-VL 7.0); stored in fp32, 30 GB to download | Apache 2.0 | medium | Repository script `inference.py` (diffusers support is not done). torch 2.6 pin, so use its own venv. No flash-attn on T4. 50 steps, text guidance 5.0, image guidance 2.0 (the card says 1.2-2.0 for editing). Needs CPU offload or a split over the 2 GPUs. |
| *Classic:* RealEdit | CVPR 2025 | hf:peter-sushko/RealEdit | about 2.2 (.bin, fp32 on disk) | card: CC; GitHub: MIT | easy | `StableDiffusionInstructPix2PixPipeline`, `safety_checker=None`, Euler Ancestral scheduler, 50 steps, image guidance 2 (card example). InstructPix2Pix fine-tuned on RealEdit, the source of 92% of our samples. |
| *Classic:* MagicBrush | NeurIPS 2023 | hf:vinesmsuic/magicbrush-jul7 | about 2.2 (.bin) | CreativeML OpenRAIL-M | easy | Community diffusers port of the authors' checkpoint (osunlp/InstructPix2Pix-MagicBrush, .ckpt only). Same pipeline, 20 steps, guidance 7, image guidance 1.5. |
| *Classic:* InstructPix2Pix | CVPR 2023 | hf:timbrooks/instruct-pix2pix | 2.7 | MIT | easy | `StableDiffusionInstructPix2PixPipeline` with `variant="fp16"` (otherwise the fp32 files are downloaded). Pipeline defaults: 100 steps, guidance 7.5, image guidance 1.5. |

**This list is short.** Most new open generalists are 30-58 GB in fp16 and do not fit 2 T4s without quantization (see "Left out"). Of the new ones that fit, only VAREdit 2B has a conference paper.

## 2. Remove (objects, people, text, logos)

All of these need a mask (see Grounding).

| Model | Date | Source | Weights (fp16) | License | Effort | Settings and notes |
|---|---|---|---|---|---|---|
| OSOR-SDXL-Inpainting *(added 2026-11-21)* | ECCV 2026 | hf:QinmingZhou/OSOR (`osor-sdxlinpainting/weights/sdxlinpainting_phase2.pth`), gh:zhouqm-git/osor | 3.1 (rank-256 LoRA + 5-channel conv_out, .pth) on SDXL Inpainting 0.1 (6.9) | code MIT; checkpoint OpenRAIL++-M | medium | **One step** (t = 400), removes shadows and reflections too. Its code imports `src.models...`, which collides with our `src` package, so `src/editors/models/osor.py` re-implements `inference_enhance.py` (UNet + peft LoRA + 5-channel conv_out; the module layout was checked against all 1,544 checkpoint keys). bfloat16 is hard-coded upstream: run in float16. Fixed prompt "Remove the instance of object", shorter side 512. Needs `peft`. |
| ObjectClear | CVPR 2026 | hf:jixin0101/ObjectClear, gh:zjx0101/ObjectClear | 7.8 | S-Lab, non-commercial | medium | Custom `ObjectClearPipeline` from the GitHub package; `--use_fp16`; shorter side 512, 20 steps, guidance 2.5. Also removes shadows and reflections. The HF weights date from 2025-07; the paper was renamed for CVPR 2026. |
| RORem | CVPR 2025 | hf:LetsThink/RORem | 6.9 | Apache 2.0 | easy | `AutoPipelineForInpainting` (an SDXL inpainting pipeline); 512 x 512, 50 steps, strength 0.99, guidance 1.0. The card's code has bugs: write the call yourself. The repository's pins (diffusers 0.30) are only needed for training. |
| MI-GAN *(added 2026-11-21)* | ICCV 2023 | TorchScript `migan_traced.pt` from gh:Sanster/models, release `migan` (27 MB, MD5 76eb3b1a…), as used by IOPaint | 0.03 | MIT | easy | Light GAN distilled from Co-Mod-GAN; a second fast baseline next to LaMa. 512 x 512 only: crop around the mask (128 px margin), scale to 512, pad to a square, input `[0.5 - mask, image * (1 - mask)]` in [-1, 1] (IOPaint's pre-processing). No pip package (iopaint pins clash). |
| *Classic:* LaMa (big-lama) | WACV 2022 | TorchScript `big-lama.pt` from gh:enesmsahin/simple-lama-inpainting, release v0.1.0 | 0.2 | Apache 2.0 | easy | Load with `torch.jit.load`, without the pip package, whose pins (numpy < 2, pillow < 10) clash with Kaggle. White in the mask = fill. The output is padded to a multiple of 8: crop it back. |

**Remove Background** (1,193 samples) often asks for a plain or white background. The cut-out models of list 7 also run on these samples.

## 3. Add and Replace (new objects, people, clothes, backgrounds)

| Model | Date | Source | Weights (fp16) | License | Effort | Settings and notes |
|---|---|---|---|---|---|---|
| SDXL Inpainting 0.1 | 09/2023 | hf:diffusers/stable-diffusion-xl-1.0-inpainting-0.1 | 6.9 | OpenRAIL++ (card: research use) | easy | `AutoPipelineForInpainting`, `variant="fp16"`; long side 1024 (aspect kept), 20 steps, guidance 8.0, strength 0.99 (the card says 1.0 lowers quality). White in the mask = repaint. For Add, the mask is a box around the anchor (the existing thing the new one goes on) or Qwen's free box, minus the things to keep. Smoke runs: it often ignored the prompt when it repainted a large background ("a galaxy" gave a plain white background); strength 1.0 for the background op fixed it (full run, 2026-11-21). |
| Paint by Inpaint *(added 2026-11-21)* | CVPR 2025 | hf:paint-by-inpaint/add-finetuned-mb, gh:RotsteinNoam/Paint-by-Inpaint | 2.1 (already fp16) | MIT | easy | **Mask-free**: adds an object from the instruction alone and places it itself (trained on PIPE, 1M pairs made by removing objects; this variant fine-tuned on MagicBrush additions). `StableDiffusionInstructPix2PixPipeline`, Euler Ancestral, 50 steps, guidance 7, image guidance 1.5 (README). Long side 512. Add samples only. |
| EraseDraw v0.2 *(added 2026-11-21)* | ECCV 2024 | hf:alpercanberk/erasedraw-v0.2 | 2.1 (stored fp32, 4.3) | MIT | easy | **Mask-free** addition, the same pipeline; 10 steps, image guidance 1, guidance 7.5 (model card). The card says code is "coming soon"; the weights run without it. Long side 512. Add samples only. |

**Why two mask-free models were added.**
- Our mask for Add comes from Qwen2.5-VL: it places existing things well but new things badly (smoke runs of 2026-11-20/21: facial hair boxed on the chest, a camel over a person).
- Paint by Inpaint and EraseDraw decide the placement themselves, so they test Add without that weakness; SDXL Inpainting tests whether a mask helps.
- The insertion models built on 12B FLUX (OmniPaint, ICEdit, Add-it) are still too big; PowerPaint and BrushNet are still Gradio-only with old pins.

## 4. Enhance (quality, sharpness, lighting, colour tone, retouching)

| Model | Date | Source | Weights (fp16) | License | Effort | Settings and notes |
|---|---|---|---|---|---|---|
| VOSR 2.0 | CVPR 2026 | hf:CSWRY/VOSR (`VOSR2/`, `Qwen-Image-vae-2d/`, `torch_cache/`), gh:cswry/VOSR | about 7.0 in fp32 (DiT 1.4B 5.6, DINOv2-L 1.2, VAE 0.2) | Apache 2.0 | medium | One-step generative super-resolution. VOSR 2.0 does not use SD 2.1 (only the older 0.5B models do). `inference_vosr_onestep.py -c preset/ckpts/VOSR2 -u 1` for sharpening without upscaling (the code allows 1; the README only shows 4). Runs in fp32 (no dtype flag); use `--tile_size 512` for large photos. Run from the repository root (relative paths). Bad images are skipped silently: count the outputs. |
| AceTone | CVPR 2026 | hf:Vivre/AceTone-3B-PST-Preview, gh:martian422/AceTone (branch `open-source-ready`) | 8.1 (a Qwen2.5-VL-3B-type VLM) | Apache 2.0 | medium | Writes a 3D colour LUT from the instruction, then applies it. It changes global colour and tone only ("make it warmer", "vintage look"), never local content. Needs the repository's `model/` code and VQ checkpoint; write a short inference script from `demo/predict_lut.py`. Patch bf16 to fp16 and flash-attention-2 to `sdpa`. |
| HVI-CIDNet | CVPR 2025 | hf:Fediory/HVI-CIDNet-Generalization, gh:Fediory/HVI-CIDNet | 0.008 (1.9M parameters) | MIT | easy | For dark photos. `eval_hf.py` with the "generalization" weights, which the authors recommend for real photos; `alpha_i` sets the brightness. Skip `requirements.txt` (torch 1.13). Nothing stops it from brightening a photo that is already well lit, so run it only on lighting requests. |
| InstructIR | ECCV 2024 | hf:marcosv/InstructIR, gh:mv-lab/InstructIR | 0.06 + text encoder TaylorAI/bge-micro-v2 | MIT | medium | Needs code from the repository: `configs/eval5d.yml`, `models/`, `text/`. Handles 7 degradations named in the instruction: noise, blur, rain, haze, low light, enhancement, upscaling. The authors say it generalises poorly to real photos. |
| *Classic:* Real-ESRGAN x4plus | ICCV 2021 (workshop) | `RealESRGAN_x4plus.pth` from gh:xinntao/Real-ESRGAN, release v0.1.0 | 0.07 | BSD-3 | easy | Load with the `spandrel` package (0.4.2, 2026). The `realesrgan` and `basicsr` packages are from 2022 and break on new torchvision. |

**Enhance Person** (1,426 samples) is mostly retouching (skin, eyes, smiles). No specialist covers it, so the shared list decides this row.

## 5. Restore (old, damaged or faded photos)

| Model | Date | Source | Weights (fp16) | License | Effort | Settings and notes |
|---|---|---|---|---|---|---|
| VOSR 2.0 | CVPR 2026 | as in list 4 | about 7.0 | Apache 2.0 | medium | For blurred, low-resolution and compressed photos. It does not remove scratches or tears. |
| InstantIR | 10/2024 | hf:InstantX/InstantIR, gh:instantX-research/InstantIR | about 11-12 (SDXL base about 7, InstantIR .pt about 4, DINOv2-large 0.6) | Apache 2.0 | medium | Custom `InstantIRPipeline` from the repository. Pass `--vision_encoder_path facebook/dinov2-large` (the default is a private path). 30 steps, cfg 7.0, prompt optional, batch size 1. Old pins (diffusers 0.28, torch 2.0), so use its own venv. One T4. |
| *Classic:* CodeFormer | NeurIPS 2022 | gh:sczhou/CodeFormer, release v0.1.0 | 0.4 + face detection models | S-Lab, non-commercial | medium | `inference_codeformer.py -w 0.7`. Bundles its own basicsr. Documented for Python 3.8, so the smoke run must confirm it works on 3.11. |
| *Classic:* GFPGAN v1.4 | CVPR 2021 | gh:TencentARC/GFPGAN, `GFPGANv1.4.pth` in release v1.3.0 | small | Apache 2.0 (README) | medium | Restores faces; whole photos work (faces are found and pasted back), and the background is upscaled with Real-ESRGAN. `basicsr` imports `torchvision.transforms.functional_tensor`, which new torchvision removed: patch the import. |

## 6. Change-Color (recolouring, colourising black-and-white photos)

| Model | Date | Source | Weights (fp16) | License | Effort | Settings and notes |
|---|---|---|---|---|---|---|
| SDXL Inpainting 0.1, one region | 09/2023 | hf:diffusers/stable-diffusion-xl-1.0-inpainting-0.1 | 6.9 | OpenRAIL++ | easy | For "make the car red": mask the object and prompt with the new colour. Same settings as list 3. |
| DDColor | ICCV 2023 | hf:piddnad/ddcolor_modelscope, gh:piddnad/DDColor | 0.9 | Apache 2.0 | medium | Use the `modelscope` checkpoint, which MODEL_ZOO.md recommends for photos outside ImageNet. Clone the repository: `DDColorHF.from_pretrained` needs its bundled basicsr on the path. Input 512; the colour is upscaled back to full size. Colourises the whole image only. |

AceTone (list 4) also runs here on requests that change the colour tone of the whole photo.

**No recent photo colouriser.**
- We found no usable colourisation model for photos from 2025-2026.
- ColorFLUX (CVPR 2026) has only unofficial weights, no inference code, and is built on 12B FLUX.
- The other recent work at good conferences is on anime and sketches.

## 7. Image-Formatting (cut-out, background removal, extend, resize, crop)

| Model | Date | Source | Weights (fp16) | License | Effort | Settings and notes |
|---|---|---|---|---|---|---|
| BEN2 (Base) | 01/2025 | hf:PramaLLC/BEN2 | 0.38 | MIT | easy | `pip install git+https://github.com/PramaLLC/BEN2.git`, then `BEN_Base.from_pretrained("PramaLLC/BEN2")` and `model.inference(image)`. The two READMEs give different APIs: check in the smoke run. |
| BiRefNet | 07/2024 (journal CAAI AIR 2024) | hf:ZhengPeng7/BiRefNet | 0.44 | MIT | easy | `AutoModelForImageSegmentation`, `trust_remote_code=True`, `.half()`; resize to 1024 x 1024, ImageNet normalisation. Output RGBA, or composite on white. Skip the repository's `torch==2.5.1` pin. |
| SDXL Inpainting 0.1, outpainting | 09/2023 | hf:diffusers/stable-diffusion-xl-1.0-inpainting-0.1 | 6.9 | OpenRAIL++ | easy | Our own padding: enlarge the canvas, mask the new border. The card does not document this use. |
| *Classic:* Real-ESRGAN x4plus | ICCV 2021 (workshop) | as in list 4 | 0.07 | BSD-3 | easy | For "make it bigger". |
| *Classic:* plain code | | PIL | | | easy | Crop, resize, rotate, change format. Sometimes the right answer. |

## Grounding (masks for lists 2, 3, 6 and 7)

| Tool | Date | Source | Weights | License | Notes |
|---|---|---|---|---|---|
| Grounding DINO base | ECCV 2024 | hf:IDEA-Research/grounding-dino-base | 0.9 GB (fp32) | Apache 2.0 | `AutoModelForZeroShotObjectDetection`. The text must be lowercase and end with a dot ("the man on the left."). Box threshold 0.4, text threshold 0.3. The argument name changed across transformers versions (`box_threshold` or `threshold`). |
| SAM 2.1 large | ICLR 2025 | hf:facebook/sam2.1-hiera-large | 0.9 GB (fp32) | Apache 2.0 | `Sam2Model` and `Sam2Processor`, with the Grounding DINO boxes as `input_boxes`, `multimask_output=False`. The card's example uses bf16: use fp16 or fp32 on T4. |

**How the masks are made:**
1. The target phrase comes from the instruction.
2. Grounding DINO finds its box, and SAM 2.1 turns the box into a mask. For Add, where there is nothing to find yet, Qwen2.5-VL-7B gives the box instead.
3. Every mask is saved, so a bad mask can be told apart from a bad edit.
4. If nothing is found, the sample is recorded as `no-mask`.

## Left out

| Model | Date | Why |
|---|---|---|
| DIM-4.6B-Edit | ICLR 2026 | Its demo sends every instruction to GPT-4o (a paid API) first; bf16 and flash-attention-2 hard-coded; non-commercial |
| VAREdit 8B | ICLR 2026 | 33.5 GB in fp32 (about 17 GB in fp16), more than one T4; single-GPU code |
| NP-Edit ("Learning an Image Editing Model without Image Editing Pairs") | ICLR 2026 | Only an unofficial reimplementation on Z-Image 6B: 20.5 GB on one GPU, flash-attn 4, tested on H200 |
| InstantRetouch | CVPR 2026 | Code only, no weights (you would have to train it) |
| RDBM | CVPR 2026 | Training code only, no weights, no inference script |
| ColorFLUX | CVPR 2026 | Unofficial upload of weights, no inference code; built on FLUX.1 [dev] (about 41 GB to download), non-commercial |
| ICEdit | NeurIPS 2025 | A LoRA on FLUX.1 Fill [dev]: 33.9 GB in fp16, bf16-only script, non-commercial, gated base |
| OmniPaint | ICCV 2025 | LoRAs on FLUX.1 [dev]: 33.7 GB; insertion needs a reference image of the object, not text |
| PiSA-SR | CVPR 2025 | Built on Stable Diffusion 2.1, whose official Hugging Face repository was removed; only unofficial mirrors remain |
| HYPIR | SIGGRAPH 2025 | Built on Stable Diffusion 2.1; non-commercial |
| OSDFace | CVPR 2025 | Built on Stable Diffusion 2.1; aligned 512 px faces only (no detection or paste-back); weights only on Google Drive; non-commercial |
| SmartEraser | CVPR 2025 | Weights only on Baidu and Google Drive; custom pipeline on diffusers 0.30 and Gradio 3; RORem and ObjectClear cover removal. Update 2026-11-21: a third-party HF mirror now exists (see "Candidates for later") |
| DarkIR | CVPR 2025 | Script hard-codes `CUDA_VISIBLE_DEVICES="1"`; the HF weights do not match the shipped config; HVI-CIDNet covers low light |
| FLUX.2 [klein] 9B | 01/2026 | 34.7 GB in fp16 (transformer 18.2, text encoder 16.4); gated, non-commercial. Possible later with 8-bit weights |
| Qwen-Image-2.1 (also Turbo) | 09/2026 | 33.1 GB; needs transformers >= 5.17; examples at 2048 px |
| FireRed-Image-Edit 1.0 / 1.1 | 02/2026 | 57.7 GB (transformer 40.9, text encoder 16.6); its own "optimized" Int8 path still needs 30 GB |
| LLaDA-Image (and Turbo) | 08/2026 | 49.2 GB; its text encoder alone is 32.7 GB |
| Z-Image-Edit | arXiv 2025 | Weights not released (README: "to be released") |
| LongCat-Image-Edit | 12/2025 | 29.3 GB; its 16.6 GB text encoder does not fit in one T4 |
| HiDream-O1-Image (and Dev) | 05/2026 | 8.8B stored in fp32 (35 GB download), 17.6 GB in fp16; custom script, torch >= 2.10, flash-attn on by default |
| Qwen-Image-Edit-2511, Step1X-Edit v1.2, FLUX.1 Kontext and Fill, RealRestorer, OmniEraser | 2024-2026 | 33-54 GB in fp16 |
| MiniMax H3-Image, HunyuanImage 3.0 | 2026 | Own serving library, or far too large |
| PowerPaint v2, BrushNet | ECCV 2024 | Gradio-only code with old pins (diffusers 0.27, transformers 4.28, Python 3.9), or community weights only |
| IC-Light | 05/2024 | Removes the photo's background before relighting (RMBG step) and regenerates the content; Gradio-only, old pins |
| Bringing Old Photos Back to Life | CVPR 2020 | Archived; needs a dlib build and Synchronized-BatchNorm copies |
| DeOldify | 2019 | Archived; fastai 1.0.60 and torch 1.11 do not run on Python 3.11 |
| ControlNet Recolor (SDXL) | 2024 | Only a community conversion, no license stated; colourises the whole image only, as DDColor does |
| Training-free methods | | Left out by choice |

## Candidates for later (search of 2026-11-21)

Checked against their HF file lists, READMEs and venue pages; they fit 2 x T4 in fp16 but need more work than the models added above.

| Model | Date | Task | Weights | Why not yet |
|---|---|---|---|---|
| CLIPAway | NeurIPS 2024 | Remove | about 6 GB in fp16 (SD 1.5 inpainting + IP-Adapter + AlphaCLIP); all files in the HF Space yigitekin/CLIPAway | Own venv (diffusers 0.29, torch 2.3), clone AlphaCLIP. A distinct approach (removal in CLIP space) |
| SmartEraser | CVPR 2025 | Remove | 5.5 GB fp32 on hf:creative-graphic-design/smarteraser-checkpoints (**third-party mirror**, not checked byte for byte against the authors' Drive files) | Own venv (diffusers 0.30), custom pipeline |
| Attentive Eraser | AAAI 2025 (oral) | Remove | SDXL base | Training-free (left out by choice); a diffusers community pipeline, so easy |
| SD3 ControlNet Inpainting (alimama-creative) | 2024 release | Replace, background | ControlNet about 2.1 + SD3-Medium (gated, non-commercial research licence) | Needs an HF token with access; the better candidate for backgrounds, where SDXL Inpainting fails |
| HD-Painter | ICLR 2025 | Add/Replace | SD 1.5 inpainting | Training-free; old pins (torch 2.1, mmcv, xformers) |
| Diffree | arXiv 2024 | Add | 7.7 GB .ckpt (SD 1.5) | Predicts its own placement mask, which could replace Qwen's for Add; very old pins (torch 2.0, transformers 4.19) |
| UltraEdit (region) | NeurIPS 2024 | Add/Replace | SD3-Medium, 31 GB stored fp32 | Bundled diffusers fork; T5 on its own GPU |

Rejected in that search: ASUKA (CVPR 2025) and GeoRemover (NeurIPS 2025), FLUX.1-Fill only; Add-it (ICLR 2025), FLUX.1-dev in bfloat16; BrushEdit, MagicQuill (CVPR 2025), InstructAny2Pix: 32 to 66 GB or Gradio-only; LayerDecomp, ObjectDrop, EraDiff, MagicEraser: no weights; TurboClear, MiniMax-Remover, ROSE: speed or video variants of models we have.

## How to run (summary)

**Samples.**
- About 30 samples per task from the dev split, fixed seed, about 240 in total.
- Generalists run on all of them. A specialist runs only on its task's samples.

**Inputs.**
- The same `instruction` for every model.
- The original image is resized to a long side of 1024 px, or to the model's own size where the list says so.
- Seed 0.

**Outputs.**
- Images go to `work/editors/<model>/<sample id>.png`.
- Masks go to `work/editors/masks/<sample id>.png`.
- One line per sample goes to `work/editors/<model>/runs.jsonl`: status (`ok`, `failed`, `no-mask`), seconds per image, peak GPU memory, settings.

**Order of work.**
1. Smoke run on 5 samples: it loads in fp16, fits, gives no black images, and gives seconds per image.
2. Full run. Every Kaggle run needs the coordinator's approval first.
3. Scoring with our scorer.

**Kaggle notes.**
- Turn internet on, and add an HF token as a secret for gated models.
- Large downloads: keep the Hugging Face cache outside `/kaggle/working`, which has a size limit.
- Pickle weights (`.bin`, `.pt`, `.pth`, `.pkl`) come only from the sources listed here.
- Models with old pins get their own venv, as the filters and scorer baselines already do.

**Never publish outputs of samples that show identifiable people.**

Lists 2 and 3 run with `src/editors` and `notebooks/editors_remove_add.ipynb` (2 x T4, fp16). Times are the median seconds per image after the first one; memory is torch's peak.

| Model | Member | Smoke | Full run |
|---|---|---|---|
| LaMa (big-lama) | Kiên Vũ | ok (2026-11-20) | remove, 30 samples (2026-11-21): 28 ok, 2 no mask; 0.66 s, 1.1 GB. Small objects fine; large regions smeared |
| MI-GAN | Kiên Vũ | ok (2026-11-21) | remove, 30 samples (2026-11-21): 28 ok, 2 no mask; 0.09 s, 0.24 GB. On par with LaMa; large regions smeared |
| RORem | Kiên Vũ | ok (2026-11-20) | remove, 30 samples (2026-11-21): 28 ok, 2 no mask; 7.4 s, 7.7 GB. Good |
| ObjectClear | Kiên Vũ | ok (2026-11-20) | remove, 30 samples (2026-11-21): 28 ok, 2 no mask; 9.1 s, 9.0 GB. Good; sometimes writes stray text or smudges when removing text |
| OSOR-SDXL-Inpainting | Kiên Vũ | ok (2026-11-21; fp16 works, no black images) | remove, 30 samples (2026-11-21): 28 ok, 2 no mask; 0.86 s, 8.2 GB. Cleanest of the five on the contact sheets (cone, people at a waterfall, leash) and 10 x faster than RORem |
| SDXL Inpainting 0.1 (Add/Replace) | Kiên Vũ | ok (2026-11-21, plan_v3) | add_replace, 30 samples (2026-11-21): 29 ok, 1 no mask; 13.7 s, 9.6 GB. Good on Replace of small objects and text (sapphire to diamond, "Moonlight" to "Son of Saul") and on anchored additions (beard, hat); strength 1.0 for backgrounds fixed the ignored prompt. Weak where Qwen's free Add box is wrong |
| Paint by Inpaint | Kiên Vũ | ok (2026-11-21) | add_replace, 14 Add samples (2026-11-21): 14 ok; 8.9 s, 2.8 GB. Places the new thing itself, often better than a wrong Add box (beach ball, ghosts); sometimes changes the whole photo |
| EraseDraw v0.2 | Kiên Vũ | ok (2026-11-21) | add_replace, 14 Add samples (2026-11-21): 14 ok; 1.9 s, 2.8 GB. Fast; additions often small or misplaced, sometimes off-colour |
| Grounding DINO base + SAM 2.1 large | Kiên Vũ | ok | remove: 28 of 30 masks (2 not found: "three blotches", "the stand"); add_replace: 29 of 30 (Qwen named no target for a logo). Rules v3 (2026-11-21): additions worn on a face or head are anchored on it; Add boxes capped at 30% of the image; a background plan with nothing to keep is asked again |

Findings of the full runs (2026-11-21; no scores yet, the scorer is built separately: judged on the contact sheets):
- Remove: OSOR first (fast and clean), then RORem and ObjectClear; LaMa and MI-GAN are the fast baselines.
- Add/Replace: SDXL Inpainting for Replace and anchored additions; for free additions the mask-free models (Paint by
  Inpaint, EraseDraw) avoid the VLM's poor placement.
- Most remaining errors come from the plan, not the editors: Qwen sometimes reads an addition as a replacement
  ("Photoshop Ronald McDonald into the photo" replaced the young man), adds extra targets ("the dog" for "remove
  the leash", "the Christmas tree" for "make the dog look like Dobby"), or places a free Add box badly (a Rolls
  Royce box of 1% at the man's feet).
