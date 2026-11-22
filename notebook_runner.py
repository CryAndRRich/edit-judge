"""Enhance models benchmark on Kaggle (MODEL_LIST.md, list 4): one self-contained script, pushed with
`kaggle kernels push` (kernel-metadata.json). It needs the dataset cryandrrich/editjudge as input.

  orchestrator (this file, no arguments): picks the samples, then runs every model in its own worker process
      (python notebook_runner.py --worker NAME), because the repositories use clashing top-level package names
      (`models` in VOSR and InstructIR) and a crash or an out-of-memory in one model must not stop the others;
  worker: loads one model, runs it on the selected samples and writes, under /kaggle/working/out/work/editors/:
      <model>/<sample_id>.png      the output image (same size as the input image)
      <model>/runs.jsonl           one line per sample: status (ok, failed, skipped, unsupported), seconds, peak GPU
                                   memory, settings, and the black / non-finite checks
      <model>/meta.json            load time and GPU, once per run
      samples.json                 the selected samples (id, instruction, low-light flag)
      _inputs/<sample_id>.png      the inputs as the models saw them

Edit MODE below: "smoke" (5 samples) or "full" (about 30). Weights and repositories go to /tmp (not saved as output).
"""

import argparse
import glob
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import traceback
from pathlib import Path

MODE = "full"                       # "smoke": 5 samples; "full": FULL_SAMPLES samples
FULL_SAMPLES = 30
SMOKE_SAMPLES = 5
MODELS = ["realesrgan", "hvi_cidnet", "instructir", "vosr"]    # AceTone dropped: needs a reference photo
SEED = 0
LONG_SIDE = 1024                    # inputs larger than this are shrunk to it (never enlarged)

OUT = Path("/kaggle/working/out") if Path("/kaggle").exists() else Path("out")
EDITORS = OUT / "work" / "editors"
CACHE = Path(os.environ.get("EJ_CACHE", "/tmp/ej-cache"))
REPOS = CACHE / "repos"

# Instructions that ask for more light. HVI-CIDNet is a low-light model and brightens whatever it gets, so it only
# runs on these (MODEL_LIST.md).
LOW_LIGHT = re.compile(r"\b(dark\w*|dim\w*|bright\w*|lighten\w*|low[- ]light|underexpos\w*|exposure|night\w*|"
                       r"illuminat\w*|lighting|adjust (the )?light)\b", re.I)    # a keyword heuristic, checked by eye


def is_low_light(text: str) -> bool:
    return bool(LOW_LIGHT.search(text or ""))


# ------------------------------------------------------------------ samples


def find_dataset() -> Path:
    for pattern in ("/kaggle/input/*/pairs.jsonl", "/kaggle/input/*/*/pairs.jsonl", "/kaggle/input/*/*/*/pairs.jsonl"):
        found = sorted(glob.glob(pattern))
        if found:
            return Path(found[0]).parent
    local = os.environ.get("EJ_DATA_DIR")
    if local and (Path(local) / "pairs.jsonl").exists():
        return Path(local)
    raise SystemExit("no input with pairs.jsonl: add the dataset cryandrrich/editjudge")


def split_of(sample_id: str) -> str:            # the same rule as src/scorer/data.py
    position = int(hashlib.sha1(sample_id.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "train" if position < 0.70 else "dev" if position < 0.85 else "test"


def select_samples(dataset: Path, count: int, smoke: bool) -> list[dict]:
    """Samples of the dev split with the category Enhance (any subject), in a fixed hash order. The smoke set is
    the first 4 plus the first low-light sample, so HVI-CIDNet is exercised too."""
    samples = {}
    with open(dataset / "pairs.jsonl", encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            sample_id = row["a"].split("/")[0]
            if sample_id in samples or split_of(sample_id) != "dev":
                continue
            if any(category.startswith("Enhance") for category in row.get("categories") or []):
                text = " ".join((row.get("instruction") or row["request"]).split())
                samples[sample_id] = {"id": sample_id, "instruction": text, "request": row["request"],
                                      "categories": row["categories"], "low_light": is_low_light(text)}
    ordered = sorted(samples.values(), key=lambda s: hashlib.sha1(f"order:{s['id']}".encode()).hexdigest())
    if not smoke:
        return ordered[:count]
    chosen = ordered[:count - 1]
    extra = next((s for s in ordered[count - 1:] if s["low_light"]), ordered[count - 1])
    return chosen + [extra] if not any(s["low_light"] for s in chosen) else ordered[:count]


def prepare_inputs(dataset: Path, samples: list[dict]) -> None:
    from PIL import Image, ImageOps
    inputs = EDITORS / "_inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    for sample in samples:
        with Image.open(dataset / "images" / sample["id"] / "original.jpg") as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
        scale = LONG_SIDE / max(image.size)
        if scale < 1:
            image = image.resize((round(image.width * scale), round(image.height * scale)), Image.LANCZOS)
        image.save(inputs / f"{sample['id']}.png")
        sample["size"] = list(image.size)


# ------------------------------------------------------------------ helpers shared by the workers


def sh(command: list[str], **kwargs) -> None:
    subprocess.run(command, check=True, **kwargs)


def clone(url: str, name: str, branch: str | None = None) -> Path:
    folder = REPOS / name
    if not folder.exists():
        REPOS.mkdir(parents=True, exist_ok=True)
        sh(["git", "clone", "--depth", "1", *(["--branch", branch] if branch else []), url, str(folder)])
    return folder


def hf_download(repo: str, **kwargs) -> str:
    from huggingface_hub import snapshot_download
    return snapshot_download(repo, cache_dir=str(CACHE / "hf"), **kwargs)


def pad_to_multiple(array, factor: int):
    """Reflect-pad an HxWxC array so both sides are multiples of `factor`; returns it and the original size."""
    import numpy as np
    h, w = array.shape[:2]
    pad_h, pad_w = (-h) % factor, (-w) % factor
    if pad_h or pad_w:
        array = np.pad(array, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
    return array, (h, w)


def tensor_to_pil(tensor):
    """A 3xHxW float tensor in [0, 1] -> (PIL image, whether it had NaN or inf)."""
    import torch
    from PIL import Image
    tensor = tensor.detach().float().cpu()
    nonfinite = not bool(torch.isfinite(tensor).all())
    array = (torch.nan_to_num(tensor, nan=0.0, posinf=1.0, neginf=0.0).clamp(0, 1) * 255).round().byte()
    return Image.fromarray(array.permute(1, 2, 0).numpy()), nonfinite


def pil_to_tensor(image, device, dtype=None):
    import numpy as np
    import torch
    tensor = torch.from_numpy(np.asarray(image, dtype=np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
    return tensor.to(dtype) if dtype else tensor


# ------------------------------------------------------------------ models: load(device) and run(image, text)


class RealESRGAN:
    """Real-ESRGAN x4plus through spandrel (the `realesrgan` and `basicsr` packages break on new torchvision).
    The x4 output is shrunk back to the input size, so every model's output is comparable. fp16, tiles of 256."""
    name = "realesrgan"
    settings = {"weights": "RealESRGAN_x4plus.pth (release v0.1.0)", "native_scale": 4, "dtype": "fp16",
                "tile": 256, "tile_pad": 16, "output": "shrunk to the input size (LANCZOS)"}
    URL = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"

    def load(self, device):
        import torch
        from spandrel import ModelLoader
        path = CACHE / "RealESRGAN_x4plus.pth"
        if not path.exists():
            CACHE.mkdir(parents=True, exist_ok=True)
            sh(["curl", "-L", "--fail", "-s", "-o", str(path), self.URL])
        self.model = ModelLoader().load_from_file(path).to(device).eval().half()
        self.device = device
        self.torch = torch

    def run(self, image, text):
        torch = self.torch
        x = pil_to_tensor(image, self.device, torch.float16)
        _, _, h, w = x.shape
        tile, pad, scale = self.settings["tile"], self.settings["tile_pad"], 4
        out = torch.zeros(1, 3, h * scale, w * scale, device=self.device, dtype=torch.float16)
        with torch.no_grad():
            for top in range(0, h, tile):
                for left in range(0, w, tile):
                    t0, l0 = max(top - pad, 0), max(left - pad, 0)
                    t1, l1 = min(top + tile + pad, h), min(left + tile + pad, w)
                    y = self.model(x[:, :, t0:t1, l0:l1])
                    bottom, right = min(top + tile, h), min(left + tile, w)
                    y = y[:, :, (top - t0) * scale:(bottom - t0) * scale, (left - l0) * scale:(right - l0) * scale]
                    out[:, :, top * scale:bottom * scale, left * scale:right * scale] = y
        result, nonfinite = tensor_to_pil(out[0])
        from PIL import Image
        return result.resize(image.size, Image.LANCZOS), nonfinite


class HVICIDNet:
    """HVI-CIDNet, "generalization" weights, alpha_s = alpha_i = gamma = 1 (eval_hf.py defaults). fp32 (1.9 M
    parameters). Runs only on low-light instructions."""
    name = "hvi_cidnet"
    settings = {"weights": "Fediory/HVI-CIDNet-Generalization", "alpha_s": 1.0, "alpha_i": 1.0, "gamma": 1.0,
                "dtype": "fp32", "only_low_light": True}

    def wants(self, sample):
        return sample["low_light"]

    def load(self, device):
        import torch
        import safetensors.torch as sf
        repo = clone("https://github.com/Fediory/HVI-CIDNet.git", "HVI-CIDNet")
        sys.path.insert(0, str(repo))
        from net.CIDNet import CIDNet
        self.model = CIDNet().to(device).eval()
        weights = Path(hf_download("Fediory/HVI-CIDNet-Generalization", allow_patterns=["model.safetensors", "config.json"]))
        print(self.model.load_state_dict(sf.load_file(weights / "model.safetensors"), strict=False), flush=True)
        self.device, self.torch = device, torch

    def run(self, image, text):
        import numpy as np
        torch = self.torch
        array, (h, w) = pad_to_multiple(np.asarray(image), 8)
        from PIL import Image
        x = pil_to_tensor(Image.fromarray(array), self.device)
        with torch.no_grad():
            self.model.trans.alpha_s = self.settings["alpha_s"]
            self.model.trans.alpha = self.settings["alpha_i"]
            self.model.trans.gated = True
            self.model.trans.gated2 = True
            y = self.model(x ** self.settings["gamma"])
        return tensor_to_pil(y[0, :, :h, :w])


class InstructIR:
    """InstructIR (7 degradations), the instruction as the prompt. fp32 (0.06 GB + a small text encoder)."""
    name = "instructir"
    settings = {"weights": "marcosv/InstructIR (im_instructir-7d.pt, lm_instructir-7d.pt)", "text_encoder":
                "TaylorAI/bge-micro-v2", "dtype": "fp32", "padding": "reflect to a multiple of 16"}

    def load(self, device):
        import argparse
        import torch
        import yaml
        repo = clone("https://github.com/mv-lab/InstructIR.git", "InstructIR")
        sys.path.insert(0, str(repo))
        from models import instructir
        from text.models import LanguageModel, LMHead

        def namespace(d):
            return argparse.Namespace(**{k: namespace(v) if isinstance(v, dict) else v for k, v in d.items()})

        cfg = namespace(yaml.safe_load(open(repo / "configs" / "eval5d.yml")))
        weights = Path(hf_download("marcosv/InstructIR", allow_patterns=["*.pt"]))
        self.model = instructir.create_model(input_channels=cfg.model.in_ch, width=cfg.model.width,
                                             enc_blks=cfg.model.enc_blks, middle_blk_num=cfg.model.middle_blk_num,
                                             dec_blks=cfg.model.dec_blks, txtdim=cfg.model.textdim).to(device)
        self.model.load_state_dict(torch.load(weights / "im_instructir-7d.pt", map_location="cpu"), strict=True)
        self.model.eval()
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        self.lm = LanguageModel(model=cfg.llm.model)
        self.head = LMHead(embedding_dim=cfg.llm.model_dim, hidden_dim=cfg.llm.embd_dim,
                           num_classes=cfg.llm.nclasses).to(device)
        self.head.load_state_dict(torch.load(weights / "lm_instructir-7d.pt", map_location="cpu"), strict=True)
        self.head.eval()
        self.device, self.torch = device, torch

    def run(self, image, text):
        import numpy as np
        from PIL import Image
        torch = self.torch
        array, (h, w) = pad_to_multiple(np.asarray(image), 16)
        x = pil_to_tensor(Image.fromarray(array), self.device)
        with torch.no_grad():
            text_embedding, _ = self.head(self.lm(text).to(self.device))
            y = self.model(x, text_embedding)
        return tensor_to_pil(y[0, :, :h, :w])


class VOSR2:
    """VOSR 2.0 one-step, as inference_vosr_onestep.py -c preset/ckpts/VOSR2 -u 1 --tile_size 512, in fp32 (the
    script has no dtype flag). The model is loaded once and the script's own functions are reused."""
    name = "vosr"
    settings = {"weights": "CSWRY/VOSR (VOSR2, Qwen-Image-vae-2d, torch_cache)", "upscale": 1, "tile_size": 512,
                "tile_overlap": 32, "infer_steps": 1, "align_method": "wavelet", "dtype": "fp32",
                "padding": "reflect to a multiple of 16"}

    def load(self, device):
        import torch
        repo = clone("https://github.com/cswry/VOSR.git", "VOSR")
        hf_download("CSWRY/VOSR", local_dir=str(repo / "preset" / "ckpts"),
                    allow_patterns=["VOSR2/*", "Qwen-Image-vae-2d/*", "torch_cache/*"])
        self.home = os.getcwd()
        os.chdir(repo)                                  # the script's paths (preset/ckpts/...) are relative
        sys.path.insert(0, str(repo))
        import inference_vosr_onestep as vosr_script
        from models.lightningdit import LightningDiT
        from models.qwenimage_vae2d import AutoencoderKLQwenImage2D
        from safetensors.torch import load_file

        checkpoint = Path("preset/ckpts/VOSR2")
        args = argparse.Namespace(**json.load(open(checkpoint / "args.json")))
        for key, value in {"upscale": 1, "process_size": 512, "align_method": "wavelet", "tile_size": 512,
                           "tile_overlap": 32, "vae_tile_size": None, "vae_tile_overlap": None,
                           "posterior_mode": True, "infer_steps": 1, "seed": SEED}.items():
            setattr(args, key, value)
        args.auxiliary_time_cond = bool(getattr(args, "auxiliary_time_cond", None) is not False)
        if args.ae_type != "qwen":
            raise SystemExit(f"VOSR2 expected the qwen VAE, got {args.ae_type}")
        self.vae = AutoencoderKLQwenImage2D.from_pretrained("preset/ckpts/Qwen-Image-vae-2d").to(device).eval()
        self.venc = vosr_script.load_dinov2(args, device)
        self.model = LightningDiT(
            input_size=args.resolution // 8, patch_size=args.patch_size, in_channels=2 * 16, out_channels=16,
            hidden_size=args.dim, depth=args.depth, num_heads=args.num_heads, mlp_ratio=args.mlp_ratio,
            z_dims=args.enc_dim, encdim_ratio=args.encdim_ratio, auxiliary_time_cond=args.auxiliary_time_cond,
            use_qknorm=args.use_qknorm, use_swiglu=args.use_swiglu, use_rope=args.use_rope,
            use_rmsnorm=args.use_rmsnorm, wo_shift=args.wo_shift, num_fused_layers=len(args.layer_dinov2b_list))
        weights = next(p for p in (checkpoint / "clean_weights" / "ema_model.safetensors",
                                   checkpoint / "checkpoints" / "ema_model.safetensors",
                                   checkpoint / "ema_model.safetensors") if p.is_file())
        print(self.model.load_state_dict(load_file(str(weights)), strict=False), flush=True)
        self.model.to(device).eval()
        self.model.forward = self.model.forward_flexible
        self.args, self.script, self.device, self.torch = args, vosr_script, device, torch

    def run(self, image, text):
        import numpy as np
        from PIL import Image
        torch = self.torch
        array, (h, w) = pad_to_multiple(np.asarray(image), 16)
        padded = Image.fromarray(array)
        lq = pil_to_tensor(padded, self.device) * 2.0 - 1.0
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        with torch.no_grad():
            sr = self.script.tiled_latent_inference(self.model, self.vae, self.venc, lq, self.args,
                                                    device=self.device, light_decoder=None)
        out, nonfinite = tensor_to_pil(sr[0] * 0.5 + 0.5)
        out = self.script.wavelet_color_fix(out, padded)
        return out.crop((0, 0, w, h)), nonfinite

    def close(self):
        os.chdir(self.home)


class AceTone:
    """Not runnable on text. The only released checkpoint, Vivre/AceTone-3B-PST-Preview ("PST" = photo style
    transfer), is conditioned on the raw photo AND a reference photo ("mimic the toning method", demo/predict_lut.py,
    eval/predict_lut_ddp.py); the repository releases no inference path that reads an instruction, and our samples
    have no reference photo (using a human edit as the reference would leak the answer). The worker records
    `unsupported` for every sample instead of inventing a way to run it."""
    name = "acetone"
    settings = {"weights": "Vivre/AceTone-3B-PST-Preview", "needs": "a reference image, not an instruction"}
    unsupported = ("the released checkpoint (PST-Preview) takes a reference photo, not a text instruction; "
                   "no text-conditioned inference exists in gh:martian422/AceTone")


MODEL_CLASSES = {cls.name: cls for cls in (RealESRGAN, HVICIDNet, InstructIR, VOSR2, AceTone)}


# ------------------------------------------------------------------ worker and orchestrator


def worker(name: str) -> int:
    import torch
    from PIL import Image
    import numpy as np

    samples = json.loads((EDITORS / "samples.json").read_text())["samples"]
    folder = EDITORS / name
    folder.mkdir(parents=True, exist_ok=True)
    runs = folder / "runs.jsonl"
    runs.unlink(missing_ok=True)
    model = MODEL_CLASSES[name]()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base = {"model": name, "settings": model.settings, "seed": SEED}

    def log(row):
        with open(runs, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({**base, **row}, ensure_ascii=False) + "\n")

    if getattr(model, "unsupported", None):
        for sample in samples:
            log({"sample_id": sample["id"], "status": "unsupported", "error": model.unsupported})
        return 0

    started = time.monotonic()
    model.load(device)
    if device == "cuda":
        torch.cuda.synchronize()
    meta = {"model": name, "device": device, "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
            "load_seconds": round(time.monotonic() - started, 1),
            "load_gpu_gb": round(torch.cuda.memory_allocated() / 1e9, 3) if device == "cuda" else None,
            "torch": torch.__version__}
    (folder / "meta.json").write_text(json.dumps(meta, indent=1))
    print(f"[{name}] loaded: {meta}", flush=True)

    failures, first = 0, True
    for sample in samples:
        row = {"sample_id": sample["id"], "instruction": sample["instruction"], "low_light": sample["low_light"]}
        if hasattr(model, "wants") and not model.wants(sample):
            log({**row, "status": "skipped", "error": "not a low-light instruction"})
            continue
        image = Image.open(EDITORS / "_inputs" / f"{sample['id']}.png").convert("RGB")
        try:
            if device == "cuda":
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
            t0 = time.monotonic()
            output, nonfinite = model.run(image, sample["instruction"])
            if device == "cuda":
                torch.cuda.synchronize()
            seconds = time.monotonic() - t0
            output.save(folder / f"{sample['id']}.png")
            mean_out = float(np.asarray(output, dtype=np.float32).mean())
            mean_in = float(np.asarray(image, dtype=np.float32).mean())
            log({**row, "status": "ok", "seconds": round(seconds, 3), "first_image": first,
                 "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 1e9, 3) if device == "cuda" else None,
                 "peak_reserved_gb": round(torch.cuda.max_memory_reserved() / 1e9, 3) if device == "cuda" else None,
                 "nonfinite": nonfinite, "black": mean_out < 2.0 <= mean_in, "mean_in": round(mean_in, 2),
                 "mean_out": round(mean_out, 2), "input_size": list(image.size), "output_size": list(output.size)})
        except Exception as error:                       # an out-of-memory or a model bug: log it, go on
            failures += 1
            traceback.print_exc()
            log({**row, "status": "failed", "error": f"{type(error).__name__}: {str(error)[:300]}",
                 "oom": "out of memory" in str(error).lower()})
            if device == "cuda":
                torch.cuda.empty_cache()
        first = False
        print(f"[{name}] {sample['id']} done", flush=True)
    if hasattr(model, "close"):
        model.close()
    return 1 if failures == len(samples) else 0


def orchestrate() -> None:
    EDITORS.mkdir(parents=True, exist_ok=True)
    smoke = MODE == "smoke"
    count = SMOKE_SAMPLES if smoke else FULL_SAMPLES
    dataset = find_dataset()
    samples = select_samples(dataset, count, smoke)
    prepare_inputs(dataset, samples)
    (EDITORS / "samples.json").write_text(json.dumps(
        {"mode": MODE, "seed": SEED, "split": "dev", "category": "Enhance*", "long_side": LONG_SIDE,
         "samples": samples}, ensure_ascii=False, indent=1))
    print(f"mode {MODE}: {len(samples)} samples, {sum(s['low_light'] for s in samples)} low-light", flush=True)
    sh([sys.executable, "-m", "pip", "install", "-q", "spandrel", "einops", "timm", "safetensors", "fairscale"])
    sh(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv"])
    codes = {}
    for name in MODELS:
        print(f"\n===== {name} =====", flush=True)
        log_file = EDITORS / f"{name}.log"
        with open(log_file, "w") as log:
            process = subprocess.Popen([sys.executable, "-u", os.path.abspath(__file__), "--worker", name],
                                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
                                       env={**os.environ, "HF_HOME": str(CACHE / "hf")})
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
            codes[name] = process.wait()
        print(f"===== {name}: exit {codes[name]} =====", flush=True)
    (EDITORS / "summary.json").write_text(json.dumps({"mode": MODE, "exit_codes": codes}, indent=1))
    print(json.dumps(codes), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker")
    cli = parser.parse_args()
    if cli.worker:
        sys.exit(worker(cli.worker))
    orchestrate()
