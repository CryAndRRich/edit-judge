"""VAREdit 2B (Mao et al., ICLR 2026; MIT): instruction editing with a visual autoregressive model (Infinity 2B,
next-scale prediction over bitwise tokens) conditioned on the source image and a flan-t5-xl text encoding.

The repository has no package to install: its code is downloaded at COMMIT into EJ_MODELS_DIR and put on the path;
its imports (timm, einops, pytz, imageio, opencv, pandas, PyYAML) are already on Kaggle, so it runs in the main
environment. Weights: `2B-512.pth` and `flan-t5-xl/` of HiDream-ai/VAREdit at REVISION, the VAE
`infinity_vae_d32reg.pth` of FoundationVision/Infinity at VAE_REVISION (infer.py's MODEL_CONFIGS["2B"]), all in
one folder of EJ_MODELS_DIR (21.8 GB: point EJ_MODELS_DIR outside /kaggle/working, which holds 20 GB).

Settings of the README: cfg 3.0, tau 0.1 (infer.py's defaults are 4.0 and 0.5, its example uses tau 1.0); the
rest as in infer.py's generate_image (512 x 512, top-k 900, top-p 0.97, guidance on the logits). The 2B model works
at 512 x 512 only: the repository's own `transform` pads the image to a square (white, at the bottom or right) and
its crop function resizes the output to the padded size and crops the content back, so the output has the input's
size. The prompt is the instruction.

Changes for the T4 (no bf16, no flash-attn), none of which changes the model:
- the repository casts the transformer blocks to bf16 and runs under bf16 autocast (tools/run_infinity.py lines
  145, 214, 236-237); the downloaded code is patched to fp16 (PATCHES). The text encoder is fp16 as upstream
  (load_tokenizer), where transformers keeps T5's `wo` layers in fp32 and clamps fp16 overflows;
- the checkpoint is loaded on the CPU (line 247 loads it straight onto the GPU next to the fp32 model: 17.6 GB);
- infinity/models/basic.py imports flash_attn (lines 18-19), used for the text cross-attention: a stand-in module
  computes it with PyTorch's scaled_dot_product_attention, one sequence at a time;
- the fused adaptive layer norm (infinity/models/fused_op.py, torch.compile) is replaced by the same computation
  in eager mode (the non-fused path of CrossAttnBlock does not pass `scale_ind`, so it cannot be used instead);
- the VAE decoder, which upstream runs inside the bf16 autocast, runs in fp32 (fp16 could overflow).
"""

import sys

from .. import TASKS
from ..base import Editor

REPO = "HiDream-ai/VAREdit"
COMMIT = "31c9868853577c8640a8a348ae3f9d5e1b27d9b7"
MODEL = "HiDream-ai/VAREdit"
REVISION = "290181d9b28d6f98ea47afa9cde6b78c48f8145a"
VAE_MODEL = "FoundationVision/Infinity"
VAE_REVISION = "e7108923249fb3c836ebfbc3f4b4d318d6131293"
CHECKPOINT = "2B-512.pth"
VAE_FILE = "infinity_vae_d32reg.pth"
SIZE = 512
CFG, TAU = 3.0, 0.1
PATCHES = (     # (file, text, replacement, expected count) applied to the downloaded code
    ("tools/run_infinity.py", "dtype=torch.bfloat16", "dtype=torch.float16", 3),
    ("tools/run_infinity.py", "block.bfloat16()", "block.half()", 1),
    ("tools/run_infinity.py", "torch.load(model_path, map_location=device)",
     "torch.load(model_path, map_location='cpu')", 1),
)


def repository_code():
    """The repository at COMMIT with PATCHES applied, unpacked once into EJ_MODELS_DIR; returns its folder."""
    import tarfile
    import urllib.request

    from ...paths import MODELS_DIR
    folder = MODELS_DIR / f"VAREdit-{COMMIT}"
    marker = folder / ".fp16-patched"
    if not marker.exists():
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        archive = MODELS_DIR / f"VAREdit-{COMMIT[:7]}.tar.gz"
        urllib.request.urlretrieve(f"https://github.com/{REPO}/archive/{COMMIT}.tar.gz", archive)
        with tarfile.open(archive) as tar:
            try:
                tar.extractall(MODELS_DIR, filter="data")
            except TypeError:                         # Python before 3.11.4: no extraction filters
                tar.extractall(MODELS_DIR)
        archive.unlink()
        for relative, text, replacement, count in PATCHES:
            path = folder / relative
            source = path.read_text()
            if source.count(text) != count:
                raise RuntimeError(f"{relative}: expected {count} x {text!r}, found {source.count(text)}")
            path.write_text(source.replace(text, replacement))
        marker.write_text("\n".join(text for _, text, _, _ in PATCHES) + "\n")
    return folder


def weights_folder():
    """One folder with the checkpoint, flan-t5-xl/ and the VAE, as infer.py's `pretrain_root` expects."""
    from huggingface_hub import hf_hub_download, snapshot_download

    from ...paths import MODELS_DIR
    folder = MODELS_DIR / f"VAREdit-weights-{REVISION[:7]}"
    snapshot_download(MODEL, revision=REVISION, allow_patterns=[CHECKPOINT, "flan-t5-xl/*"], local_dir=folder)
    hf_hub_download(VAE_MODEL, VAE_FILE, revision=VAE_REVISION, local_dir=folder)
    return folder


def unavailable_flash_attn(*arguments, **options):
    raise NotImplementedError("flash-attn is not available on the T4 (VAREdit uses it only with customized_flash_attn)")


def sdpa_varlen_kvpacked(**arguments):
    """flash_attn_varlen_kvpacked_func with PyTorch attention: `q` (total_q, heads, dim) and `kv`
    (total_k, 2, heads, dim) hold the sequences packed one after the other, bounded by cu_seqlens_q / cu_seqlens_k;
    returns (total_q, heads, dim)."""
    import torch
    import torch.nn.functional as functional

    queries, packed = arguments["q"], arguments["kv"]
    query_bounds = arguments["cu_seqlens_q"].tolist()
    key_bounds = arguments["cu_seqlens_k"].tolist()
    outputs = []
    for index in range(len(query_bounds) - 1):
        query = queries[query_bounds[index]:query_bounds[index + 1]].transpose(0, 1)        # heads, length, dim
        keys_values = packed[key_bounds[index]:key_bounds[index + 1]]
        keys, values = keys_values[:, 0].transpose(0, 1), keys_values[:, 1].transpose(0, 1)
        attended = functional.scaled_dot_product_attention(query, keys.to(query.dtype), values.to(query.dtype),
                                                           scale=arguments.get("softmax_scale"))
        outputs.append(attended.transpose(0, 1))
    return torch.cat(outputs, dim=0)


def eager_ada_layer_norm(**arguments):
    """infinity/models/fused_op.py:fused_ada_layer_norm without torch.compile (same keyword arguments)."""
    import torch.nn.functional as functional

    hidden = functional.layer_norm(arguments["x"].float(), normalized_shape=(arguments["C"],),
                                   eps=arguments["eps"])
    return hidden.mul(arguments["scale"].add(1)).add_(arguments["shift"])


def install_flash_attn_stand_in():
    """A `flash_attn` module for basic.py's imports; its fused ops stay unavailable (basic.py catches that)."""
    import importlib.machinery
    import types

    module = types.ModuleType("flash_attn")
    # importlib.util.find_spec raises ValueError for a module without a spec (transformers calls it while importing
    # T5); with one, transformers finds no installed flash_attn distribution and keeps its own attention
    module.__spec__ = importlib.machinery.ModuleSpec("flash_attn", None)
    module.flash_attn_func = unavailable_flash_attn
    module.flash_attn_varlen_kvpacked_func = sdpa_varlen_kvpacked
    sys.modules["flash_attn"] = module


class VAREdit2B(Editor):
    name = "varedit-2b"
    local_dirs = (f"VAREdit-{COMMIT}", f"VAREdit-weights-{REVISION[:7]}")
    download_gb = 22.0
    model_id = f"{MODEL}@{REVISION[:7]} {CHECKPOINT} + {VAE_MODEL}@{VAE_REVISION[:7]} (code {COMMIT[:7]})"
    tasks = TASKS
    uses_mask = False
    settings = {"size": SIZE, "pad": "square, white", "cfg": CFG, "tau": TAU, "top_k": 900, "top_p": 0.97,
                "precision": "fp16 (VAE decoder fp32)", "prompt": "instruction"}

    def fetch(self) -> None:
        repository_code()
        weights_folder()

    def load(self, device: str) -> None:
        import numpy as np
        import torch

        if not device.startswith("cuda"):
            raise RuntimeError("VAREdit's code runs on CUDA only")
        if torch.device(device).index is not None:
            torch.cuda.set_device(device)         # the repository puts everything on the current CUDA device
        self.torch = torch
        sys.path.insert(0, str(repository_code()))
        install_flash_attn_stand_in()
        import infinity.models.infinity as infinity_module
        infinity_module.fused_ada_layer_norm = eager_ada_layer_norm
        from infer import load_model, transform
        from tools.run_infinity import dynamic_resolution_h_w, gen_one_img, h_div_w_templates

        self.transform, self.generate = transform, gen_one_img
        weights = weights_folder()
        components = load_model(str(weights), str(weights / CHECKPOINT), "2B", SIZE)
        self.args, self.model, self.vae, self.tokenizer, self.text_encoder = components[:5]
        decode = self.vae.decode

        def decode_fp32(codes):
            with torch.autocast("cuda", enabled=False):
                return decode(codes.float())

        self.vae.decode = decode_fp32
        template = h_div_w_templates[np.argmin(np.abs(h_div_w_templates - 1.0))]
        scales = dynamic_resolution_h_w[template]["0.25M"]["scales"]
        self.schedule = [(1, height, width) for (_, height, width) in scales]

    def edit(self, image, mask, prompt: str, seed: int = 0):
        import numpy as np
        from PIL import Image

        pixels, crop_back = self.transform(image, SIZE)
        with self.torch.no_grad():
            generated = self.generate(
                self.model, self.vae, self.tokenizer, self.text_encoder, prompt, pixels, g_seed=seed, gt_leak=0,
                gt_ls_Bl=None, cfg_list=CFG, tau_list=TAU, scale_schedule=self.schedule,
                cfg_insertion_layer=[self.args.cfg_insertion_layer], vae_type=self.args.vae_type,
                sampling_per_bits=self.args.sampling_per_bits, enable_positive_prompt=0,
                apply_spatial_patchify=self.args.apply_spatial_patchify)
        rgb = np.ascontiguousarray(generated.cpu().numpy()[..., ::-1])     # the model returns BGR (for cv2)
        return crop_back(Image.fromarray(rgb.astype(np.uint8)))
