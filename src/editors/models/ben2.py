"""BEN2 Base (Prama LLC, arXiv 2501.06230, 01/2025; MIT), the background remover of list 7: a Swin-B segmenter
(MVANet-style, four 512 px patches plus a global view) that predicts an alpha matte of the foreground.

The authors ship the same code twice: the pip package of the GitHub repository (`ben2.BEN_Base.from_pretrained`,
PyTorchModelHubMixin, which reads `model.safetensors`) and the single file `BEN2.py` of the Hugging Face repository
(`BEN2.BEN_Base()` + `loadcheckpoints("BEN2_Base.pth")`, a 1.1 GB checkpoint). No unpinned git install here: `BEN2.py`
and `model.safetensors` (the package's weights, 0.38 GB fp32) are downloaded from the HF repository at REVISION, the
file is imported by path and the weights are loaded with `load_state_dict(strict=True)`. Its dependencies (einops,
timm, opencv, torchvision) are already on Kaggle, so it runs in the main environment.

Settings of the model card: `model.inference(image)` with `refine_foreground=False` (the default; the refinement is
an optional post-processing of the matte edges). Inside, the image is resized to 1024 x 1024 (aspect not kept,
LANCZOS), ImageNet-normalized, and on CUDA the forward runs under `torch.autocast(float16)` with fp32 weights (no
`.half()`, the authors' own fp16 path); the matte is min-max normalized, resized back and returned as an RGBA image
of the input's size. Deterministic: `inference` reseeds everything with 9. The prompt is ignored.
Deviation: the RGBA result is composited on plain white (RGB), since the benchmark compares RGB images.
Importing `BEN2.py` sets process-wide state: cudnn deterministic (benchmark off) and fp32 matmul precision "highest".
"""

from ..base import Editor

REPO = "PramaLLC/BEN2"
REVISION = "e48a20765fb421d19dcdb0bf3cc61e802ca5ec8f"
BACKGROUND = (255, 255, 255)


class BEN2(Editor):
    name = "ben2"
    repos = (REPO,)
    download_gb = 0.4
    model_id = f"{REPO}@{REVISION[:7]}"
    tasks = ("formatting",)
    ops = ("cutout",)
    uses_mask = False
    settings = {"size": 1024, "refine_foreground": False, "autocast": "fp16", "background": "white"}

    def fetch(self) -> None:
        from huggingface_hub import hf_hub_download
        hf_hub_download(REPO, "BEN2.py", revision=REVISION)
        hf_hub_download(REPO, "model.safetensors", revision=REVISION)

    def load(self, device: str) -> None:
        import importlib.util

        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file

        code = hf_hub_download(REPO, "BEN2.py", revision=REVISION)
        weights = hf_hub_download(REPO, "model.safetensors", revision=REVISION)
        spec = importlib.util.spec_from_file_location("ben2_hf", code)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.model = module.BEN_Base()
        self.model.load_state_dict(load_file(weights), strict=True)
        self.model.to(device).eval()

    def edit(self, image, mask, prompt: str, seed: int = 0):
        from PIL import Image

        # inference() writes the alpha into the image it is given: hand it a copy
        cutout = self.model.inference(image.copy(), refine_foreground=False)
        canvas = Image.new("RGB", cutout.size, BACKGROUND)
        canvas.paste(cutout.convert("RGB"), mask=cutout.getchannel("A"))
        return canvas
