"""ObjectClear (Zhao et al., CVPR 2026; S-Lab licence, non-commercial): SDXL inpainting with an object encoder and
attention-guided fusion; it also removes the object's shadows and reflections.

The repository has no package to install: its code is downloaded at COMMIT into EJ_MODELS_DIR and put on the path.
Its requirements are unpinned and already on Kaggle (diffusers, transformers, opencv), so it runs in the main
environment. Settings of inference_objectclear.py: the fp16 weights (`variant="fp16"`), shorter side 512 with the
repository's `resize_by_short_side`, its fixed prompt, 20 steps, guidance 2.5 (the default; the paper's tables use
1.0), attention-guided fusion on; the output is resized back by the runner.
"""

import sys

from ..base import Editor

REPO = "zjx0101/ObjectClear"
COMMIT = "c052d91ecd8772744a5ab97527441c813ab83009"
MODEL = "jixin0101/ObjectClear"
REVISION = "c73af80888dbd519819d0a521b5c8d0f3cda6859"
PROMPT = "remove the instance of object"
SHORT_SIDE = 512


def repository_code():
    """The repository at COMMIT, unpacked once into EJ_MODELS_DIR/ObjectClear-<commit>; returns its folder."""
    import tarfile
    import urllib.request

    from ...paths import MODELS_DIR
    folder = MODELS_DIR / f"ObjectClear-{COMMIT}"
    if not (folder / "objectclear").exists():
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        archive = MODELS_DIR / f"ObjectClear-{COMMIT[:7]}.tar.gz"
        urllib.request.urlretrieve(f"https://github.com/{REPO}/archive/{COMMIT}.tar.gz", archive)
        with tarfile.open(archive) as tar:
            try:
                tar.extractall(MODELS_DIR, filter="data")
            except TypeError:                         # Python before 3.11.4: no extraction filters
                tar.extractall(MODELS_DIR)
        archive.unlink()
    return folder


def weights(fp16: bool) -> str:
    """The weights folder at REVISION. The loader downloads the repository itself but takes no revision: it is
    pinned here, with the loader's own filter (only the weights of one precision), and the local folder handed
    to it."""
    from huggingface_hub import snapshot_download
    return snapshot_download(MODEL, revision=REVISION, ignore_patterns=(
        ["*model.safetensors", "*model.bin"] if fp16 else ["*.fp16.safetensors", "*.fp16.bin"]))


class ObjectClear(Editor):
    name = "objectclear"
    model_id = f"{MODEL}@{REVISION[:7]} (code {COMMIT[:7]})"
    tasks = ("remove",)
    settings = {"short_side": SHORT_SIDE, "steps": 20, "guidance": 2.5, "agf": True, "prompt": PROMPT}
    repos = (MODEL,)
    download_gb = 7.8

    def fetch(self) -> None:
        repository_code()
        weights(fp16=True)                            # the GPU loads the fp16 files

    def load(self, device: str) -> None:
        import torch

        sys.path.insert(0, str(repository_code()))
        from objectclear.pipelines import ObjectClearPipeline
        from objectclear.utils import resize_by_short_side

        self.torch, self.device, self.resize = torch, device, resize_by_short_side
        fp16 = device == "cuda"
        self.pipe = ObjectClearPipeline.from_pretrained_with_custom_modules(
            weights(fp16), torch_dtype=torch.float16 if fp16 else torch.float32, apply_attention_guided_fusion=True,
            variant="fp16" if fp16 else None)
        self.pipe.to(device)
        self.pipe.set_progress_bar_config(disable=True)

    def edit(self, image, mask, prompt: str, seed: int = 0):
        from PIL import Image

        image = self.resize(image, SHORT_SIDE, resample=Image.BICUBIC)
        mask = self.resize(mask, SHORT_SIDE, resample=Image.NEAREST)
        generator = self.torch.Generator(device=self.device).manual_seed(seed)
        return self.pipe(prompt=PROMPT, image=image, mask_image=mask, generator=generator, num_inference_steps=20,
                         guidance_scale=2.5, height=image.height, width=image.width,
                         return_attn_map=False).images[0]

