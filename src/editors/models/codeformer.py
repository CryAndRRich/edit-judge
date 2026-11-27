"""CodeFormer (Zhou et al., NeurIPS 2022; NTU S-Lab License 1.0, non-commercial): blind face restoration with a
codebook lookup transformer, the classic restorer of list 5. It restores faces only; the rest of the photo is kept.

The repository has no package to install: its code is downloaded at COMMIT into EJ_MODELS_DIR and put on the path.
It bundles its own `basicsr` and `facelib`. `python basicsr/setup.py develop` is not needed: without `--cuda_ext` it
compiles nothing and only writes `basicsr/version.py` (which `basicsr/__init__.py` imports) and installs the
requirements. `basicsr/__init__.py` also pulls in the training code (data, losses, models, train), so `basicsr` is
registered as an empty package pointing at the folder and only `basicsr.archs` and `basicsr.utils` are imported;
`facelib` has no `__init__.py` and is found as a namespace package. The deformable-convolution extension is only
tried at import (skipped unless BASICSR_JIT=True) and CodeFormer does not use it. The bundled code does not import
`torchvision.transforms.functional_tensor` and uses no NumPy aliases removed in NumPy 2; `basicsr/archs/arch_util.py`
imports `distutils` (still in Python 3.11's standard library). The three weight files hold only tensors, so the
`weights_only` default of torch.load from torch 2.6 loads them.

Settings of inference_codeformer.py on whole images, `-w 0.7`: RetinaFace-ResNet50 face detection on the image
scaled so its short side is 640, faces with eyes closer than 5 px skipped, each face aligned to 512 x 512 with the
FFHQ 5-point template, restored with fidelity weight w = 0.7 and AdaIN colour correction, pasted back through a
ParseNet face-parsing mask; no background upsampler (`--bg_upsampler` defaults to None) and no face upsampler, fp32
as in the script. Deviations: `--upscale 1` instead of the default 2 (`--upscale` is the factor of the final image: the
background is resized with cv2 bilinear and the faces pasted at that scale; the runner resizes back anyway); the
release files are downloaded into EJ_MODELS_DIR with a size check instead of the repository's `weights/` folder;
a failing face is an error of the sample (the script keeps the unrestored face). The helper itself scales images whose
short side is below 512 up to 512 (bilinear); the runner resizes the output back. A photo without a detected face
comes back unchanged (apart from that resizing). The pasted result is clipped to 0-255 as uint8: the helper
returns uint16 when a blended pixel goes above 256, which PIL cannot save. No prompt; deterministic.
"""

import sys

from ..base import Editor
from .fetch import repository_code
from .lama import download

REPO = "sczhou/CodeFormer"
COMMIT = "b33cc7d639d6545bfcccc7e0bc6ae51f24e79c2b"
RELEASE = f"https://github.com/{REPO}/releases/download/v0.1.0"
WEIGHTS = {"codeformer.pth": 376637898, "detection_Resnet50_Final.pth": 109497761, "parsing_parsenet.pth": 85331193}
FIDELITY = 0.7
UPSCALE = 1
DETECTOR = "retinaface_resnet50"
DETECT_SHORT_SIDE = 640
EYE_DISTANCE = 5


def weight_file(url: str, **ignored) -> str:
    """Stands in for facelib's load_file_from_url: the release file in EJ_MODELS_DIR, its size checked."""
    return str(download(url, WEIGHTS[url.rsplit("/", 1)[1]]))


def import_repository(folder) -> None:
    """Make the repository's `basicsr` (without its __init__.py) and `facelib` importable."""
    import types

    package_dir = str(folder / "basicsr")
    loaded = sys.modules.get("basicsr")
    if loaded is None:
        package = types.ModuleType("basicsr")
        package.__path__ = [package_dir]
        sys.modules["basicsr"] = package
    elif list(getattr(loaded, "__path__", [])) != [package_dir]:
        raise RuntimeError(f"another basicsr is already imported ({loaded}); run CodeFormer in its own process")
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))


class CodeFormer(Editor):
    name = "codeformer"
    local_dirs = (f"CodeFormer-{COMMIT}",)
    download_gb = 0.6
    model_id = f"{RELEASE}/codeformer.pth (code {COMMIT[:7]})"
    tasks = ("restore",)
    uses_mask = False
    settings = {"fidelity_weight": FIDELITY, "upscale": UPSCALE, "detector": DETECTOR, "parsing": "parsenet",
                "face_size": 512, "detect_short_side": DETECT_SHORT_SIDE, "eye_dist_threshold": EYE_DISTANCE,
                "adain": True, "bg_upsampler": None, "face_upsample": False, "dtype": "fp32"}

    def fetch(self) -> None:
        repository_code(REPO, COMMIT)
        for file_name in WEIGHTS:
            weight_file(f"{RELEASE}/{file_name}")

    def load(self, device: str) -> None:
        import torch

        import_repository(repository_code(REPO, COMMIT))
        from basicsr.archs.codeformer_arch import CodeFormer as CodeFormerNet
        from basicsr.utils import img2tensor, tensor2img
        import facelib.detection as detection_package
        import facelib.detection.retinaface.retinaface as retinaface_module
        import facelib.parsing as parsing_package
        from facelib.utils.face_restoration_helper import FaceRestoreHelper
        from torchvision.transforms.functional import normalize

        # the detector puts its inputs on a module-level device picked at import ("cuda", "mps" or "cpu"); use ours
        retinaface_module.device = torch.device(device)
        detection_package.load_file_from_url = weight_file
        parsing_package.load_file_from_url = weight_file
        self.torch, self.device = torch, device
        self.img2tensor, self.tensor2img, self.normalize = img2tensor, tensor2img, normalize
        self.net = CodeFormerNet(dim_embd=512, codebook_size=1024, n_head=8, n_layers=9,
                                 connect_list=["32", "64", "128", "256"]).to(device)
        checkpoint = torch.load(weight_file(f"{RELEASE}/codeformer.pth"), map_location="cpu", weights_only=True)
        self.net.load_state_dict(checkpoint["params_ema"])
        self.net.eval()
        self.helper = FaceRestoreHelper(UPSCALE, face_size=512, crop_ratio=(1, 1), det_model=DETECTOR,
                                        save_ext="png", use_parse=True, device=torch.device(device))

    def edit(self, image, mask, prompt: str, seed: int = 0):
        import numpy as np
        from PIL import Image

        helper = self.helper
        helper.clean_all()
        helper.read_image(np.ascontiguousarray(np.asarray(image)[:, :, ::-1]))     # the helper works in BGR
        helper.get_face_landmarks_5(only_center_face=False, resize=DETECT_SHORT_SIDE, eye_dist_threshold=EYE_DISTANCE)
        helper.align_warp_face()
        for cropped_face in helper.cropped_faces:
            face = self.img2tensor(cropped_face / 255.0, bgr2rgb=True, float32=True)
            self.normalize(face, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
            with self.torch.no_grad():
                output = self.net(face.unsqueeze(0).to(self.device), w=FIDELITY, adain=True)[0]
            restored = self.tensor2img(output, rgb2bgr=True, min_max=(-1, 1)).astype("uint8")
            helper.add_restored_face(restored, cropped_face)
        helper.get_inverse_affine(None)
        result = helper.paste_faces_to_input_image(upsample_img=None, draw_box=False)
        # the helper returns uint16 when a blended pixel goes above 256 (sample 6db168dba581, 2026-11-26)
        result = np.clip(result, 0, 255).astype(np.uint8)
        return Image.fromarray(np.ascontiguousarray(result[:, :, ::-1]))
