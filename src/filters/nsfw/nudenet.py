from ..base import Detector

EXPOSED = {"FEMALE_BREAST_EXPOSED", "FEMALE_GENITALIA_EXPOSED", "MALE_GENITALIA_EXPOSED", "BUTTOCKS_EXPOSED",
           "ANUS_EXPOSED"}


class NudeNet(Detector):
    """NudeNet 3.4.2, detector 320n (ONNX): finds body parts; score = highest confidence of an exposed part
    (EXPOSED), 0 when none is found. A detector score, not a calibrated probability. Runs one image at a time."""
    task, name, model_id, batch_size = "nsfw", "nudenet", "nudenet==3.4.2 (320n.onnx)", 1
    # onnxruntime-gpu first, then NudeNet without its CPU onnxruntime dependency
    install = ("onnxruntime-gpu[cuda,cudnn]>=1.21,<1.24", "--no-deps nudenet==3.4.2")

    def load(self, device):
        import importlib.util
        from pathlib import Path

        import onnxruntime as ort
        from nudenet import NudeDetector
        import numpy as np
        self.np = np
        if hasattr(ort, "preload_dlls"):                 # the CUDA 12 and cuDNN libraries of the [cuda,cudnn] extra
            ort.preload_dlls()
        weights = Path(importlib.util.find_spec("nudenet").origin).parent / "320n.onnx"
        providers = (["CUDAExecutionProvider"] if device == "cuda" else []) + ["CPUExecutionProvider"]
        self.detector = NudeDetector.__new__(NudeDetector)        # the constructor would open a CPU session
        self.detector.onnx_session = ort.InferenceSession(str(weights), providers=providers)
        self.detector.input_width = self.detector.input_height = 320
        self.detector.input_name = self.detector.onnx_session.get_inputs()[0].name
        if device == "cuda" and "CUDAExecutionProvider" not in self.detector.onnx_session.get_providers():
            raise RuntimeError("NudeNet fell back to the CPU: check onnxruntime-gpu")

    def score(self, images):
        out = []
        for image in images:
            bgr = self.np.ascontiguousarray(self.np.asarray(image)[:, :, ::-1])   # NudeNet reads OpenCV (BGR) arrays
            found = self.detector.detect(bgr)
            exposed = [float(detection["score"]) for detection in found if detection["class"] in EXPOSED]
            out.append(max(exposed, default=0.0))
        return out
