from ..base import Detector

REPO, FILE, REVISION = "qfisch/yolov8n-watermark-detection", "best.pt", "26ce92a7e777ae955247e5c4a74ee3ea5279924a"


class YoloWatermark(Detector):
    """qfisch YOLOv8n fine-tuned to find watermark boxes (classes "logo" and "text"); image size 640, confidence
    floor 0.001; score = highest box confidence, 0 when there is no box. Licence AGPL-3.0, academic use only."""
    task, name, model_id, batch_size = "watermark", "yolo", f"{REPO}/{FILE}@{REVISION[:7]}", 16
    install = ("ultralytics==8.3.228",)

    def load(self, device):
        from pathlib import Path
        from unittest.mock import patch

        from huggingface_hub import hf_hub_download
        from ultralytics import YOLO
        from .checkpoint import load_allowlisted
        self.device = device
        path = hf_hub_download(REPO, FILE, revision=REVISION)
        checkpoint = load_allowlisted(path, "yolo")

        def safe_load(weight, *args, **kwargs):            # instead of the library's weights_only=False loader
            if Path(weight).resolve() != Path(path).resolve():
                raise ValueError("unexpected checkpoint path")
            return checkpoint, str(path)

        with patch("ultralytics.nn.tasks.torch_safe_load", safe_load):
            self.model = YOLO(str(path), task="detect")
        self.model.model.to(device).float().eval()
        if self.model.names != {0: "logo", 1: "text"}:
            raise ValueError(f"unexpected classes {self.model.names}")

    def score(self, images):
        results = self.model.predict(source=images, device=self.device, imgsz=640, conf=0.001, classes=[0, 1],
                                     verbose=False, half=False, batch=len(images))
        return [float(result.boxes.conf.max()) if result.boxes is not None and len(result.boxes) else 0.0
                for result in results]
