import time
from pathlib import Path
from unittest.mock import patch

from .base import WatermarkAdapter
from .safe_checkpoint import load_allowlisted


class YoloAdapter(WatermarkAdapter):
    name = 'yolov8n'

    def load(self, device):
        from ultralytics import YOLO
        self.device = device
        checkpoint, globals_found = load_allowlisted(self.checkpoint, self.name)

        def safe_loader(weight, *args, **kwargs):
            if Path(weight).resolve() != Path(self.checkpoint).resolve():
                raise ValueError('Unexpected checkpoint path')
            return checkpoint, str(self.checkpoint)

        # Replace the library loader: prevent its weights_only=False and auto-install fallback.
        with patch('ultralytics.nn.tasks.torch_safe_load', safe_loader):
            self.model = YOLO(str(self.checkpoint), task='detect')
        self.model.model.to(device).float().eval()
        names = self.model.names
        # Model card explicitly describes watermark text AND logo; checkpoint names
        # are exactly {0: 'logo', 1: 'text'}, not generic object detector classes.
        if names == {0: 'logo', 1: 'text'}:
            self.watermark_classes = [0, 1]
        else:
            self.watermark_classes = [i for i, name in names.items() if 'watermark' in name.lower()]
        if not self.watermark_classes:
            raise ValueError(f'No explicit watermark class in {names}; review mapping')
        self.metadata = dict(classes=names, watermark_classes=self.watermark_classes, conf=.001, imgsz=640,
                             class_mapping_evidence='qfisch model card: watermark detection of both texts and logos; exact checkpoint names logo/text',
                             preprocess='PIL RGB + EXIF transpose; Ultralytics letterbox imgsz=640, scale 1/255, no ImageNet normalization',
                             score='max confidence over watermark boxes, default 0',
                             globals=globals_found, load_policy='weights_only=True with explicit allowlist and restricted dill set type shim; stock loader replaced',
                             model_only_timing='Ultralytics result.speed[inference] forward timing; distinct from adapter end-to-end')

    def infer(self, image_rgb):
        result = self.model.predict(source=image_rgb, device=self.device, imgsz=640, conf=.001,
                                    classes=self.watermark_classes, verbose=False, half=False, batch=1)[0]
        boxes = []
        if result.boxes is not None:
            for xyxy, confidence, cls in zip(result.boxes.xyxy.cpu().tolist(), result.boxes.conf.cpu().tolist(), result.boxes.cls.cpu().tolist()):
                boxes.append(dict(xyxy=xyxy, confidence=confidence, class_id=int(cls)))
        return max((b['confidence'] for b in boxes), default=0.0), boxes, result.speed['inference']
