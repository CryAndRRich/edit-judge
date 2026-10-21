import math
import time

import torch
from PIL import ImageOps


def validate_prediction(prediction, threshold=.5):
    score = prediction['score']
    if not math.isfinite(score) or not 0 <= score <= 1:
        raise ValueError(f'Invalid probability: {score}')
    if prediction['label'] != int(score >= threshold):
        raise ValueError('Label/threshold mismatch')
    if not isinstance(prediction['boxes'], list):
        raise ValueError('Boxes must be list')
    if not math.isfinite(prediction['timing_ms']) or prediction['timing_ms'] <= 0:
        raise ValueError('Invalid latency')
    return prediction


class WatermarkAdapter:
    name = 'base'

    def __init__(self, checkpoint, threshold=.5):
        self.checkpoint = checkpoint
        self.threshold = threshold
        self.metadata = {}

    def sync(self):
        if self.device.startswith('cuda'):
            torch.cuda.synchronize(self.device)

    def load(self, device):
        raise NotImplementedError

    def infer(self, image_rgb):
        raise NotImplementedError

    @torch.inference_mode()
    def predict(self, image_rgb):
        self.sync()
        start = time.perf_counter()
        rgb = ImageOps.exif_transpose(image_rgb).convert('RGB')
        score, boxes, model_ms = self.infer(rgb)
        self.sync()
        result = dict(score=float(score), label=int(score >= self.threshold), boxes=boxes,
                      timing_ms=(time.perf_counter()-start)*1000, model_only_ms=float(model_ms))
        return validate_prediction(result, self.threshold)

    def classifier_infer(self, image, score_function):
        x = self.transform(image).unsqueeze(0).to(self.device)
        self.sync()
        start = time.perf_counter()
        output = self.model(x)
        self.sync()
        model_ms = (time.perf_counter()-start)*1000
        return score_function(output).item(), [], model_ms
