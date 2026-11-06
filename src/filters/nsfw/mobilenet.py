import zipfile

from ...paths import MODELS_DIR
from ..base import Detector
from .opennsfw2 import tf_memory_growth

URL = "https://github.com/GantMan/nsfw_model/releases/download/1.2.0/mobilenet_v2_140_224.1.zip"
LABELS = ["drawings", "hentai", "neutral", "porn", "sexy"]
POSITIVE = [1, 3, 4]                                    # hentai, porn, sexy


class MobileNetNSFW(Detector):
    """GantMan nsfw_model 1.2.0, MobileNetV2 1.4 / 224, TensorFlow SavedModel with 5 classes; input RGB resized to
    224 x 224 (nearest) and divided by 255; score = P(hentai) + P(porn) + P(sexy)."""
    task, name, model_id, batch_size = "nsfw", "mobilenet", URL, 32
    install = ("tensorflow>=2.20,<2.22 opennsfw2[keras3]==0.19.0",)   # the same environment as opennsfw2

    def load(self, device):
        import numpy as np
        import requests
        tf = tf_memory_growth()
        self.np, self.tf = np, tf
        folder = MODELS_DIR / "mobilenet_v2_140_224"
        if not list(folder.rglob("saved_model.pb")):
            folder.mkdir(parents=True, exist_ok=True)
            archive = folder / "model.zip"
            with requests.get(URL, stream=True, timeout=(30, 120)) as r:
                r.raise_for_status()
                with open(archive, "wb") as fh:
                    for chunk in r.iter_content(1 << 20):
                        fh.write(chunk)
            with zipfile.ZipFile(archive) as zf:
                if any(not (folder / m).resolve().is_relative_to(folder.resolve()) for m in zf.namelist()):
                    raise ValueError("unsafe path in the model zip")
                zf.extractall(folder)
            archive.unlink()
        model_dir = next(folder.rglob("saved_model.pb")).parent
        if (model_dir / "class_labels.txt").read_text().split() != LABELS:
            raise ValueError("unexpected class order in the checkpoint")
        self.tf_device = "/GPU:0" if device == "cuda" else "/CPU:0"
        with tf.device(self.tf_device):
            self.model = tf.saved_model.load(str(model_dir))
        self.fn = self.model.signatures["serving_default"]
        self.input_name = next(iter(self.fn.structured_input_signature[1]))

    def score(self, images):
        x = self.np.stack([self.np.asarray(im.resize((224, 224), resample=0), dtype=self.np.float32) / 255.0
                           for im in images])
        with self.tf.device(self.tf_device):
            probs = next(iter(self.fn(**{self.input_name: self.tf.convert_to_tensor(x)}).values())).numpy()
        return self.np.clip(probs[:, POSITIVE].sum(axis=1), 0, 1).tolist()
