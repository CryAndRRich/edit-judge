from ..base import Detector


def tf_memory_growth():
    """Let TensorFlow share the GPU with PyTorch instead of taking all of its memory."""
    import tensorflow as tf
    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)
    return tf


class OpenNSFW2(Detector):
    """Yahoo Open-NSFW weights through opennsfw2 (TensorFlow/Keras), Yahoo preprocessing; score = probability of
    the NSFW class."""
    task, name, model_id, batch_size = "nsfw", "opennsfw2", "opennsfw2==0.19.0 (Yahoo Open-NSFW)", 32
    install = ("tensorflow>=2.20,<2.22 opennsfw2[keras3]==0.19.0",)

    def load(self, device):
        import numpy as np
        tf = tf_memory_growth()
        import opennsfw2 as n2
        self.np, self.tf, self.n2 = np, tf, n2
        self.tf_device = "/GPU:0" if device == "cuda" else "/CPU:0"
        with tf.device(self.tf_device):
            self.model = n2.make_open_nsfw_model()          # downloads the weights to ~/.opennsfw2 once

    def score(self, images):
        x = self.np.stack([self.n2.preprocess_image(im, self.n2.Preprocessing.YAHOO) for im in images])
        with self.tf.device(self.tf_device):
            probs = self.model(x, training=False).numpy()
        return probs[:, 1].tolist()
