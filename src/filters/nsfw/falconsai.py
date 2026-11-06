from ..base import Detector


class Falconsai(Detector):
    """Falconsai/nsfw_image_detection: a ViT image classifier with labels "normal" and "nsfw"; score =
    softmax probability of "nsfw"."""
    task, name, model_id, batch_size = "nsfw", "falconsai", "Falconsai/nsfw_image_detection", 32

    def load(self, device):
        import torch
        from transformers import AutoImageProcessor, AutoModelForImageClassification
        self.torch, self.device = torch, device
        self.processor = AutoImageProcessor.from_pretrained(self.model_id)
        self.model = AutoModelForImageClassification.from_pretrained(self.model_id, use_safetensors=True)
        self.model.to(device).eval()
        labels = {str(v).lower(): int(k) for k, v in self.model.config.id2label.items()}
        self.positive = labels["nsfw"]

    def score(self, images):
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        with self.torch.inference_mode():
            probs = self.model(**inputs).logits.float().softmax(-1)
        return probs[:, self.positive].tolist()
