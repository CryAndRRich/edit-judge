from ..base import Detector


class OwenElliott(Detector):
    """OwenElliott/image-safety-classifier-m: a timm classifier with labels SFW, NSFW and NSFL; score = probability
    of NSFW only (NSFL, gore, is not added), as in the benchmark."""
    task, name, model_id, batch_size = "nsfw", "owen", "hf-hub:OwenElliott/image-safety-classifier-m", 32
    install = ("timm>=0.9.16",)

    def load(self, device):
        import timm
        import torch
        from timm.data import create_transform, resolve_model_data_config
        self.torch, self.device = torch, device
        self.model = timm.create_model(self.model_id, pretrained=True).to(device).eval()
        self.transform = create_transform(**resolve_model_data_config(self.model), is_training=False)
        self.positive = [str(x).upper() for x in self.model.pretrained_cfg["label_names"]].index("NSFW")

    def score(self, images):
        x = self.torch.stack([self.transform(im) for im in images]).to(self.device)
        with self.torch.inference_mode():
            probs = self.model(x).float().softmax(-1)
        return probs[:, self.positive].tolist()
