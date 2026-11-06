from ..base import Detector

REPO, FILE, REVISION = "qwertyforce/watermark_detection", "model.pt", "6114db333950e5d26376961cd9a7a463ea602215"


class EfficientNetWatermark(Detector):
    """qwertyforce EfficientNetV2-B0 watermark classifier (a whole pickled timm model with one output logit);
    preprocessing from the timm config stored in the checkpoint (192 px centre crop); score = sigmoid(logit). The
    model card does not state the fine-tuning transform, so the centre crop may cut marks near the edges."""
    task, name, model_id, batch_size = "watermark", "efficientnet", f"{REPO}/{FILE}@{REVISION[:7]}", 32
    install = ("timm>=0.9.16",)

    def load(self, device):
        import torch
        from huggingface_hub import hf_hub_download
        from timm.data import create_transform, resolve_data_config
        from .checkpoint import load_allowlisted
        self.torch, self.device = torch, device
        model = load_allowlisted(hf_hub_download(REPO, FILE, revision=REVISION), "efficientnet")
        head = model.get_classifier()
        if not isinstance(head, torch.nn.Linear) or head.out_features != 1:
            raise ValueError(f"unexpected classifier {head}")
        self.model = model.to(device).float().eval()
        cfg = getattr(model, "pretrained_cfg", None) or getattr(model, "default_cfg", {})
        self.transform = create_transform(**resolve_data_config(cfg, model=model), is_training=False)

    def score(self, images):
        x = self.torch.stack([self.transform(im) for im in images]).to(self.device)
        with self.torch.inference_mode():
            return self.model(x).float().reshape(-1).sigmoid().tolist()
