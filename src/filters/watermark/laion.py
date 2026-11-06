from ..base import Detector

URL = "https://github.com/LAION-AI/LAION-5B-WatermarkDetection/raw/main/models/watermark_model_v1.pt"


class LaionWatermark(Detector):
    """LAION watermark detector, used to filter LAION-5B (Schuhmann et al., NeurIPS 2022 Datasets and Benchmarks;
    github.com/LAION-AI/LAION-5B-WatermarkDetection, MIT licence): timm efficientnet_b3a with the head
    1536 -> 625 -> 256 -> 2 (ReLU, dropout 0.3); input resized to 256 x 256, ImageNet normalisation;
    score = softmax probability of class 0 (watermark), as in the authors' example_use.py."""
    task, name, model_id, batch_size = "watermark", "laion", URL, 32
    install = ("timm>=0.9.16",)

    def load(self, device):
        import timm
        import torch
        from torch import nn
        from torchvision import transforms
        from ...paths import MODELS_DIR
        self.torch, self.device = torch, device
        path = MODELS_DIR / "laion_watermark_model_v1.pt"
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.hub.download_url_to_file(URL, str(path), progress=False)
        model = timm.create_model("efficientnet_b3a", pretrained=False, num_classes=2)
        model.classifier = nn.Sequential(nn.Linear(1536, 625), nn.ReLU(), nn.Dropout(p=0.3), nn.Linear(625, 256),
                                         nn.ReLU(), nn.Linear(256, 2))
        model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        self.model = model.to(device).eval()
        self.transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor(),
                                             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    def score(self, images):
        batch = self.torch.stack([self.transform(image) for image in images]).to(self.device)
        with self.torch.inference_mode():
            return self.model(batch).float().softmax(1)[:, 0].tolist()
