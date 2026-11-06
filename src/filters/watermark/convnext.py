from ..base import Detector

REPO, FILE, REVISION = ("boomb0om/watermark-detectors", "convnext-tiny_watermarks_detector.pth",
                        "bd472165157f6d14ddcd3b24846ffd828e8fee8c")


class ConvNextWatermark(Detector):
    """boomb0om ConvNeXt-tiny watermark classifier (state dict); input resized to 256 x 256, ImageNet
    normalisation; score = softmax probability of class 1 (watermark), as in the authors' predictor."""
    task, name, model_id, batch_size = "watermark", "convnext", f"{REPO}/{FILE}@{REVISION[:7]}", 32

    def load(self, device):
        import torch
        from huggingface_hub import hf_hub_download
        from torchvision import transforms
        from .convnext_arch import ConvNeXt
        self.torch, self.device = torch, device
        weights = torch.load(hf_hub_download(REPO, FILE, revision=REVISION), map_location="cpu", weights_only=True)
        self.model = ConvNeXt()
        self.model.load_state_dict(weights, strict=True)
        self.model.to(device).float().eval()
        self.transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor(),
                                             transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    def score(self, images):
        batch = self.torch.stack([self.transform(image) for image in images]).to(self.device)
        with self.torch.inference_mode():
            return self.model(batch).float().softmax(1)[:, 1].tolist()
