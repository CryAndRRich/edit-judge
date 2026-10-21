import torch
from torchvision import transforms
from .base import WatermarkAdapter
from .convnext_arch import ConvNeXt


class ConvnextAdapter(WatermarkAdapter):
    name = 'convnext_tiny'

    def load(self, device):
        self.device = device
        weights = torch.load(self.checkpoint, map_location='cpu', weights_only=True)
        self.model = ConvNeXt()
        self.model.load_state_dict(weights, strict=True)
        self.model.to(device).float().eval()
        self.transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor(),
            transforms.Normalize((.485, .456, .406), (.229, .224, .225))])
        self.metadata = dict(architecture='ConvNeXt tiny [3,3,9,3], custom 768/512/256/2 head',
                             preprocess='RGB, EXIF transpose, Resize(256,256) bilinear, ImageNet normalize',
                             score='softmax(logits)[1]', fp16=False, load_policy='weights_only=True, strict state_dict')

    def infer(self, image_rgb):
        return self.classifier_infer(image_rgb, lambda x: x.softmax(dim=1)[0, 1])
