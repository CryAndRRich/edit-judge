import torch
from torchvision import transforms
from .base import WatermarkAdapter
from .safe_checkpoint import load_allowlisted


class EfficientnetAdapter(WatermarkAdapter):
    name = 'efficientnetv2_b0'

    def load(self, device):
        self.device = device
        model, globals_found = load_allowlisted(self.checkpoint, self.name)
        if not isinstance(model, torch.nn.Module):
            raise ValueError('Expected EfficientNet module checkpoint')
        self.model = model.to(device).float().eval()
        cfg = getattr(model, 'pretrained_cfg', None) or getattr(model, 'default_cfg', {})
        classifier = model.get_classifier()
        if not isinstance(classifier, torch.nn.Linear) or classifier.out_features != 1:
            raise ValueError(f'blocked_mapping: unexpected classifier {classifier}')
        if any(isinstance(x, torch.nn.Sigmoid) for x in classifier.modules()):
            raise ValueError('Unexpected sigmoid head: explicit review required')
        # Use checkpoint-embedded timm evaluation config, never select transform on test accuracy.
        from timm.data import resolve_data_config, create_transform
        config = resolve_data_config(cfg, model=model)
        self.transform = create_transform(**config, is_training=False)
        self.metadata = dict(architecture=type(model).__module__+'.'+type(model).__name__,
                             classifier=str(classifier), output='one raw logit; apply sigmoid exactly once',
                             score='sigmoid(logit); watermark=1 per model card',
                             preprocess=str(self.transform), data_config=config, embedded_config=cfg,
                             globals=globals_found, load_policy='weights_only=True with explicit audited class allowlist',
                             mapping_note='Author card specifies watermark probability via sigmoid; examples logged without choosing orientation or threshold')

    def infer(self, image_rgb):
        return self.classifier_infer(image_rgb, lambda x: x.reshape(-1)[0].sigmoid())
