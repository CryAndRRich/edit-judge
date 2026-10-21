"""Fail-closed weights-only deserialization with a small explicit reviewed allowlist.

No import name from a checkpoint is dynamically imported. Unknown globals block load.
This reduces pickle risk; it is not a security sandbox or proof of benign tensors.
"""
import torch
from torch import nn


def restricted_load_type(name):
    # Checkpoint pickletools inspection found dill._load_type('set'). Do not load dill.
    if name != 'set':
        raise ValueError(f'Blocked dill type request: {name!r}')
    return set


def load_allowlisted(path, kind):
    common = [nn.Linear, nn.AdaptiveAvgPool2d, nn.Identity, nn.Sequential, nn.Sigmoid,
              nn.Conv2d, nn.Flatten, nn.SiLU, nn.Upsample, nn.ModuleList,
              nn.BCEWithLogitsLoss, nn.MaxPool2d, nn.BatchNorm2d, set]
    if kind == 'efficientnetv2_b0':
        from timm.layers.conv2d_same import Conv2dSame
        from timm.layers.adaptive_avgmax_pool import SelectAdaptivePool2d
        from timm.layers.drop import DropPath
        from timm.layers.norm_act import BatchNormAct2d
        from timm.models._efficientnet_blocks import ConvBnAct, EdgeResidual, SqueezeExcite, InvertedResidual
        from timm.models.efficientnet import EfficientNet
        allowed = common + [Conv2dSame, SelectAdaptivePool2d, DropPath, BatchNormAct2d,
                            ConvBnAct, EdgeResidual, SqueezeExcite, InvertedResidual, EfficientNet]
    elif kind == 'yolov8n':
        from ultralytics.nn.tasks import DetectionModel
        from ultralytics.nn.modules.block import SPPF, Bottleneck, C2f, DFL
        from ultralytics.nn.modules.conv import Concat, Conv
        from ultralytics.nn.modules.head import Detect
        from ultralytics.utils import IterableSimpleNamespace
        from ultralytics.utils.loss import v8DetectionLoss, BboxLoss
        from ultralytics.utils.tal import TaskAlignedAssigner
        allowed = common + [DetectionModel, SPPF, Bottleneck, C2f, DFL, Concat, Conv,
                            Detect, IterableSimpleNamespace, v8DetectionLoss, BboxLoss, TaskAlignedAssigner,
                            (restricted_load_type, 'dill._dill._load_type')]
    else:
        raise ValueError('Unknown checkpoint kind')
    names = {f'{x.__module__}.{x.__qualname__}' if not isinstance(x, tuple) else x[1] for x in allowed}
    globals_found = set(torch.serialization.get_unsafe_globals_in_checkpoint(path))
    unknown = globals_found - names
    if unknown:
        raise ValueError(f'Unreviewed checkpoint globals: {sorted(unknown)}')
    with torch.serialization.safe_globals(allowed):
        result = torch.load(path, map_location='cpu', weights_only=True)
    return result, sorted(globals_found)
