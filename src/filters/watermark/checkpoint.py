"""Load pickled checkpoints (whole models) with torch.load(weights_only=True) and an explicit list of the classes
they may contain. A checkpoint that refers to any other class is refused, and nothing named in a checkpoint is
imported. This lowers the pickle risk; it is not a sandbox."""


def set_type(name):
    """The YOLO checkpoint calls dill._dill._load_type("set"); answer it without loading dill."""
    if name != "set":
        raise ValueError(f"blocked dill type request: {name!r}")
    return set


def allowed_classes(kind: str) -> list:
    from torch import nn
    common = [nn.Linear, nn.AdaptiveAvgPool2d, nn.Identity, nn.Sequential, nn.Sigmoid, nn.Conv2d, nn.Flatten,
              nn.SiLU, nn.Upsample, nn.ModuleList, nn.BCEWithLogitsLoss, nn.MaxPool2d, nn.BatchNorm2d, set]
    if kind == "efficientnet":
        from timm.layers.adaptive_avgmax_pool import SelectAdaptivePool2d
        from timm.layers.conv2d_same import Conv2dSame
        from timm.layers.drop import DropPath
        from timm.layers.norm_act import BatchNormAct2d
        from timm.models._efficientnet_blocks import ConvBnAct, EdgeResidual, InvertedResidual, SqueezeExcite
        from timm.models.efficientnet import EfficientNet
        return common + [Conv2dSame, SelectAdaptivePool2d, DropPath, BatchNormAct2d, ConvBnAct, EdgeResidual,
                         SqueezeExcite, InvertedResidual, EfficientNet]
    if kind == "yolo":
        from ultralytics.nn.modules.block import C2f, DFL, SPPF, Bottleneck
        from ultralytics.nn.modules.conv import Concat, Conv
        from ultralytics.nn.modules.head import Detect
        from ultralytics.nn.tasks import DetectionModel
        from ultralytics.utils import IterableSimpleNamespace
        from ultralytics.utils.loss import BboxLoss, v8DetectionLoss
        from ultralytics.utils.tal import TaskAlignedAssigner
        return common + [DetectionModel, SPPF, Bottleneck, C2f, DFL, Concat, Conv, Detect, IterableSimpleNamespace,
                         v8DetectionLoss, BboxLoss, TaskAlignedAssigner, (set_type, "dill._dill._load_type")]
    raise ValueError(f"unknown checkpoint kind {kind!r}")


def load_allowlisted(path, kind: str):
    import torch
    allowed = allowed_classes(kind)
    names = {item[1] if isinstance(item, tuple) else f"{item.__module__}.{item.__qualname__}" for item in allowed}
    unknown = set(torch.serialization.get_unsafe_globals_in_checkpoint(path)) - names
    if unknown:
        raise ValueError(f"checkpoint refers to classes that were not reviewed: {sorted(unknown)}")
    with torch.serialization.safe_globals(allowed):
        return torch.load(path, map_location="cpu", weights_only=True)
