# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Original source license notice: "This source code is licensed under the license
# found in the LICENSE file in the root directory of this source tree."
"""ConvNeXt architecture from boomb0om/watermark-detection at 1b770cf.

Adapted locally from wmdetection/models/convnext.py (Meta Platforms copyright).
Only inference architecture retained; pretrained URL/download paths removed.
See reports/latest/sources for the original reviewed source.
"""
import torch
from torch import nn
from torch.nn import functional as F


class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-6, data_format='channels_last'):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps, self.data_format, self.normalized_shape = eps, data_format, (dim,)

    def forward(self, x):
        if self.data_format == 'channels_last':
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        u = x.mean(1, keepdim=True)
        s = (x-u).pow(2).mean(1, keepdim=True)
        return self.weight[:, None, None] * ((x-u)/torch.sqrt(s+self.eps)) + self.bias[:, None, None]


class Block(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm(dim)
        self.pwconv1 = nn.Linear(dim, 4*dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4*dim, dim)
        self.gamma = nn.Parameter(1e-6*torch.ones(dim))

    def forward(self, x):
        residual = x
        x = self.norm(self.dwconv(x).permute(0, 2, 3, 1))
        x = self.gamma*self.pwconv2(self.act(self.pwconv1(x)))
        return residual + x.permute(0, 3, 1, 2)


class ConvNeXt(nn.Module):
    def __init__(self):
        super().__init__()
        dims = [96, 192, 384, 768]
        self.downsample_layers = nn.ModuleList([
            nn.Sequential(nn.Conv2d(3, 96, kernel_size=4, stride=4), LayerNorm(96, data_format='channels_first')),
            *[nn.Sequential(LayerNorm(dims[i], data_format='channels_first'),
                            nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2)) for i in range(3)]
        ])
        self.stages = nn.ModuleList([nn.Sequential(*[Block(d) for _ in range(n)]) for d, n in zip(dims, [3, 3, 9, 3])])
        self.norm = nn.LayerNorm(768, eps=1e-6)
        self.head = nn.Sequential(nn.Linear(768, 512), nn.GELU(), nn.Linear(512, 256), nn.GELU(), nn.Linear(256, 2))

    def forward(self, x):
        for down, stage in zip(self.downsample_layers, self.stages):
            x = stage(down(x))
        return self.head(self.norm(x.mean([-2, -1])))
