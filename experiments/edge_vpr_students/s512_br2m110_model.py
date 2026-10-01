"""Frozen BR2M-110 architecture selected by the K210 allocator sweep."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureMixer110(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(110)
        self.fc1 = nn.Linear(110, 112)
        self.fc2 = nn.Linear(112, 110)

    def forward(self, x):
        return x + self.fc2(F.relu(self.fc1(self.norm(x))))


class BR2M110(nn.Module):
    """192x352 RGB input to an L2-normalized 512-D descriptor."""

    def __init__(self):
        super().__init__()
        widths = [16, 24, 64, 112, 144]
        layers = []
        in_channels = 3
        for stage, out_channels in enumerate(widths):
            layers.extend(
                [
                    nn.Conv2d(in_channels, out_channels, 3, 2 if stage == 0 else 1, 1),
                    nn.ReLU(),
                    nn.Conv2d(out_channels, out_channels, 3, 1, 1),
                    nn.ReLU(),
                ]
            )
            if stage < 3:
                layers.append(nn.MaxPool2d(2, 2))
            in_channels = out_channels
        self.backbone = nn.Sequential(*layers)
        self.spatial_adapter = nn.MaxPool2d((2, 3), (2, 1))
        self.mixers = nn.Sequential(*[FeatureMixer110() for _ in range(4)])
        self.channel_projection = nn.Linear(144, 128)
        self.row_projection = nn.Linear(110, 4)

    def raw(self, x):
        x = self.backbone(x)
        x = self.spatial_adapter(x).flatten(2)
        x = self.mixers(x)
        x = self.channel_projection(x.permute(0, 2, 1)).permute(0, 2, 1)
        return self.row_projection(x).flatten(1)

    def forward(self, x):
        return F.normalize(self.raw(x), p=2, dim=1, eps=1e-12)


FROZEN_CONFIG = {
    "identity": "S512-BR2M-110",
    "input_nchw": [1, 3, 352, 192],
    "input_wh": [192, 352],
    "backbone_widths": [16, 24, 64, 112, 144],
    "backbone_output": [1, 144, 22, 12],
    "spatial_adapter": {"type": "MaxPool2d", "kernel": [2, 3], "stride": [2, 1]},
    "mixer_grid": [11, 10],
    "tokens": 110,
    "mixer_depth": 4,
    "mixer_hidden": 112,
    "descriptor_dim": 512,
    "parameters": 691300,
    "final_normalization": "L2, eps=1e-12; CPU-side in K210 deployment",
}
