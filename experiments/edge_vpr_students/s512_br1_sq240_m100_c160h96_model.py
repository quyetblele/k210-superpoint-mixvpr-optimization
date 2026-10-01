"""Frozen final success-first SQ240 student: 100 tokens, C160/H96."""
from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F

IDENTITY = "S512-BR1-SQ240-M100-C160H96"
INPUT_WH = (240, 240)
WIDTHS = (16, 24, 64, 112, 160)
GRID = (10, 10)
TOKENS = 100
MIXER_DEPTH = 4
MIXER_HIDDEN = 96
DESCRIPTOR_DIM = 512


class FeatureMixer(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(TOKENS)
        self.fc1 = nn.Linear(TOKENS, MIXER_HIDDEN)
        self.fc2 = nn.Linear(MIXER_HIDDEN, TOKENS)

    def forward(self, x):
        return x + self.fc2(F.relu(self.fc1(self.norm(x))))


class S512BR1SQ240M100C160H96(nn.Module):
    def __init__(self):
        super().__init__()
        layers = []
        in_channels = 3
        for stage, out_channels in enumerate(WIDTHS):
            layers.extend([
                nn.Conv2d(in_channels, out_channels, 3, 2 if stage == 0 else 1, 1),
                nn.ReLU(),
                nn.Conv2d(out_channels, out_channels, 3, 1, 1),
                nn.ReLU(),
            ])
            if stage < 3:
                layers.append(nn.MaxPool2d(2, 2))
            in_channels = out_channels
        self.backbone = nn.Sequential(*layers)
        self.static_grid_adapter = nn.MaxPool2d(6, 1)
        self.mixers = nn.Sequential(*[FeatureMixer() for _ in range(MIXER_DEPTH)])
        self.channel_projection = nn.Linear(WIDTHS[-1], 128)
        self.row_projection = nn.Linear(TOKENS, 4)

    def raw(self, x):
        x = self.static_grid_adapter(self.backbone(x)).flatten(2)
        x = self.mixers(x)
        x = self.channel_projection(x.permute(0, 2, 1)).permute(0, 2, 1)
        return self.row_projection(x).flatten(1)

    def forward(self, x):
        return F.normalize(self.raw(x), p=2, dim=1, eps=1e-12)
