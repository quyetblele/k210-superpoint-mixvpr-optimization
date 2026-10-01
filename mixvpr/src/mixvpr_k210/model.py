from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelSpec

class ConvStage(nn.Module):
    def __init__(self, in_channels: int, out_channels: int,
                 first_stride: int, pool: bool):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, 3, first_stride, 1
        )
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, 3, 1, 1
        )
        self.pool = nn.MaxPool2d(2, 2) if pool else nn.Identity()

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return self.pool(x)

class FeatureMixer(nn.Module):
    def __init__(self, tokens: int, hidden: int):
        super().__init__()
        self.norm = nn.LayerNorm(tokens)
        self.fc1 = nn.Linear(tokens, hidden)
        self.fc2 = nn.Linear(hidden, tokens)

    def forward(self, x):
        return x + self.fc2(F.relu(self.fc1(self.norm(x))))

class K210MixVPR(nn.Module):
    """Single configurable 512-D student family for progressive pruning."""

    def __init__(self, spec: ModelSpec):
        super().__init__()
        self.spec = spec
        stages = []
        in_channels = 3
        for i, out_channels in enumerate(spec.widths):
            stages.append(ConvStage(
                in_channels=in_channels,
                out_channels=out_channels,
                first_stride=2 if i == 0 else 1,
                pool=i < 3,
            ))
            in_channels = out_channels
        self.stages = nn.ModuleList(stages)
        self.static_grid_adapter = nn.MaxPool2d(6, 1)
        self.mixers = nn.ModuleList([
            FeatureMixer(spec.tokens, spec.mixer_hidden)
            for _ in range(spec.mixer_depth)
        ])
        self.channel_projection = nn.Linear(
            spec.widths[-1], spec.projection
        )
        self.row_projection = nn.Linear(
            spec.tokens, spec.rows
        )

    def raw(self, x):
        for stage in self.stages:
            x = stage(x)
        x = self.static_grid_adapter(x).flatten(2)
        if not torch.jit.is_tracing() and x.shape[-1] != self.spec.tokens:
            raise RuntimeError(
                f"token contract broken: expected {self.spec.tokens}, "
                f"got shape {tuple(x.shape)}"
            )
        for mixer in self.mixers:
            x = mixer(x)
        x = self.channel_projection(
            x.permute(0, 2, 1)
        ).permute(0, 2, 1)
        return self.row_projection(x).flatten(1)

    def forward(self, x):
        return F.normalize(
            self.raw(x), p=2, dim=1, eps=1e-12
        )

def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())

def conv_linear_macs(model: nn.Module,
                     shape=(1, 3, 240, 240)) -> int:
    macs = [0]

    def hook(layer, _args, out):
        if isinstance(layer, nn.Conv2d):
            k = (
                layer.in_channels // layer.groups
                * layer.kernel_size[0]
                * layer.kernel_size[1]
            )
        else:
            k = layer.in_features
        macs[0] += out.numel() * k

    handles = [
        module.register_forward_hook(hook)
        for module in model.modules()
        if isinstance(module, (nn.Conv2d, nn.Linear))
    ]
    with torch.inference_mode():
        model.raw(torch.zeros(*shape))
    for handle in handles:
        handle.remove()
    return int(macs[0])
