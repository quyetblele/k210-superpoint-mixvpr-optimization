#!/usr/bin/env python3
"""Raw-head SuperPoint wrapper for a later KPU-oriented export.

This module deliberately excludes softmax, NMS, thresholding, border removal,
top-k selection, descriptor normalization, and descriptor sampling. It does not
export ONNX by itself.
"""

import sys
from pathlib import Path
from typing import Tuple

import torch
from torch import nn


EDGE_AI_ROOT = Path("/home/quyet/edge_ai_project")
HLOC_ROOT = EDGE_AI_ROOT / "third_party" / "hloc"
THIRD_PARTY_ROOT = EDGE_AI_ROOT / "third_party"
CHECKPOINT_PATH = (
    THIRD_PARTY_ROOT
    / "SuperGluePretrainedNetwork"
    / "models"
    / "weights"
    / "superpoint_v1.pth"
)

# Effective configuration of src/run_active_pipeline.py ->
# rebuild_map_v2_pipeline.py -> build_map_v2.py -> superpoint_aachen.
BASELINE_CONFIG = {
    "name": "superpoint",
    "nms_radius": 3,
    "keypoint_threshold": 0.005,
    "max_keypoints": 3072,
    "remove_borders": 4,
    "fix_sampling": False,
}


def load_local_baseline() -> nn.Module:
    """Load the exact HLOC/MagicLeap baseline and its vendored checkpoint."""
    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError("SuperPoint checkpoint not found: {}".format(CHECKPOINT_PATH))

    for path in (THIRD_PARTY_ROOT, HLOC_ROOT):
        resolved = str(path)
        if resolved not in sys.path:
            sys.path.insert(0, resolved)

    from hloc import extractors
    from hloc.utils.base_model import dynamic_load

    model_class = dynamic_load(extractors, "superpoint")
    return model_class(dict(BASELINE_CONFIG)).eval().cpu()


class SuperPointKPUWrapper(nn.Module):
    """Return convPb logits and convDb descriptors before post-processing."""

    def __init__(self, baseline: nn.Module):
        super().__init__()
        # This is the already-loaded MagicLeap network, so parameters are the
        # exact baseline weights rather than a copied or translated state dict.
        self.net = baseline.net

    def forward(self, image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if image.ndim != 4 or image.shape[1] != 1:
            raise ValueError("Expected FP32 grayscale input [N, 1, H, W]")
        if image.dtype != torch.float32:
            raise ValueError("Expected FP32 input, got {}".format(image.dtype))

        net = self.net
        x = net.relu(net.conv1a(image))
        x = net.relu(net.conv1b(x))
        x = net.pool(x)
        x = net.relu(net.conv2a(x))
        x = net.relu(net.conv2b(x))
        x = net.pool(x)
        x = net.relu(net.conv3a(x))
        x = net.relu(net.conv3b(x))
        x = net.pool(x)
        x = net.relu(net.conv4a(x))
        x = net.relu(net.conv4b(x))

        detector_logits = net.convPb(net.relu(net.convPa(x)))
        descriptor_map = net.convDb(net.relu(net.convDa(x)))
        return detector_logits, descriptor_map
