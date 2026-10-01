"""Export exactly the 110/121-token SQ240 anchor-family probes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
import torch.nn.functional as F

from sq240_resource_probe_export import (
    MODEL_DIR, REPORT_DIR, Mixer, SQ240Probe, canonicalize,
    conv_linear_macs, sha256,
)

SPECS = {
    "g11x10_d4_h112_c160_crop": (11, 10, 2, 2, 112),
    "g11x11_d4_h112_c160_crop": (11, 11, 2, 2, 112),
    "g11x10_d4_h96_c160_crop": (11, 10, 2, 2, 96),
    "g11x11_d4_h96_c160_crop": (11, 11, 2, 2, 96),
}


class StaticCrop(nn.Module):
    def __init__(self, height: int, width: int, top: int, left: int):
        super().__init__(); self.height = height; self.width = width
        self.top = top; self.left = left

    def forward(self, x):
        return x[:, :, self.top:self.top + self.height, self.left:self.left + self.width]


class SpatialCropProbe(nn.Module):
    def __init__(self, height: int, width: int, top: int, left: int, hidden: int):
        super().__init__()
        # This is the exact already-proven custom C160 SQ240 backbone object.
        self.backbone = SQ240Probe(10, 4, 96, 160).backbone
        self.adapter = StaticCrop(height, width, top, left)
        self.tokens = height * width
        self.mixers = nn.Sequential(*[Mixer(self.tokens, hidden) for _ in range(4)])
        self.channel_projection = nn.Linear(160, 128)
        self.row_projection = nn.Linear(self.tokens, 4)

    def raw(self, x):
        x = self.adapter(self.backbone(x)).flatten(2)
        x = self.mixers(x)
        x = self.channel_projection(x.permute(0, 2, 1)).permute(0, 2, 1)
        return self.row_projection(x).flatten(1)

    def forward(self, x):
        return F.normalize(self.raw(x), p=2, dim=1, eps=1e-12)


class PreL2(nn.Module):
    def __init__(self, model): super().__init__(); self.model = model
    def forward(self, x): return self.model.raw(x)


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--name", choices=tuple(SPECS), required=True)
    args = parser.parse_args(); height, width, top, left, hidden = SPECS[args.name]
    MODEL_DIR.mkdir(parents=True, exist_ok=True); REPORT_DIR.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(20260907); model = SpatialCropProbe(height, width, top, left, hidden)
    model.train(); optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
    sanity_input = torch.linspace(-1, 1, 2 * 3 * 240 * 240).reshape(2, 3, 240, 240)
    before = model.backbone[0].weight.detach().clone(); output = model(sanity_input)
    weights = torch.linspace(-1, 1, output.numel()).reshape_as(output)
    loss = (output * weights).mean(); optimizer.zero_grad(); loss.backward()
    gradients_finite = all(torch.isfinite(parameter.grad).all() for parameter in model.parameters()
                           if parameter.grad is not None)
    optimizer.step(); update = float((model.backbone[0].weight.detach() - before).abs().max())

    model.eval(); sample = torch.linspace(-1, 1, 3 * 240 * 240).reshape(1, 3, 240, 240)
    with torch.no_grad():
        backbone = model.backbone(sample); adapted = model.adapter(backbone)
    clean = MODEL_DIR / f"{args.name}_inferred.onnx"; canonical = MODEL_DIR / f"{args.name}_canonical.onnx"
    torch.onnx.export(PreL2(model).eval(), sample, str(clean), opset_version=13, dynamo=False,
                      input_names=["normalized_rgb"], output_names=["raw_descriptor"])
    model.eval(); inferred = onnx.shape_inference.infer_shapes(onnx.load(clean), strict_mode=True)
    onnx.checker.check_model(inferred); onnx.save(inferred, clean)
    rewrite = canonicalize(clean, canonical)
    clean_session = ort.InferenceSession(str(clean), providers=["CPUExecutionProvider"])
    canonical_session = ort.InferenceSession(str(canonical), providers=["CPUExecutionProvider"])
    tests = {"linspace": sample.numpy(), "random": np.random.default_rng(20260907).uniform(
        -1, 1, (1, 3, 240, 240)).astype(np.float32), "zeros": np.zeros((1, 3, 240, 240), np.float32)}
    parity = {}
    for test_name, value in tests.items():
        clean_raw = clean_session.run(None, {"normalized_rgb": value})[0]
        canonical_raw = canonical_session.run(None, {"normalized_rgb": value})[0]
        with torch.no_grad(): full = model(torch.from_numpy(value)).numpy()
        cpu_l2 = canonical_raw / np.maximum(np.linalg.norm(canonical_raw, axis=1, keepdims=True), 1e-12)
        parity[test_name] = {"clean_canonical_max_abs": float(np.max(np.abs(clean_raw - canonical_raw))),
            "partition_max_abs": float(np.max(np.abs(full - cpu_l2))),
            "cosine": float(np.dot(full.ravel(), cpu_l2.ravel()) /
                            (np.linalg.norm(full) * np.linalg.norm(cpu_l2))),
            "shape": list(canonical_raw.shape), "finite": bool(np.isfinite(canonical_raw).all())}
    report = {"name": args.name, "input": [1, 3, 240, 240],
        "backbone_output": list(backbone.shape), "adapter": {"op": "static center Slice",
        "top": top, "left": left, "height": height, "width": width},
        "adapted_output": list(adapted.shape), "tokens": model.tokens, "depth": 4,
        "hidden": hidden, "late_channels": 160, "descriptor": 512,
        "params": sum(parameter.numel() for parameter in model.parameters()),
        "conv_linear_macs": conv_linear_macs(model, sample),
        "sanity": {"output": list(output.shape), "loss_finite": bool(torch.isfinite(loss)),
                   "gradients_finite": bool(gradients_finite), "update_max_abs": update},
        "clean_onnx": str(clean), "canonical_onnx": str(canonical),
        "canonical_sha256": sha256(canonical), "rewrite": rewrite, "parity": parity}
    (REPORT_DIR / f"{args.name}_export.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__": main()
