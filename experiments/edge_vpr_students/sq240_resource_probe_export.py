"""Export static SQ240 capacity probes without training them.

The probe family fixes 240x240 input and a raw 512-D deployment output.  It
keeps the BR1 early backbone, removes only the final 2x2 pool (15x15 late
feature map), and uses a compile-time MaxPool adapter to obtain 10x10 or 9x9.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper
import onnxruntime as ort
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path("/home/quyet/k210_lab")
MODEL_DIR = ROOT / "models" / "sq240_resource_probes"
REPORT_DIR = ROOT / "reports" / "edge_vpr_students" / "sq240_resource_probes"


class Mixer(nn.Module):
    def __init__(self, tokens: int, hidden: int):
        super().__init__()
        self.norm = nn.LayerNorm(tokens)
        self.fc1 = nn.Linear(tokens, hidden)
        self.fc2 = nn.Linear(hidden, tokens)

    def forward(self, x):
        return x + self.fc2(F.relu(self.fc1(self.norm(x))))


class SQ240Probe(nn.Module):
    def __init__(self, grid: int, depth: int, hidden: int, late_channels: int):
        super().__init__()
        if grid not in (9, 10, 12):
            raise ValueError("grid must be 9, 10, or 12")
        widths = [16, 24, 64, 112, late_channels]
        layers = []
        in_channels = 3
        for stage, out_channels in enumerate(widths):
            layers.extend([
                nn.Conv2d(in_channels, out_channels, 3, 2 if stage == 0 else 1, 1),
                nn.ReLU(),
                nn.Conv2d(out_channels, out_channels, 3, 1, 1),
                nn.ReLU(),
            ])
            # Total stride 16, matching the already compiler-proven BR2M
            # late-grid topology while preserving all BR1 convolutions.
            if stage < 3:
                layers.append(nn.MaxPool2d(2, 2))
            in_channels = out_channels
        self.backbone = nn.Sequential(*layers)
        # 15x15 -> grid x grid, entirely static. Kernel 4/6/7 gives
        # 12/10/9 respectively at stride 1.
        self.adapter = nn.MaxPool2d(16 - grid, 1)
        self.tokens = grid * grid
        self.mixers = nn.Sequential(*[Mixer(self.tokens, hidden) for _ in range(depth)])
        self.channel_projection = nn.Linear(late_channels, 128)
        self.row_projection = nn.Linear(self.tokens, 4)

    def raw(self, x):
        x = self.adapter(self.backbone(x)).flatten(2)
        x = self.mixers(x)
        x = self.channel_projection(x.permute(0, 2, 1)).permute(0, 2, 1)
        return self.row_projection(x).flatten(1)

    def forward(self, x):
        return F.normalize(self.raw(x), p=2, dim=1, eps=1e-12)


class PreL2(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model.raw(x)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def conv_linear_macs(model: nn.Module, sample: torch.Tensor) -> int:
    total = 0
    handles = []

    def hook(module, inputs, output):
        nonlocal total
        if isinstance(module, nn.Conv2d):
            total += (output.numel() * (module.in_channels // module.groups)
                      * module.kernel_size[0] * module.kernel_size[1])
        elif isinstance(module, nn.Linear):
            total += output.numel() * module.in_features

    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            handles.append(module.register_forward_hook(hook))
    with torch.no_grad():
        model.raw(sample)
    for handle in handles:
        handle.remove()
    return total


def canonicalize(source: Path, destination: Path) -> dict:
    model = onnx.load(source)
    before = len(model.graph.node)
    reshape = None
    for node in model.graph.node:
        if node.op_type != "Reshape" or len(node.input) != 2:
            continue
        producer = next((n for n in model.graph.node if node.input[1] in n.output), None)
        if producer is not None and producer.op_type == "Concat":
            reshape = node
            break
    if reshape is None:
        raise RuntimeError("static Concat -> Reshape chain not found")
    shape = None
    for value in list(model.graph.value_info) + list(model.graph.input):
        if value.name == reshape.input[0]:
            shape = [d.dim_value for d in value.type.tensor_type.shape.dim]
            break
    if not shape or len(shape) != 4 or any(v <= 0 for v in shape):
        raise RuntimeError(f"reshape input is not fully static: {shape}")
    replacement = np.asarray([shape[0], shape[1], -1], dtype=np.int64)
    name = f"static_reshape_1x{shape[1]}x{shape[2] * shape[3]}"
    reshape.input[1] = name
    model.graph.initializer.append(numpy_helper.from_array(replacement, name))
    removed = []
    while True:
        live = {v for node in model.graph.node for v in node.input}
        dead = [
            node for node in model.graph.node
            if node.op_type in {"Shape", "Slice", "Gather", "Unsqueeze", "Concat", "Cast", "Constant"}
            and all(output not in live for output in node.output)
        ]
        if not dead:
            break
        for node in dead:
            removed.append(node.name or node.output[0])
            model.graph.node.remove(node)
    model = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    onnx.checker.check_model(model)
    onnx.save(model, destination)
    return {"reshape_input_shape": shape, "initializer": replacement.tolist(),
            "nodes_before": before, "nodes_after": len(model.graph.node), "removed": removed}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid", type=int, choices=[9, 10, 12], required=True)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--hidden", type=int, required=True)
    parser.add_argument("--late-channels", type=int, choices=[144, 160, 176, 192], default=144)
    args = parser.parse_args()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    name = f"g{args.grid}x{args.grid}_d{args.depth}_h{args.hidden}_c{args.late_channels}"
    clean = MODEL_DIR / f"{name}_inferred.onnx"
    canonical = MODEL_DIR / f"{name}_canonical.onnx"
    torch.manual_seed(20260907)
    model = SQ240Probe(args.grid, args.depth, args.hidden, args.late_channels)

    # A real forward/backward/update construction sanity check, not training.
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
    sanity_input = torch.linspace(-1, 1, 2 * 3 * 240 * 240).reshape(2, 3, 240, 240)
    before = model.backbone[0].weight.detach().clone()
    sanity_output = model(sanity_input)
    # A weighted descriptor objective avoids the constant ||normalized||^2
    # sanity loss, whose exact gradient is zero by construction.
    weights = torch.linspace(-1, 1, sanity_output.numel()).reshape_as(sanity_output)
    loss = (sanity_output * weights).mean()
    optimizer.zero_grad(); loss.backward()
    gradients_finite = all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    optimizer.step()
    update_max_abs = float((model.backbone[0].weight.detach() - before).abs().max())

    model.eval()
    sample = torch.linspace(-1, 1, 3 * 240 * 240).reshape(1, 3, 240, 240)
    with torch.no_grad():
        backbone_shape = list(model.backbone(sample).shape)
        adapted_shape = list(model.adapter(model.backbone(sample)).shape)
    torch.onnx.export(PreL2(model), sample, str(clean), opset_version=13, dynamo=False,
                      input_names=["normalized_rgb"], output_names=["raw_descriptor"])
    inferred = onnx.shape_inference.infer_shapes(onnx.load(clean), strict_mode=True)
    onnx.checker.check_model(inferred)
    onnx.save(inferred, clean)
    rewrite = canonicalize(clean, canonical)
    clean_session = ort.InferenceSession(str(clean), providers=["CPUExecutionProvider"])
    canonical_session = ort.InferenceSession(str(canonical), providers=["CPUExecutionProvider"])
    tests = {
        "linspace": sample.numpy(),
        "random": np.random.default_rng(20260907).uniform(-1, 1, (1, 3, 240, 240)).astype(np.float32),
        "zeros": np.zeros((1, 3, 240, 240), dtype=np.float32),
    }
    parity = {}
    for test_name, value in tests.items():
        clean_raw = clean_session.run(None, {"normalized_rgb": value})[0]
        canonical_raw = canonical_session.run(None, {"normalized_rgb": value})[0]
        with torch.no_grad():
            full = model(torch.from_numpy(value)).numpy()
        cpu_l2 = canonical_raw / np.maximum(np.linalg.norm(canonical_raw, axis=1, keepdims=True), 1e-12)
        parity[test_name] = {
            "clean_canonical_max_abs": float(np.max(np.abs(clean_raw - canonical_raw))),
            "partition_max_abs": float(np.max(np.abs(full - cpu_l2))),
            "cosine": float(np.dot(full.ravel(), cpu_l2.ravel()) / (np.linalg.norm(full) * np.linalg.norm(cpu_l2))),
            "shape": list(canonical_raw.shape), "finite": bool(np.isfinite(canonical_raw).all()),
        }
    report = {
        "name": name, "input": [1, 3, 240, 240], "backbone_output": backbone_shape,
        "adapted_output": adapted_shape, "tokens": model.tokens, "depth": args.depth,
        "hidden": args.hidden, "late_channels": args.late_channels,
        "descriptor": 512, "params": sum(p.numel() for p in model.parameters()),
        "conv_linear_macs": conv_linear_macs(model, sample),
        "sanity": {"output": list(sanity_output.shape), "loss_finite": bool(torch.isfinite(loss)),
                   "gradients_finite": bool(gradients_finite), "update_max_abs": update_max_abs},
        "clean_onnx": str(clean), "canonical_onnx": str(canonical),
        "canonical_sha256": sha256(canonical), "rewrite": rewrite, "parity": parity,
    }
    (REPORT_DIR / f"{name}_export.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
