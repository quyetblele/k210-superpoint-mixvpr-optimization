"""Export frozen-backbone BR2M resource probes for the K210 compiler.

This is a compiler-feasibility sweep only.  Every probe keeps the BR2M input,
CNN widths/downsampling, 512-D head, and CPU-side final L2 partition fixed.
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
MODEL_DIR = ROOT / "models" / "br2m_resource_probes"
REPORT_DIR = ROOT / "reports" / "edge_vpr_students" / "br2m_resource_probes"


class Mixer(nn.Module):
    def __init__(self, tokens: int, hidden: int):
        super().__init__()
        self.norm = nn.LayerNorm(tokens)
        self.fc1 = nn.Linear(tokens, hidden)
        self.fc2 = nn.Linear(hidden, tokens)

    def forward(self, x):
        return x + self.fc2(F.relu(self.fc1(self.norm(x))))


class BR2MProbe(nn.Module):
    def __init__(self, grid: str, depth: int, hidden: int):
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
        adapters = {
            "22x12": nn.Identity(),
            "22x6": nn.MaxPool2d((1, 2), (1, 2)),
            "11x12": nn.MaxPool2d((2, 1), (2, 1)),
            # Overlapping rectangular pooling provides static intermediate
            # token grids for allocator boundary localization without touching
            # the frozen CNN.  Input to every adapter is exactly 22x12.
            "11x11": nn.MaxPool2d((2, 2), (2, 1)),
            "11x10": nn.MaxPool2d((2, 3), (2, 1)),
            "11x9": nn.MaxPool2d((2, 4), (2, 1)),
            "11x8": nn.MaxPool2d((2, 5), (2, 1)),
            "11x7": nn.MaxPool2d((2, 6), (2, 1)),
            "11x6": nn.MaxPool2d(2, 2),
        }
        if grid not in adapters:
            raise ValueError(f"unsupported grid: {grid}")
        self.adapter = adapters[grid]
        height, width = (int(v) for v in grid.split("x"))
        self.tokens = height * width
        self.mixers = nn.Sequential(*[Mixer(self.tokens, hidden) for _ in range(depth)])
        self.channel_projection = nn.Linear(144, 128)
        self.row_projection = nn.Linear(self.tokens, 4)

    def forward(self, x):
        x = self.adapter(self.backbone(x)).flatten(2)
        x = self.mixers(x)
        x = self.channel_projection(x.permute(0, 2, 1)).permute(0, 2, 1)
        return self.row_projection(x).flatten(1)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonicalize_static_reshape(source: Path, destination: Path) -> dict:
    model = onnx.load(source)
    before = len(model.graph.node)
    consumers = {}
    for node in model.graph.node:
        for value in node.input:
            consumers.setdefault(value, []).append(node)
    target = None
    for node in model.graph.node:
        if node.op_type != "Reshape" or len(node.input) != 2:
            continue
        shape_input = node.input[1]
        producer = next((n for n in model.graph.node if shape_input in n.output), None)
        if producer is not None and producer.op_type == "Concat":
            target = (node, producer)
            break
    if target is None:
        raise RuntimeError("dynamic Concat -> Reshape shape chain was not found")
    reshape, concat = target
    data_shape = None
    for info in list(model.graph.value_info) + list(model.graph.input) + list(model.graph.output):
        if info.name == reshape.input[0]:
            data_shape = [d.dim_value for d in info.type.tensor_type.shape.dim]
            break
    if not data_shape or len(data_shape) != 4 or any(v <= 0 for v in data_shape):
        raise RuntimeError(f"reshape data shape is not fully static: {data_shape}")
    replacement = np.asarray([data_shape[0], data_shape[1], -1], dtype=np.int64)
    initializer_name = f"static_reshape_1x{data_shape[1]}x{data_shape[2] * data_shape[3]}"
    reshape.input[1] = initializer_name
    model.graph.initializer.append(numpy_helper.from_array(replacement, initializer_name))

    live = {value for node in model.graph.node for value in node.input}
    removed = []
    changed = True
    while changed:
        changed = False
        for node in list(model.graph.node):
            if node.op_type in {"Shape", "Slice", "Gather", "Unsqueeze", "Concat", "Cast", "Constant"}:
                if all(output not in live for output in node.output):
                    removed.append(node.name or node.output[0])
                    model.graph.node.remove(node)
                    live = {value for current in model.graph.node for value in current.input}
                    changed = True
    inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    onnx.checker.check_model(inferred)
    onnx.save(inferred, destination)
    return {
        "reshape_node": reshape.name,
        "reshape_data_shape": data_shape,
        "initializer": initializer_name,
        "initializer_value": replacement.tolist(),
        "nodes_before": before,
        "nodes_after": len(inferred.graph.node),
        "removed_nodes": removed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--grid",
        required=True,
        choices=["22x12", "22x6", "11x12", "11x11", "11x10", "11x9", "11x8", "11x7", "11x6"],
    )
    parser.add_argument("--depth", required=True, type=int)
    parser.add_argument("--hidden", required=True, type=int)
    args = parser.parse_args()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    name = f"g{args.grid}_d{args.depth}_h{args.hidden}"
    clean = MODEL_DIR / f"{name}_inferred.onnx"
    canonical = MODEL_DIR / f"{name}_canonical.onnx"
    torch.manual_seed(20260906)
    model = BR2MProbe(args.grid, args.depth, args.hidden).eval()
    sample = torch.linspace(-1, 1, 3 * 352 * 192).reshape(1, 3, 352, 192)
    with torch.no_grad():
        output = model(sample)
    torch.onnx.export(
        model,
        sample,
        str(clean),
        opset_version=13,
        dynamo=False,
        input_names=["normalized_rgb"],
        output_names=["raw_descriptor"],
    )
    inferred = onnx.shape_inference.infer_shapes(onnx.load(clean), strict_mode=True)
    onnx.checker.check_model(inferred)
    onnx.save(inferred, clean)
    rewrite = canonicalize_static_reshape(clean, canonical)
    session_clean = ort.InferenceSession(str(clean), providers=["CPUExecutionProvider"])
    session_canonical = ort.InferenceSession(str(canonical), providers=["CPUExecutionProvider"])
    tests = {
        "linspace": sample.numpy(),
        "random": np.random.default_rng(20260906).uniform(-1, 1, (1, 3, 352, 192)).astype(np.float32),
        "zeros": np.zeros((1, 3, 352, 192), dtype=np.float32),
    }
    parity = {}
    for test_name, value in tests.items():
        a = session_clean.run(None, {"normalized_rgb": value})[0]
        b = session_canonical.run(None, {"normalized_rgb": value})[0]
        parity[test_name] = {
            "max_abs": float(np.max(np.abs(a - b))),
            "cosine": float(np.dot(a.ravel(), b.ravel()) / (np.linalg.norm(a) * np.linalg.norm(b))),
            "finite": bool(np.isfinite(b).all()),
            "shape": list(b.shape),
        }
    report = {
        "name": name,
        "input": [1, 3, 352, 192],
        "backbone_output": [1, 144, 22, 12],
        "grid": args.grid,
        "tokens": model.tokens,
        "depth": args.depth,
        "hidden": args.hidden,
        "params": sum(p.numel() for p in model.parameters()),
        "clean_onnx": str(clean),
        "canonical_onnx": str(canonical),
        "canonical_sha256": sha256(canonical),
        "rewrite": rewrite,
        "parity": parity,
    }
    (REPORT_DIR / f"{name}_export.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
