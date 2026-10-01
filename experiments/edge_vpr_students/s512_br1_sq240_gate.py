"""Frozen BR1-SQ240 architecture and PyTorch/ONNX/pre-L2 gate."""
from __future__ import annotations

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
MODELS = ROOT / "models"
REPORTS = ROOT / "reports" / "edge_vpr_students"


class Mixer49(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(49)
        self.fc1 = nn.Linear(49, 128)
        self.fc2 = nn.Linear(128, 49)

    def forward(self, x):
        return x + self.fc2(F.relu(self.fc1(self.norm(x))))


class BR1SQ240(nn.Module):
    def __init__(self):
        super().__init__()
        widths = [16, 24, 64, 112, 144]
        layers = []
        channels = 3
        for stage, width in enumerate(widths):
            layers.extend(
                [
                    nn.Conv2d(channels, width, 3, 2 if stage == 0 else 1, 1),
                    nn.ReLU(),
                    nn.Conv2d(width, width, 3, 1, 1),
                    nn.ReLU(),
                ]
            )
            if stage < 4:
                layers.append(nn.MaxPool2d(2, 2))
            channels = width
        self.backbone = nn.Sequential(*layers)
        self.mixers = nn.Sequential(Mixer49(), Mixer49())
        self.channel_projection = nn.Linear(144, 128)
        self.row_projection = nn.Linear(49, 4)

    def raw(self, x):
        x = self.backbone(x).flatten(2)
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


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonicalize(source, destination):
    model = onnx.load(source)
    before = len(model.graph.node)
    reshape = None
    producer = None
    for node in model.graph.node:
        if node.op_type == "Reshape" and len(node.input) == 2:
            candidate = next((n for n in model.graph.node if node.input[1] in n.output), None)
            if candidate is not None and candidate.op_type == "Concat":
                reshape, producer = node, candidate
                break
    if reshape is None:
        raise RuntimeError("static Concat-to-Reshape chain not found")
    shape = None
    for value in list(model.graph.value_info) + list(model.graph.input):
        if value.name == reshape.input[0]:
            shape = [d.dim_value for d in value.type.tensor_type.shape.dim]
            break
    if shape != [1, 144, 7, 7]:
        raise RuntimeError(f"unexpected real SQ240 backbone shape: {shape}")
    name = "sq240_static_reshape_1x144x49"
    reshape.input[1] = name
    model.graph.initializer.append(numpy_helper.from_array(np.asarray([1, 144, -1], np.int64), name))
    removed = []
    while True:
        live = {v for node in model.graph.node for v in node.input}
        dead = [
            node
            for node in model.graph.node
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
    return {"shape": shape, "initializer": [1, 144, -1], "nodes": [before, len(model.graph.node)], "removed": removed}


def main():
    torch.manual_seed(20260906)
    model = BR1SQ240().eval()
    sample = torch.linspace(-1, 1, 3 * 240 * 240).reshape(1, 3, 240, 240)
    with torch.no_grad():
        feature = model.backbone(sample)
        full = model(sample)
    assert list(feature.shape) == [1, 144, 7, 7]
    assert list(full.shape) == [1, 512]
    clean = MODELS / "s512_br1_sq240_prenorm_inferred.onnx"
    canonical = MODELS / "s512_br1_sq240_prenorm_canonical.onnx"
    torch.onnx.export(
        PreL2(model), sample, str(clean), opset_version=13, dynamo=False,
        input_names=["normalized_rgb"], output_names=["raw_descriptor"]
    )
    graph = onnx.shape_inference.infer_shapes(onnx.load(clean), strict_mode=True)
    onnx.checker.check_model(graph)
    onnx.save(graph, clean)
    rewrite = canonicalize(clean, canonical)
    sessions = [ort.InferenceSession(str(path), providers=["CPUExecutionProvider"]) for path in (clean, canonical)]
    parity = {}
    tests = {
        "linspace": sample.numpy(),
        "random": np.random.default_rng(20260906).uniform(-1, 1, (1, 3, 240, 240)).astype(np.float32),
        "zeros": np.zeros((1, 3, 240, 240), np.float32),
    }
    for name, value in tests.items():
        raw_clean, raw_canonical = [s.run(None, {"normalized_rgb": value})[0] for s in sessions]
        with torch.no_grad():
            normalized = model(torch.from_numpy(value)).numpy()
        cpu_normalized = raw_canonical / np.maximum(np.linalg.norm(raw_canonical, axis=1, keepdims=True), 1e-12)
        parity[name] = {
            "clean_canonical_max_abs": float(np.max(np.abs(raw_clean - raw_canonical))),
            "partition_max_abs": float(np.max(np.abs(normalized - cpu_normalized))),
            "cosine": float(np.dot(normalized.ravel(), cpu_normalized.ravel()) / (np.linalg.norm(normalized) * np.linalg.norm(cpu_normalized))),
            "raw_shape": list(raw_canonical.shape),
            "raw_norm": float(np.linalg.norm(raw_canonical)),
            "final_norm": float(np.linalg.norm(cpu_normalized)),
        }
    report = {
        "identity": "S512-BR1-SQ240",
        "input": [1, 3, 240, 240],
        "backbone_output": list(feature.shape),
        "tokens": 49,
        "mixers": 2,
        "hidden": 128,
        "descriptor": 512,
        "params": sum(p.numel() for p in model.parameters()),
        "canonical_onnx": str(canonical),
        "canonical_sha256": sha(canonical),
        "rewrite": rewrite,
        "parity": parity,
    }
    (REPORTS / "s512_br1_sq240_gate.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
