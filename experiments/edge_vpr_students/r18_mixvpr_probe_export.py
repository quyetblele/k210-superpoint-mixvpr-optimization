"""Export exactly the three frozen ResNet18-derived MixVPR K210 probes."""
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
MODEL_DIR = ROOT / "models/r18_mixvpr_probes"
REPORT_DIR = ROOT / "reports/edge_vpr_students/r18_mixvpr_probes"
SPECS = {
    "r18_196": (224, 196, 196),
    "r18_169": (208, 169, 169),
    "r18_144": (192, 144, 144),
    "r18_144_h96": (192, 144, 96),
    "r18_144_h112": (192, 144, 112),
}
WIDTHS = (20, 40, 80, 160)  # Uniform 5/16 width multiplier.


class BasicBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = (nn.Identity() if stride == 1 and in_channels == out_channels else
                         nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                                       nn.BatchNorm2d(out_channels)))

    def forward(self, x):
        identity = self.shortcut(x)
        x = F.relu(self.bn1(self.conv1(x)))
        return F.relu(self.bn2(self.conv2(x)) + identity)


class Mixer(nn.Module):
    def __init__(self, tokens: int, hidden: int):
        super().__init__()
        self.norm = nn.LayerNorm(tokens)
        self.fc1 = nn.Linear(tokens, hidden)
        self.fc2 = nn.Linear(hidden, tokens)

    def forward(self, x):
        return x + self.fc2(F.relu(self.fc1(self.norm(x))))


class R18MixVPR(nn.Module):
    def __init__(self, tokens: int, hidden: int):
        super().__init__(); self.tokens = tokens
        self.stem = nn.Sequential(nn.Conv2d(3, WIDTHS[0], 3, 2, 1, bias=False),
                                  nn.BatchNorm2d(WIDTHS[0]), nn.ReLU())
        stages = []; in_channels = WIDTHS[0]
        for stage, out_channels in enumerate(WIDTHS):
            stride = 1 if stage == 0 else 2
            stages.extend([BasicBlock(in_channels, out_channels, stride),
                           BasicBlock(out_channels, out_channels, 1)])
            in_channels = out_channels
        self.backbone = nn.Sequential(*stages)
        self.mixers = nn.Sequential(*[Mixer(tokens, hidden) for _ in range(4)])
        self.channel_projection = nn.Linear(160, 128)
        self.row_projection = nn.Linear(tokens, 4)

    def raw(self, x):
        x = self.backbone(self.stem(x)).flatten(2)
        x = self.mixers(x)
        x = self.channel_projection(x.permute(0, 2, 1)).permute(0, 2, 1)
        return self.row_projection(x).flatten(1)

    def forward(self, x):
        return F.normalize(self.raw(x), p=2, dim=1, eps=1e-12)


class PreL2(nn.Module):
    def __init__(self, model): super().__init__(); self.model = model
    def forward(self, x): return self.model.raw(x)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def macs(model: nn.Module, sample: torch.Tensor) -> int:
    total = 0; handles = []
    def hook(module, inputs, output):
        nonlocal total
        if isinstance(module, nn.Conv2d):
            total += output.numel() * (module.in_channels // module.groups) * module.kernel_size[0] * module.kernel_size[1]
        elif isinstance(module, nn.Linear):
            total += output.numel() * module.in_features
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)): handles.append(module.register_forward_hook(hook))
    with torch.no_grad(): model.raw(sample)
    for handle in handles: handle.remove()
    return total


def canonicalize(source: Path, destination: Path) -> dict:
    graph = onnx.load(source); before = len(graph.graph.node); replacements = []
    while True:
        producers = {output: node for node in graph.graph.node for output in node.output}
        target = next((node for node in graph.graph.node if node.op_type == "Reshape" and len(node.input) == 2
                       and node.input[1] in producers and producers[node.input[1]].op_type == "Concat"), None)
        if target is None: break
        info = next((value for value in list(graph.graph.value_info) + list(graph.graph.input)
                     if value.name == target.input[0]), None)
        shape = [dim.dim_value for dim in info.type.tensor_type.shape.dim] if info is not None else []
        if len(shape) != 4 or any(value <= 0 for value in shape):
            raise RuntimeError(f"non-static reshape data shape at {target.name}: {shape}")
        value = np.asarray([shape[0], shape[1], -1], np.int64)
        name = f"static_shape_{len(replacements)}_1x{shape[1]}x{shape[2] * shape[3]}"
        target.input[1] = name; graph.graph.initializer.append(numpy_helper.from_array(value, name))
        replacements.append({"reshape": target.name, "value": value.tolist(), "initializer": name})
    removed = []
    while True:
        live = {value for node in graph.graph.node for value in node.input}
        dead = [node for node in graph.graph.node if node.op_type in
                {"Shape", "Slice", "Gather", "Unsqueeze", "Concat", "Cast", "Constant"}
                and all(output not in live for output in node.output)]
        if not dead: break
        for node in dead: removed.append(node.name or node.output[0]); graph.graph.node.remove(node)
    graph = onnx.shape_inference.infer_shapes(graph, strict_mode=True); onnx.checker.check_model(graph)
    onnx.save(graph, destination)
    return {"replacements": replacements, "removed": removed, "nodes_before": before,
            "nodes_after": len(graph.graph.node)}


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--name", choices=tuple(SPECS), required=True)
    args = parser.parse_args(); size, tokens, hidden = SPECS[args.name]
    MODEL_DIR.mkdir(parents=True, exist_ok=True); REPORT_DIR.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(20260907); model = R18MixVPR(tokens, hidden).eval()
    sample = torch.linspace(-1, 1, 3 * size * size).reshape(1, 3, size, size)
    with torch.no_grad(): backbone_shape = list(model.backbone(model.stem(sample)).shape)
    if backbone_shape != [1, 160, size // 16, size // 16] or tokens != (size // 16) ** 2:
        raise RuntimeError(f"frozen shape specification mismatch: {backbone_shape}")
    clean = MODEL_DIR / f"{args.name}_inferred.onnx"; canonical = MODEL_DIR / f"{args.name}_canonical.onnx"
    deployment = PreL2(model).eval()
    torch.onnx.export(deployment, sample, str(clean), opset_version=13, dynamo=False,
                      input_names=["normalized_rgb"], output_names=["raw_descriptor"])
    model.eval()
    inferred = onnx.shape_inference.infer_shapes(onnx.load(clean), strict_mode=True)
    onnx.checker.check_model(inferred); onnx.save(inferred, clean)
    rewrite = canonicalize(clean, canonical)
    clean_ort = ort.InferenceSession(str(clean), providers=["CPUExecutionProvider"])
    canonical_ort = ort.InferenceSession(str(canonical), providers=["CPUExecutionProvider"])
    cases = {"linspace": sample.numpy(), "random": np.random.default_rng(20260907).uniform(
        -1, 1, (1, 3, size, size)).astype(np.float32), "zeros": np.zeros((1, 3, size, size), np.float32)}
    parity = {}
    for name, value in cases.items():
        a = clean_ort.run(None, {"normalized_rgb": value})[0]
        b = canonical_ort.run(None, {"normalized_rgb": value})[0]
        with torch.no_grad(): full = model(torch.from_numpy(value)).numpy()
        normalized = b / np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-12)
        parity[name] = {"clean_canonical_max_abs": float(np.max(np.abs(a - b))),
                        "partition_max_abs": float(np.max(np.abs(full - normalized))),
                        "cosine": float(np.dot(full.ravel(), normalized.ravel()) /
                                        (np.linalg.norm(full) * np.linalg.norm(normalized))),
                        "finite": bool(np.isfinite(b).all()), "shape": list(b.shape)}
    report = {"name": args.name, "family": "5/16-width ResNet18 BasicBlock, CIFAR-style 3x3 stem",
              "widths": list(WIDTHS), "input": [1, 3, size, size], "backbone_output": backbone_shape,
              "tokens": tokens, "mixer_depth": 4, "hidden": hidden, "descriptor": 512,
              "params": sum(parameter.numel() for parameter in model.parameters()),
              "conv_linear_macs": macs(model, sample), "canonical_onnx": str(canonical),
              "canonical_sha256": sha256(canonical), "rewrite": rewrite, "parity": parity}
    (REPORT_DIR / f"{args.name}_export.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__": main()
