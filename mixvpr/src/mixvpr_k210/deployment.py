from __future__ import annotations
from pathlib import Path
import hashlib
import json

import numpy as np
import onnx
from onnx import numpy_helper
import onnxruntime as ort
import torch
import torch.nn as nn

from .config import spec_for_stage
from .model import K210MixVPR, count_params, conv_linear_macs

class PreL2(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model.raw(x)

def sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()

def _put(path: Path, value: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")

def canonicalize(source: Path, destination: Path) -> dict:
    model = onnx.load(source)
    before = len(model.graph.node)
    reshape = None
    for node in model.graph.node:
        if node.op_type != "Reshape" or len(node.input) != 2:
            continue
        producer = next(
            (n for n in model.graph.node if node.input[1] in n.output),
            None,
        )
        if producer is not None and producer.op_type == "Concat":
            reshape = node
            break
    if reshape is None:
        raise RuntimeError("static Concat -> Reshape chain not found")

    shape = None
    values = list(model.graph.value_info) + list(model.graph.input)
    for value in values:
        if value.name == reshape.input[0]:
            shape = [
                d.dim_value for d in value.type.tensor_type.shape.dim
            ]
            break
    if not shape or len(shape) != 4 or any(v <= 0 for v in shape):
        raise RuntimeError(f"reshape input is not fully static: {shape}")

    replacement = np.asarray(
        [shape[0], shape[1], -1], dtype=np.int64
    )
    name = f"static_reshape_1x{shape[1]}x{shape[2] * shape[3]}"
    reshape.input[1] = name
    model.graph.initializer.append(
        numpy_helper.from_array(replacement, name)
    )

    removed = []
    removable = {
        "Shape", "Slice", "Gather", "Unsqueeze",
        "Concat", "Cast", "Constant"
    }
    while True:
        live = {v for node in model.graph.node for v in node.input}
        dead = [
            node for node in model.graph.node
            if node.op_type in removable
            and all(output not in live for output in node.output)
        ]
        if not dead:
            break
        for node in dead:
            removed.append(node.name or node.output[0])
            model.graph.node.remove(node)

    model = onnx.shape_inference.infer_shapes(
        model, strict_mode=True
    )
    onnx.checker.check_model(model)
    onnx.save(model, destination)
    return {
        "reshape_input_shape": shape,
        "static_initializer": replacement.tolist(),
        "nodes_before": before,
        "nodes_after": len(model.graph.node),
        "removed": removed,
    }

def build_stage_model(cfg: dict, stage_id: str,
                      checkpoint: str | Path | None = None):
    model = K210MixVPR(spec_for_stage(cfg, stage_id))
    if checkpoint is not None:
        ckpt = torch.load(
            checkpoint, map_location="cpu", weights_only=False
        )
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
    return model.eval()

def export_stage(cfg: dict, stage_id: str, output: Path,
                 checkpoint: str | Path | None = None) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    model = build_stage_model(cfg, stage_id, checkpoint)
    raw = output / "raw.onnx"
    canonical = output / "model.onnx"
    sample = torch.linspace(
        -1, 1, 3 * 240 * 240, dtype=torch.float32
    ).reshape(1, 3, 240, 240)

    torch.onnx.export(
        PreL2(model), sample, str(raw),
        opset_version=13, dynamo=False,
        input_names=["normalized_rgb"],
        output_names=["raw_descriptor"],
    )
    inferred = onnx.shape_inference.infer_shapes(
        onnx.load(raw), strict_mode=True
    )
    onnx.checker.check_model(inferred)
    onnx.save(inferred, raw)
    rewrite = canonicalize(raw, canonical)

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 2
    opts.inter_op_num_threads = 1
    session = ort.InferenceSession(
        str(canonical), opts,
        providers=["CPUExecutionProvider"],
    )

    tests = {
        "linspace": sample.numpy(),
        "zeros": np.zeros((1, 3, 240, 240), np.float32),
        "random": np.random.default_rng(20260920).normal(
            0, 1, (1, 3, 240, 240)
        ).astype(np.float32),
    }
    parity = {}
    for name, x in tests.items():
        with torch.inference_mode():
            expected = model.raw(torch.from_numpy(x)).numpy()
        actual = session.run(
            None, {"normalized_rgb": x}
        )[0]
        np.testing.assert_allclose(
            actual, expected, atol=1e-4, rtol=1e-3
        )
        parity[name] = {
            "max_abs": float(np.max(np.abs(actual - expected))),
            "shape": list(actual.shape),
            "finite": bool(np.isfinite(actual).all()),
        }

    report = {
        "status": "PASS",
        "stage_id": stage_id,
        "checkpoint": str(checkpoint) if checkpoint else None,
        "checkpoint_sha256": (
            sha256(checkpoint) if checkpoint else None
        ),
        "params": count_params(model),
        "conv_linear_macs": conv_linear_macs(model),
        "descriptor": model.spec.descriptor_dim,
        "mixer_depth": model.spec.mixer_depth,
        "widths": list(model.spec.widths),
        "hidden": model.spec.mixer_hidden,
        "onnx_sha256": sha256(canonical),
        "rewrite": rewrite,
        "parity": parity,
    }
    _put(output / "export.json", report)
    return report
