from __future__ import annotations
from pathlib import Path
import json

import numpy as np
import onnx
from onnx import numpy_helper
import onnxruntime as ort
import torch
import torch.nn as nn

from .data import sha256

class PreL2(nn.Module):
    """Deployment partition: KPU graph emits raw 512D; CPU performs L2."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model.raw(x)

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
    for value in list(model.graph.value_info) + list(model.graph.input):
        if value.name == reshape.input[0]:
            shape = [
                dim.dim_value
                for dim in value.type.tensor_type.shape.dim
            ]
            break
    if not shape or len(shape) != 4 or any(v <= 0 for v in shape):
        raise RuntimeError(f"reshape input is not fully static: {shape}")

    replacement = np.asarray(
        [shape[0], shape[1], -1], dtype=np.int64
    )
    name = (
        f"static_reshape_1x{shape[1]}x"
        f"{shape[2] * shape[3]}"
    )
    reshape.input[1] = name
    model.graph.initializer.append(
        numpy_helper.from_array(replacement, name)
    )

    removed = []
    removable = {
        "Shape", "Slice", "Gather", "Unsqueeze",
        "Concat", "Cast", "Constant",
    }
    while True:
        live = {
            value
            for node in model.graph.node
            for value in node.input
        }
        dead = [
            node for node in model.graph.node
            if node.op_type in removable
            and all(out not in live for out in node.output)
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
        "initializer": replacement.tolist(),
        "nodes_before": before,
        "nodes_after": len(model.graph.node),
        "removed": removed,
    }

def export_onnx(model, output: Path, sample: torch.Tensor) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    raw = output / "raw.onnx"
    canonical = output / "model.onnx"
    model.eval()

    torch.onnx.export(
        PreL2(model),
        sample,
        str(raw),
        opset_version=13,
        dynamo=False,
        input_names=["normalized_rgb"],
        output_names=["raw_descriptor"],
    )
    inferred = onnx.shape_inference.infer_shapes(
        onnx.load(raw), strict_mode=True
    )
    onnx.checker.check_model(inferred)
    onnx.save(inferred, raw)
    rewrite = canonicalize(raw, canonical)

    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(
        str(canonical),
        options,
        providers=["CPUExecutionProvider"],
    )

    tests = {
        "sample": sample.detach().cpu().numpy().astype(np.float32),
        "zeros": np.zeros(
            tuple(sample.shape), dtype=np.float32
        ),
    }
    parity = {}
    with torch.inference_mode():
        for name, value in tests.items():
            expected = model.raw(
                torch.from_numpy(value)
            ).cpu().numpy()
            actual = session.run(
                None, {"normalized_rgb": value}
            )[0]
            np.testing.assert_allclose(
                actual, expected, atol=1e-4, rtol=1e-3
            )
            parity[name] = {
                "shape": list(actual.shape),
                "max_abs": float(
                    np.max(np.abs(actual - expected))
                ),
                "finite": bool(np.isfinite(actual).all()),
            }

    report = {
        "status": "PASS",
        "raw_onnx": str(raw),
        "canonical_onnx": str(canonical),
        "onnx_sha256": sha256(canonical),
        "rewrite": rewrite,
        "parity": parity,
        "output_contract": "raw 512D FP32; CPU L2 eps=1e-12",
    }
    (output / "export.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    return report
