#!/usr/bin/env python3
"""Inspect the Gate 1 ONNX model and validate its simple static graph."""

from collections import Counter
from pathlib import Path
import sys
from typing import Tuple

import onnx
from onnx import TensorProto, checker, shape_inference


MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "tiny_fp32.onnx"
FORBIDDEN_OPERATORS = {"QuantizeLinear", "DequantizeLinear", "TopK", "Resize"}


def format_type(value_info: onnx.ValueInfoProto) -> Tuple[str, str]:
    tensor_type = value_info.type.tensor_type
    dtype = TensorProto.DataType.Name(tensor_type.elem_type)
    dims = []
    for dim in tensor_type.shape.dim:
        if dim.HasField("dim_value"):
            dims.append(str(dim.dim_value))
        elif dim.HasField("dim_param"):
            dims.append(f"<{dim.dim_param}>")
        else:
            dims.append("?")
    return dtype, "[" + ", ".join(dims) + "]"


def has_dynamic_shape(value_info: onnx.ValueInfoProto) -> bool:
    return any(
        not dim.HasField("dim_value") for dim in value_info.type.tensor_type.shape.dim
    )


def main() -> int:
    if not MODEL_PATH.is_file():
        print(f"Model not found: {MODEL_PATH}", file=sys.stderr)
        return 1

    model = onnx.load(MODEL_PATH)
    inferred_model = shape_inference.infer_shapes(model)
    opsets = ", ".join(
        f"{entry.domain or 'ai.onnx'}={entry.version}" for entry in model.opset_import
    )
    histogram = Counter(node.op_type for node in model.graph.node)

    print(f"ONNX version: {onnx.__version__}")
    print(f"Opset: {opsets}")
    print("Inputs:")
    for value_info in model.graph.input:
        dtype, shape = format_type(value_info)
        print(f"  {value_info.name}: {dtype} {shape}")
    print("Outputs:")
    for value_info in inferred_model.graph.output:
        dtype, shape = format_type(value_info)
        print(f"  {value_info.name}: {dtype} {shape}")
    print("Operator histogram:")
    for operator, count in sorted(histogram.items()):
        print(f"  {operator}: {count}")

    try:
        checker.check_model(model)
    except onnx.checker.ValidationError as error:
        print(f"ONNX checker: FAIL ({error})")
        return 1
    print("ONNX checker: PASS")

    print("Inferred intermediate tensor shapes:")
    if inferred_model.graph.value_info:
        for value_info in inferred_model.graph.value_info:
            dtype, shape = format_type(value_info)
            print(f"  {value_info.name}: {dtype} {shape}")
    else:
        print("  (none available)")

    non_standard_domains = sorted({node.domain for node in model.graph.node if node.domain})
    dynamic_inputs = [
        value_info.name for value_info in model.graph.input if has_dynamic_shape(value_info)
    ]
    forbidden = sorted(set(histogram) & FORBIDDEN_OPERATORS)
    if non_standard_domains or dynamic_inputs or forbidden:
        print("Gate 1 graph constraints: FAIL")
        if non_standard_domains:
            print(f"  Non-standard domains: {', '.join(non_standard_domains)}")
        if dynamic_inputs:
            print(f"  Dynamic inputs: {', '.join(dynamic_inputs)}")
        if forbidden:
            print(f"  Forbidden operators: {', '.join(forbidden)}")
        return 1

    print("Gate 1 graph constraints: PASS (static, standard operators only)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
