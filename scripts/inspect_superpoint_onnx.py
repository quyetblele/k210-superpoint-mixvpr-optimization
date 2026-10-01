#!/usr/bin/env python3
"""Inspect the static inferred SuperPoint KPU ONNX graph."""

from collections import Counter
from pathlib import Path
import sys

import onnx
from onnx import TensorProto, checker


DEFAULT_MODEL = (
    Path(__file__).resolve().parents[1]
    / "models"
    / "superpoint_kpu_fp32_320x240_inferred.onnx"
)


def tensor_spec(value_info):
    tensor_type = value_info.type.tensor_type
    dtype = TensorProto.DataType.Name(tensor_type.elem_type)
    dims = []
    dynamic = False
    for dim in tensor_type.shape.dim:
        if dim.HasField("dim_value"):
            dims.append(dim.dim_value)
        elif dim.HasField("dim_param"):
            dims.append(dim.dim_param)
            dynamic = True
        else:
            dims.append("?")
            dynamic = True
    return dtype, dims, dynamic


def main() -> int:
    path = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else DEFAULT_MODEL
    model = onnx.load(path)
    checker_result = "PASS"
    try:
        checker.check_model(model)
    except checker.ValidationError as error:
        checker_result = "FAIL: {}".format(error)

    print("Model: {}".format(path))
    print("ONNX version: {}".format(onnx.__version__))
    print(
        "Opset: {}".format(
            ", ".join(
                "{}={}".format(entry.domain or "ai.onnx", entry.version)
                for entry in model.opset_import
            )
        )
    )
    print("Inputs:")
    for value in model.graph.input:
        dtype, shape, _ = tensor_spec(value)
        print("  {}: {} {}".format(value.name, dtype, shape))
    print("Outputs:")
    for value in model.graph.output:
        dtype, shape, _ = tensor_spec(value)
        print("  {}: {} {}".format(value.name, dtype, shape))

    histogram = Counter(node.op_type for node in model.graph.node)
    print("Node count: {}".format(len(model.graph.node)))
    print("Operator histogram:")
    for operator, count in sorted(histogram.items()):
        print("  {}: {}".format(operator, count))

    print("Intermediate tensor shapes ({}):".format(len(model.graph.value_info)))
    sized = []
    dynamic_names = []
    for value in model.graph.value_info:
        dtype, shape, dynamic = tensor_spec(value)
        print("  {}: {} {}".format(value.name, dtype, shape))
        if dynamic:
            dynamic_names.append(value.name)
        elif all(isinstance(dim, int) for dim in shape):
            elements = 1
            for dim in shape:
                elements *= dim
            sized.append((elements, value.name, dtype, shape))

    print("Largest intermediate tensors by elements:")
    for elements, name, dtype, shape in sorted(sized, reverse=True)[:10]:
        print("  {}: {} {} ({} elements)".format(name, dtype, shape, elements))

    custom_domains = sorted(
        {node.domain for node in model.graph.node if node.domain not in ("", "ai.onnx")}
    )
    for value in list(model.graph.input) + list(model.graph.output):
        _dtype, _shape, dynamic = tensor_spec(value)
        if dynamic:
            dynamic_names.append(value.name)
    print("Custom domains: {}".format(custom_domains or "none"))
    print("Dynamic dimensions: {}".format(sorted(set(dynamic_names)) or "none"))
    print("ONNX checker: {}".format(checker_result))
    return 0 if checker_result == "PASS" and not custom_domains and not dynamic_names else 1


if __name__ == "__main__":
    raise SystemExit(main())
