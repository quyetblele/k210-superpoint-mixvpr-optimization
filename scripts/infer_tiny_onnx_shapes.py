#!/usr/bin/env python3
"""Serialize ONNX shape-inference metadata for the Gate 2 nncase importer."""

from pathlib import Path

import onnx
from onnx import TensorProto, checker, shape_inference


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = REPO_ROOT / "models" / "tiny_fp32.onnx"
OUTPUT_PATH = REPO_ROOT / "models" / "tiny_fp32_inferred.onnx"
REQUIRED_VALUE_INFO = ("conv1_out", "relu1_out", "conv2_out")


def describe(value_info):
    tensor_type = value_info.type.tensor_type
    dtype = TensorProto.DataType.Name(tensor_type.elem_type)
    shape = []
    for dim in tensor_type.shape.dim:
        if not dim.HasField("dim_value"):
            raise RuntimeError("{} has a non-static dimension".format(value_info.name))
        shape.append(dim.dim_value)
    return dtype, shape


def main():
    if not SOURCE_PATH.is_file():
        raise FileNotFoundError("Source model not found: {}".format(SOURCE_PATH))

    inferred_model = shape_inference.infer_shapes(onnx.load(SOURCE_PATH))
    checker.check_model(inferred_model)
    onnx.save(inferred_model, OUTPUT_PATH)

    # Reload from disk: this verifies the metadata was serialized, not merely
    # present in the in-memory inference result.
    serialized_model = onnx.load(OUTPUT_PATH)
    checker.check_model(serialized_model)
    value_info = {entry.name: entry for entry in serialized_model.graph.value_info}
    missing = [name for name in REQUIRED_VALUE_INFO if name not in value_info]
    if missing:
        raise RuntimeError("Serialized value_info missing: {}".format(", ".join(missing)))

    print("Inferred ONNX checker: PASS")
    print("Saved: {}".format(OUTPUT_PATH))
    print("Serialized graph.value_info:")
    for name in REQUIRED_VALUE_INFO:
        dtype, shape = describe(value_info[name])
        print("  {}: {} {}".format(name, dtype, shape))


if __name__ == "__main__":
    main()
