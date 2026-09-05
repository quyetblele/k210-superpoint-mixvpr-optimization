#!/usr/bin/env python3
"""Create the minimal static FP32 ONNX model used by Gate 1."""

from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, checker, helper, numpy_helper


REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = REPO_ROOT / "models" / "tiny_fp32.onnx"


def main() -> None:
    # Deterministic FP32 initializers; no framework export is involved.
    rng = np.random.default_rng(0)
    conv1_weight = rng.standard_normal((8, 3, 3, 3)).astype(np.float32)
    conv1_bias = np.zeros((8,), dtype=np.float32)
    conv2_weight = rng.standard_normal((8, 8, 3, 3)).astype(np.float32)
    conv2_bias = np.zeros((8,), dtype=np.float32)

    model_input = helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, [1, 3, 32, 32]
    )
    model_output = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, [1, 8, 32, 32]
    )

    nodes = [
        helper.make_node(
            "Conv",
            ["input", "conv1_weight", "conv1_bias"],
            ["conv1_out"],
            name="conv1",
            kernel_shape=[3, 3],
            pads=[1, 1, 1, 1],
            strides=[1, 1],
        ),
        helper.make_node("Relu", ["conv1_out"], ["relu1_out"], name="relu1"),
        helper.make_node(
            "Conv",
            ["relu1_out", "conv2_weight", "conv2_bias"],
            ["conv2_out"],
            name="conv2",
            kernel_shape=[3, 3],
            pads=[1, 1, 1, 1],
            strides=[1, 1],
        ),
        helper.make_node("Relu", ["conv2_out"], ["output"], name="relu2"),
    ]
    graph = helper.make_graph(
        nodes,
        "tiny_fp32",
        [model_input],
        [model_output],
        initializer=[
            numpy_helper.from_array(conv1_weight, "conv1_weight"),
            numpy_helper.from_array(conv1_bias, "conv1_bias"),
            numpy_helper.from_array(conv2_weight, "conv2_weight"),
            numpy_helper.from_array(conv2_bias, "conv2_bias"),
        ],
    )
    model = helper.make_model(
        graph,
        producer_name="k210_lab",
        opset_imports=[helper.make_opsetid("", 13)],
    )
    model.ir_version = onnx.IR_VERSION
    checker.check_model(model)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, MODEL_PATH)
    print(f"Created: {MODEL_PATH}")
    print("Input: input float32 [1, 3, 32, 32]")
    print("Output: output float32 [1, 8, 32, 32]")
    print("Operators: Conv, Relu, Conv, Relu")
    print("ONNX checker: PASS")


if __name__ == "__main__":
    main()
