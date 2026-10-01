#!/usr/bin/env python3
"""Compare raw PyTorch wrapper outputs with static ONNX Runtime outputs."""

from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from superpoint_kpu_wrapper import SuperPointKPUWrapper, load_local_baseline


MODEL_PATH = (
    Path(__file__).resolve().parents[1]
    / "models"
    / "superpoint_kpu_fp32_320x240_inferred.onnx"
)
OUTPUT_NAMES = ("detector_logits", "descriptor_map")
MAX_ABS_TOLERANCE = 5e-4
MEAN_ABS_TOLERANCE = 1e-4
MIN_COSINE_SIMILARITY = 0.999999


def metrics(reference, actual):
    reference64 = reference.astype(np.float64).ravel()
    actual64 = actual.astype(np.float64).ravel()
    absolute = np.abs(reference64 - actual64)
    cosine = np.dot(reference64, actual64) / (
        np.linalg.norm(reference64) * np.linalg.norm(actual64)
    )
    return float(absolute.max()), float(absolute.mean()), float(cosine)


def deterministic_inputs():
    rng = np.random.default_rng(20260905)
    return [
        np.linspace(0.0, 1.0, num=240 * 320, dtype=np.float32).reshape(1, 1, 240, 320),
        rng.uniform(0.0, 1.0, size=(1, 1, 240, 320)).astype(np.float32),
        rng.uniform(0.0, 1.0, size=(1, 1, 240, 320)).astype(np.float32),
    ]


def main() -> None:
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    wrapper = SuperPointKPUWrapper(load_local_baseline()).eval().cpu()
    session = ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])

    failed = False
    for sample_index, image in enumerate(deterministic_inputs()):
        with torch.inference_mode():
            torch_outputs = wrapper(torch.from_numpy(image))
        onnx_outputs = session.run(list(OUTPUT_NAMES), {"image": image})
        for name, torch_output, onnx_output in zip(
            OUTPUT_NAMES, torch_outputs, onnx_outputs
        ):
            max_abs, mean_abs, cosine = metrics(torch_output.numpy(), onnx_output)
            print(
                "sample={} output={} max_abs={:.9g} mean_abs={:.9g} cosine={:.12f}".format(
                    sample_index, name, max_abs, mean_abs, cosine
                )
            )
            if (
                max_abs > MAX_ABS_TOLERANCE
                or mean_abs > MEAN_ABS_TOLERANCE
                or cosine < MIN_COSINE_SIMILARITY
            ):
                failed = True

    if failed:
        raise RuntimeError("PyTorch/ONNX parity thresholds failed")
    print(
        "Thresholds: max_abs<={} mean_abs<={} cosine>={}".format(
            MAX_ABS_TOLERANCE, MEAN_ABS_TOLERANCE, MIN_COSINE_SIMILARITY
        )
    )
    print("PyTorch/ONNX raw-output parity: PASS")


if __name__ == "__main__":
    main()
