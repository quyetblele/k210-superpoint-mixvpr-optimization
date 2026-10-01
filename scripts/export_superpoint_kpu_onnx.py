#!/usr/bin/env python3
"""Export the verified raw-head SuperPoint CNN as static FP32 ONNX.

This exports only the shared CNN encoder, convPb detector logits, and convDb
descriptor map. It does not export softmax, normalization, NMS, TopK, border
removal, or descriptor sampling.
"""

from hashlib import sha256
from pathlib import Path

import onnx
from onnx import checker, shape_inference
import torch

from superpoint_kpu_wrapper import (
    CHECKPOINT_PATH,
    SuperPointKPUWrapper,
    load_local_baseline,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
ONNX_PATH = REPO_ROOT / "models" / "superpoint_kpu_fp32_320x240.onnx"
INFERRED_ONNX_PATH = (
    REPO_ROOT / "models" / "superpoint_kpu_fp32_320x240_inferred.onnx"
)
EXPECTED_CHECKPOINT_SHA256 = (
    "52b6708629640ca883673b5d5c097c4ddad37d8048b33f09c8ca0d69db12c40e"
)
INPUT_SHAPE = (1, 1, 240, 320)


def verify_checkpoint() -> None:
    digest = sha256(CHECKPOINT_PATH.read_bytes()).hexdigest()
    if digest != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError(
            "Checkpoint SHA256 mismatch: expected {}, got {}".format(
                EXPECTED_CHECKPOINT_SHA256, digest
            )
        )
    print("Checkpoint SHA256: {} (PASS)".format(digest))


def verify_raw_wrapper_parity(baseline, wrapper, image) -> None:
    captured = {}

    def capture(name):
        def hook(_module, _inputs, output):
            captured[name] = output.detach().clone()

        return hook

    handles = [
        baseline.net.convPb.register_forward_hook(capture("detector_logits")),
        baseline.net.convDb.register_forward_hook(capture("descriptor_map")),
    ]
    try:
        with torch.inference_mode():
            baseline({"image": image})
    finally:
        for handle in handles:
            handle.remove()

    with torch.inference_mode():
        outputs = wrapper(image)
    for name, output in zip(("detector_logits", "descriptor_map"), outputs):
        if not torch.equal(output, captured[name]):
            error = (output - captured[name]).abs().max().item()
            raise RuntimeError("{} wrapper parity failed: {}".format(name, error))
        print("Pre-export parity {}: exact, max_abs_error=0".format(name))


def verify_serialized_metadata(model) -> None:
    if not model.graph.value_info:
        raise RuntimeError("Inferred ONNX has no serialized graph.value_info")
    names = {value.name for value in model.graph.value_info}
    node_outputs = {name for node in model.graph.node for name in node.output}
    intermediate_outputs = node_outputs - {value.name for value in model.graph.output}
    missing = sorted(intermediate_outputs - names)
    if missing:
        raise RuntimeError(
            "Serialized value_info missing node outputs: {}".format(", ".join(missing))
        )
    print(
        "Serialized intermediate value_info: {} entries (PASS)".format(
            len(model.graph.value_info)
        )
    )


def main() -> None:
    verify_checkpoint()
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    baseline = load_local_baseline()
    wrapper = SuperPointKPUWrapper(baseline).eval().cpu()
    image = torch.linspace(0.0, 1.0, steps=240 * 320, dtype=torch.float32).reshape(
        INPUT_SHAPE
    )
    verify_raw_wrapper_parity(baseline, wrapper, image)

    ONNX_PATH.parent.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        torch.onnx.export(
            wrapper,
            image,
            str(ONNX_PATH),
            export_params=True,
            opset_version=13,
            do_constant_folding=True,
            input_names=["image"],
            output_names=["detector_logits", "descriptor_map"],
            keep_initializers_as_inputs=False,
            dynamo=False,
        )

    exported = onnx.load(ONNX_PATH)
    checker.check_model(exported)
    print("Exported ONNX checker: PASS")

    inferred = shape_inference.infer_shapes(exported)
    checker.check_model(inferred)
    onnx.save(inferred, INFERRED_ONNX_PATH)

    # Reload to prove value_info is present in the serialized artifact.
    serialized_inferred = onnx.load(INFERRED_ONNX_PATH)
    checker.check_model(serialized_inferred)
    verify_serialized_metadata(serialized_inferred)
    print("Inferred ONNX checker: PASS")
    print("Exported: {}".format(ONNX_PATH))
    print("Inferred: {}".format(INFERRED_ONNX_PATH))


if __name__ == "__main__":
    main()
