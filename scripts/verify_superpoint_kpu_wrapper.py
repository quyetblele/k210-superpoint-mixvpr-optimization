#!/usr/bin/env python3
"""Verify raw-output parity against hooks on the original SuperPoint model."""

from hashlib import sha256
import inspect
from pathlib import Path

import torch

from superpoint_kpu_wrapper import (
    BASELINE_CONFIG,
    CHECKPOINT_PATH,
    SuperPointKPUWrapper,
    load_local_baseline,
)


INPUT_SHAPE = (1, 1, 240, 320)


def main() -> None:
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)

    baseline = load_local_baseline()
    wrapper = SuperPointKPUWrapper(baseline).eval()
    image = torch.linspace(0.0, 1.0, steps=240 * 320, dtype=torch.float32).reshape(
        INPUT_SHAPE
    )

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
        detector_logits, descriptor_map = wrapper(image)

    outputs = {
        "detector_logits": detector_logits,
        "descriptor_map": descriptor_map,
    }
    failures = []
    for name, output in outputs.items():
        reference = captured[name]
        max_abs_error = (output - reference).abs().max().item()
        exact = torch.equal(output, reference)
        print(
            "{}: shape={} dtype={} exact={} max_abs_error={:.9g}".format(
                name, list(output.shape), output.dtype, exact, max_abs_error
            )
        )
        if not exact:
            failures.append(name)

    implementation_module = inspect.getmodule(baseline.net.__class__)
    hloc_module = inspect.getmodule(baseline.__class__)
    checkpoint_sha256 = sha256(CHECKPOINT_PATH.read_bytes()).hexdigest()
    print("HLOC wrapper: {}".format(Path(hloc_module.__file__).resolve()))
    print("Implementation: {}".format(Path(implementation_module.__file__).resolve()))
    print("Checkpoint: {}".format(CHECKPOINT_PATH.resolve()))
    print("Checkpoint SHA256: {}".format(checkpoint_sha256))
    print("Config: {}".format(BASELINE_CONFIG))
    print("Input: shape={} dtype={} range=[{:.1f}, {:.1f}]".format(
        list(image.shape), image.dtype, image.min().item(), image.max().item()
    ))

    if failures:
        raise RuntimeError("Raw-output parity failed: {}".format(", ".join(failures)))
    print("Raw-output parity: PASS")


if __name__ == "__main__":
    main()
