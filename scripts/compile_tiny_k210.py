#!/usr/bin/env python3
"""Gate 2: compile the static tiny ONNX smoke-test model for Kendryte K210.

The synthetic calibration set is deterministic and only proves the PC compiler
path. A real model must be calibrated with representative deployment data.
"""

from hashlib import sha256
from importlib.metadata import version
from pathlib import Path
import sys
import traceback

import _nncase
import nncase
import numpy as np
import onnx


REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = REPO_ROOT / "models" / "tiny_fp32_inferred.onnx"
KMODEL_PATH = REPO_ROOT / "artifacts" / "tiny.kmodel"
DUMP_DIR = REPO_ROOT / "reports" / "gate2_nncase"
SUMMARY_PATH = REPO_ROOT / "reports" / "gate2_summary.md"
CALIBRATION_SAMPLES = 10
CALIBRATION_SHAPE = (CALIBRATION_SAMPLES, 3, 32, 32)
CALIBRATION_RANGE = "[-1.0, 1.0]"
CALIBRATION_SEED = 20260905
REQUIRED_VALUE_INFO = ("conv1_out", "relu1_out", "conv2_out")


def serialized_value_info():
    """Return stored intermediate metadata required by the nncase importer."""
    model = onnx.load(MODEL_PATH)
    onnx.checker.check_model(model)
    entries = {entry.name: entry for entry in model.graph.value_info}
    missing = [name for name in REQUIRED_VALUE_INFO if name not in entries]
    if missing:
        raise RuntimeError("Serialized value_info missing: {}".format(", ".join(missing)))
    details = []
    for name in REQUIRED_VALUE_INFO:
        tensor_type = entries[name].type.tensor_type
        shape = [dim.dim_value for dim in tensor_type.shape.dim]
        dtype = onnx.TensorProto.DataType.Name(tensor_type.elem_type)
        details.append("{}: {} {}".format(name, dtype, shape))
    return details


def write_summary(statuses, warnings, value_info_details, kmodel_size=None,
                  kmodel_sha256=None):
    """Write a report on both success and a compiler-stage failure."""
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Gate 2 nncase compiler summary",
        "",
        "## Environment",
        "",
        "- Python: {}".format(sys.version.split()[0]),
        "- nncase: {}".format(version("nncase")),
        "- _nncase: {}".format(_nncase.__version__),
        "- ONNX: {}".format(onnx.__version__),
        "- NumPy: {}".format(np.__version__),
        "- Target: k210",
        "",
        "## Shape-inference remediation",
        "",
        "- Original import failure: `Can't find value info for conv1_out to parse its shape`",
        "- Source model: `models/tiny_fp32.onnx` (its serialized `graph.value_info` was empty)",
        "- Remediated model: `models/tiny_fp32_inferred.onnx`, produced by ONNX shape inference and checker validation",
        "- Serialized intermediate value_info:",
    ]
    lines.extend("  - {}".format(detail) for detail in value_info_details)
    lines.extend([
        "",
        "## Configuration",
        "",
        "- Quantization: uint8 activations and uint8 weights (nncase v1 PTQ)",
        "- Calibration: {} FP32 samples, shape {}, range {}, fixed seed {}".format(
            CALIBRATION_SAMPLES, CALIBRATION_SHAPE, CALIBRATION_RANGE,
            CALIBRATION_SEED,
        ),
        "- Debug dumps: IR, ASM, and quantization error under `reports/gate2_nncase/`",
        "",
        "## Results",
        "",
    ])
    for stage in ("ONNX import", "PTQ", "Compile", "Kmodel generation"):
        lines.append("- {}: {}".format(stage, statuses[stage]))
    if kmodel_size is not None:
        lines.append("- Kmodel size: {} bytes".format(kmodel_size))
    if kmodel_sha256 is not None:
        lines.append("- Kmodel SHA256: `{}`".format(kmodel_sha256))
    lines.extend(["", "## Warnings / errors", ""])
    if warnings:
        lines.extend("- {}".format(warning) for warning in warnings)
    else:
        lines.append("- None.")
    SUMMARY_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    statuses = {
        "ONNX import": "NOT RUN",
        "PTQ": "NOT RUN",
        "Compile": "NOT RUN",
        "Kmodel generation": "NOT RUN",
    }
    warnings = [
        "Synthetic calibration data is only for this compiler smoke test; "
        "real deployment requires representative calibration data."
    ]
    kmodel_size = None
    kmodel_sha256 = None
    value_info_details = []

    try:
        if not MODEL_PATH.is_file():
            raise FileNotFoundError("Input model not found: {}".format(MODEL_PATH))
        value_info_details = serialized_value_info()

        DUMP_DIR.mkdir(parents=True, exist_ok=True)
        KMODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        options = nncase.CompileOptions()
        options.target = "k210"
        options.quant_type = "uint8"
        options.w_quant_type = "uint8"
        options.dump_dir = str(DUMP_DIR)
        options.dump_ir = True
        options.dump_asm = True
        options.dump_quant_error = True
        compiler = nncase.Compiler(options)

        compiler.import_onnx(MODEL_PATH.read_bytes(), nncase.ImportOptions())
        statuses["ONNX import"] = "PASS"

        calibration = np.random.default_rng(CALIBRATION_SEED).uniform(
            -1.0, 1.0, size=CALIBRATION_SHAPE
        ).astype(np.float32)
        ptq_options = nncase.PTQTensorOptions()
        ptq_options.samples_count = CALIBRATION_SAMPLES
        ptq_options.set_tensor_data(calibration.tobytes(order="C"))
        compiler.use_ptq(ptq_options)
        statuses["PTQ"] = "PASS"

        compiler.compile()
        statuses["Compile"] = "PASS"

        kmodel_bytes = compiler.gencode_tobytes()
        KMODEL_PATH.write_bytes(kmodel_bytes)
        kmodel_size = KMODEL_PATH.stat().st_size
        if kmodel_size == 0:
            raise RuntimeError("nncase generated an empty kmodel")
        kmodel_sha256 = sha256(kmodel_bytes).hexdigest()
        statuses["Kmodel generation"] = "PASS"
    except Exception as error:
        failed_stage = next(
            stage for stage, status in statuses.items() if status == "NOT RUN"
        )
        statuses[failed_stage] = "FAIL: {}".format(error)
        warnings.append("{}\n{}".format(error, traceback.format_exc()))
        write_summary(statuses, warnings, value_info_details, kmodel_size, kmodel_sha256)
        print("Gate 2 failed at {}: {}".format(failed_stage, error), file=sys.stderr)
        return 1

    write_summary(statuses, warnings, value_info_details, kmodel_size, kmodel_sha256)
    print("ONNX import: PASS")
    print("PTQ: PASS")
    print("Compile: PASS")
    print("Kmodel generation: PASS")
    print("Kmodel: {} ({} bytes)".format(KMODEL_PATH, kmodel_size))
    print("SHA256: {}".format(kmodel_sha256))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
