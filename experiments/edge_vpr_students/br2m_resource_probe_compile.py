"""Torch-free nncase compiler runner for one canonical BR2M probe."""
from __future__ import annotations

import argparse
import hashlib
import json
import traceback
from pathlib import Path

import _nncase
import nncase
import numpy as np
import onnx
from importlib.metadata import version

ROOT = Path("/home/quyet/k210_lab")
REPORT_DIR = ROOT / "reports" / "edge_vpr_students" / "br2m_resource_probes"
ARTIFACT_DIR = ROOT / "artifacts" / "edge_vpr_students" / "s512_br2m_resource_probes"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    args = parser.parse_args()
    source = ROOT / "models" / "br2m_resource_probes" / f"{args.name}_canonical.onnx"
    report_path = REPORT_DIR / f"{args.name}_compiler.json"
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    result = {
        "name": args.name,
        "nncase": version("nncase"),
        "_nncase": _nncase.__version__,
        "checker": "NOT_RUN",
        "import": "NOT_RUN",
        "ptq": "NOT_RUN",
        "compile": "NOT_RUN",
        "gencode": "NOT_RUN",
        "compiler_reported_resources": None,
    }
    stage = "checker"
    try:
        onnx.checker.check_model(onnx.load(source))
        result["checker"] = "PASS"
        options = nncase.CompileOptions()
        options.target = "k210"
        options.quant_type = "uint8"
        options.w_quant_type = "uint8"
        compiler = nncase.Compiler(options)
        stage = "import"
        compiler.import_onnx(source.read_bytes(), nncase.ImportOptions())
        result["import"] = "PASS"
        stage = "ptq"
        calibration = np.random.default_rng(20260906).uniform(
            -1, 1, (8, 3, 352, 192)
        ).astype(np.float32)
        ptq = nncase.PTQTensorOptions()
        ptq.samples_count = 8
        ptq.set_tensor_data(calibration.tobytes())
        compiler.use_ptq(ptq)
        result["ptq"] = "PASS"
        stage = "compile"
        compiler.compile()
        result["compile"] = "PASS"
        stage = "gencode"
        payload = compiler.gencode_tobytes()
        destination = ARTIFACT_DIR / f"{args.name}.kmodel"
        destination.write_bytes(payload)
        result.update(
            {
                "gencode": "PASS",
                "kmodel": str(destination),
                "kmodel_bytes": len(payload),
                "kmodel_sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    except Exception as error:
        result["failed_stage"] = stage
        result["error"] = f"{type(error).__name__}: {error}"
        result["traceback"] = traceback.format_exc()
    report_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
