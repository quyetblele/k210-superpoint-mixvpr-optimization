"""Torch-free pinned-nncase compiler runner for a static SQ240 probe."""
from __future__ import annotations

import argparse
import hashlib
import json
import traceback
from importlib.metadata import version
from pathlib import Path

import _nncase
import nncase
import numpy as np
import onnx

ROOT = Path("/home/quyet/k210_lab")
MODEL_DIR = ROOT / "models" / "sq240_resource_probes"
REPORT_DIR = ROOT / "reports" / "edge_vpr_students" / "sq240_resource_probes"
ARTIFACT_DIR = ROOT / "artifacts" / "edge_vpr_students" / "sq240_resource_probes"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    args = parser.parse_args()
    source = MODEL_DIR / f"{args.name}_canonical.onnx"
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    result = {"name": args.name, "nncase": version("nncase"), "_nncase": _nncase.__version__,
              **{stage: "NOT_RUN" for stage in ("checker", "import", "ptq", "compile", "gencode")}}
    stage = "checker"
    try:
        onnx.checker.check_model(onnx.load(source)); result[stage] = "PASS"
        options = nncase.CompileOptions(); options.target = "k210"
        options.quant_type = "uint8"; options.w_quant_type = "uint8"
        compiler = nncase.Compiler(options)
        stage = "import"; compiler.import_onnx(source.read_bytes(), nncase.ImportOptions()); result[stage] = "PASS"
        stage = "ptq"
        calibration = np.random.default_rng(20260907).uniform(-1, 1, (8, 3, 240, 240)).astype(np.float32)
        ptq = nncase.PTQTensorOptions(); ptq.samples_count = 8; ptq.set_tensor_data(calibration.tobytes())
        compiler.use_ptq(ptq); result[stage] = "PASS"
        stage = "compile"; compiler.compile(); result[stage] = "PASS"
        stage = "gencode"; payload = compiler.gencode_tobytes(); result[stage] = "PASS"
        destination = ARTIFACT_DIR / f"{args.name}.kmodel"; destination.write_bytes(payload)
        result.update({"kmodel": str(destination), "kmodel_bytes": len(payload),
                       "kmodel_sha256": hashlib.sha256(payload).hexdigest()})
    except Exception as error:
        result[stage] = "FAIL"
        result.update({"failed_stage": stage, "error": f"{type(error).__name__}: {error}",
                       "traceback": traceback.format_exc()})
    (REPORT_DIR / f"{args.name}_compiler.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
