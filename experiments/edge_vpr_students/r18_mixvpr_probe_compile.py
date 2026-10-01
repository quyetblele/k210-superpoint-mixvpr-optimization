"""Torch-free pinned-nncase compiler gate for one frozen R18 probe."""
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
MODEL_DIR = ROOT / "models/r18_mixvpr_probes"
REPORT_DIR = ROOT / "reports/edge_vpr_students/r18_mixvpr_probes"
ARTIFACT_DIR = ROOT / "artifacts/edge_vpr_students/r18_mixvpr_probes"
SIZES = {"r18_196": 224, "r18_169": 208, "r18_144": 192,
         "r18_144_h96": 192, "r18_144_h112": 192}


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--name", choices=tuple(SIZES), required=True)
    args = parser.parse_args(); size = SIZES[args.name]
    source = MODEL_DIR / f"{args.name}_canonical.onnx"
    REPORT_DIR.mkdir(parents=True, exist_ok=True); ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
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
        calibration = np.random.default_rng(20260907).uniform(-1, 1, (8, 3, size, size)).astype(np.float32)
        ptq = nncase.PTQTensorOptions(); ptq.samples_count = 8; ptq.set_tensor_data(calibration.tobytes())
        compiler.use_ptq(ptq); result[stage] = "PASS"
        stage = "compile"; compiler.compile(); result[stage] = "PASS"
        stage = "gencode"; payload = compiler.gencode_tobytes(); result[stage] = "PASS"
        output = ARTIFACT_DIR / f"{args.name}.kmodel"; output.write_bytes(payload)
        result.update({"kmodel": str(output), "kmodel_bytes": len(payload),
                       "kmodel_sha256": hashlib.sha256(payload).hexdigest()})
    except Exception as error:
        result[stage] = "FAIL"
        result.update({"failed_stage": stage, "error": f"{type(error).__name__}: {error}",
                       "traceback": traceback.format_exc()})
    (REPORT_DIR / f"{args.name}_compiler.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__": main()
