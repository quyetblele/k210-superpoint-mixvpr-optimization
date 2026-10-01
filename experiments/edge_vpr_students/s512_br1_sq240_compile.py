"""Torch-free pinned-nncase compiler gate for BR1-SQ240."""
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
source = ROOT / "models" / "s512_br1_sq240_prenorm_canonical.onnx"
artifact = ROOT / "artifacts" / "edge_vpr_students" / "s512_br1_sq240" / "s512_br1_sq240_prenorm.kmodel"
report_path = ROOT / "reports" / "edge_vpr_students" / "s512_br1_sq240_compiler.json"
result = {stage: "NOT_RUN" for stage in ("checker", "import", "ptq", "compile", "gencode")}
result.update({"nncase": version("nncase"), "_nncase": _nncase.__version__})
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
    calibration = np.random.default_rng(20260906).uniform(-1, 1, (8, 3, 240, 240)).astype(np.float32)
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
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_bytes(payload)
    result.update({"gencode": "PASS", "kmodel": str(artifact), "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()})
except Exception as error:
    result.update({"failed_stage": stage, "error": f"{type(error).__name__}: {error}", "traceback": traceback.format_exc()})
report_path.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result))
