from __future__ import annotations
from collections import Counter
from pathlib import Path
import argparse
import hashlib
import json
import re

import numpy as np

def sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()

def put(path: Path, value: dict):
    path.write_text(json.dumps(value, indent=2) + "\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    import nncase
    import _nncase
    from importlib.metadata import version

    onnx_path = Path(args.onnx)
    calibration_path = Path(args.calibration)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dump = out / "dump"

    result = {
        "status": "RUNNING",
        "onnx_sha256": sha256(onnx_path),
        "calibration_sha256": sha256(calibration_path),
        "nncase": version("nncase"),
        "runtime": _nncase.__version__,
        "target": "k210",
    }

    stage = "import"
    try:
        options = nncase.CompileOptions()
        options.target = "k210"
        options.quant_type = "uint8"
        options.w_quant_type = "uint8"
        options.dump_asm = True
        options.dump_dir = str(dump)

        compiler = nncase.Compiler(options)
        compiler.import_onnx(
            onnx_path.read_bytes(), nncase.ImportOptions()
        )
        result["import"] = "PASS"

        stage = "ptq"
        data = np.load(calibration_path)
        ptq = nncase.PTQTensorOptions()
        ptq.samples_count = len(data)
        ptq.set_tensor_data(data.tobytes())
        compiler.use_ptq(ptq)
        result["ptq"] = "PASS"

        stage = "compile"
        compiler.compile()
        result["compile"] = "PASS"

        stage = "gencode"
        blob = compiler.gencode_tobytes()
        kmodel = out / "model.kmodel"
        kmodel.write_bytes(blob)
        result["gencode"] = "PASS"
        result["model_bytes"] = len(blob)
        result["kmodel_sha256"] = sha256(kmodel)

        info = dump / "kmodel_info.txt"
        if info.exists():
            text = info.read_text()
            result["compiler_memory"] = {
                k: int(v)
                for k, v in re.findall(
                    r"(input|output|data|MODEL|TOTAL):?\s+[^\n]*?\((\d+) B\)",
                    text,
                )
            }

        mapping = {}
        for backend in ("stackvm", "k210"):
            files = list((dump / backend).rglob("runtime_ops.txt"))
            content = "\n".join(
                f.read_text() for f in files
            )
            mapping[backend] = dict(
                Counter(re.findall(r"\[([A-Za-z0-9_]+)\]", content))
            )
        result["mapping"] = mapping
        result["kpu_conv"] = mapping.get("k210", {}).get("KPUConv2D", 0)
        result["cpu_conv"] = mapping.get("stackvm", {}).get("Conv2D", 0)

        stage = "sim"
        sim = nncase.Simulator()
        sim.load_model(blob)
        outputs = []
        for index in (0, 24, 56):
            x = np.ascontiguousarray(data[index:index + 1])
            sim.set_input_tensor(
                0, nncase.RuntimeTensor.from_numpy(x)
            )
            sim.run()
            y = sim.get_output_tensor(0).to_numpy().copy()
            if y.shape != (1, 512):
                raise RuntimeError(f"unexpected output shape: {y.shape}")
            if not np.isfinite(y).all():
                raise RuntimeError("non-finite simulator output")
            outputs.append(y)
        np.save(out / "sim_outputs.npy", np.concatenate(outputs))
        result["sim"] = "PASS"
        result["status"] = "PASS"

    except Exception as exc:
        result["status"] = "FAIL"
        result["failed_stage"] = stage
        result["error"] = str(exc)

    put(out / "compile.json", result)
    print(json.dumps(result, indent=2))
    if result["status"] != "PASS":
        raise SystemExit(1)

if __name__ == "__main__":
    main()
