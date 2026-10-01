from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import traceback

import numpy as np

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()

def put(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    onnx_path = Path(args.onnx)
    calibration_path = Path(args.calibration)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    result = {
        key: "NOT_RUN"
        for key in ("import", "ptq", "compile", "gencode", "sim")
    }
    stage = "import"
    try:
        import nncase
        import _nncase
        from importlib.metadata import version

        if version("nncase") != "1.8.0.20220929":
            raise RuntimeError("unexpected nncase version")
        if _nncase.__version__ != "1.8.0-55be52f":
            raise RuntimeError("unexpected nncase runtime version")
        result["environment"] = {
            "nncase": version("nncase"),
            "runtime": _nncase.__version__,
        }

        options = nncase.CompileOptions()
        options.target = "k210"
        options.quant_type = "uint8"
        options.w_quant_type = "uint8"
        options.dump_asm = True
        options.dump_dir = str(output / "dump")

        compiler = nncase.Compiler(options)
        compiler.import_onnx(
            onnx_path.read_bytes(), nncase.ImportOptions()
        )
        result["import"] = "PASS"

        stage = "ptq"
        calibration = np.load(calibration_path)
        if calibration.shape[1:] != (3, 240, 240):
            raise RuntimeError(
                f"bad calibration shape: {calibration.shape}"
            )
        ptq = nncase.PTQTensorOptions()
        ptq.samples_count = len(calibration)
        ptq.set_tensor_data(
            np.ascontiguousarray(calibration).tobytes()
        )
        compiler.use_ptq(ptq)
        result["ptq"] = "PASS"

        stage = "compile"
        compiler.compile()
        result["compile"] = "PASS"

        stage = "gencode"
        blob = compiler.gencode_tobytes()
        if not blob:
            raise RuntimeError("empty kmodel")
        kmodel = output / "model.kmodel"
        kmodel.write_bytes(blob)
        result["gencode"] = "PASS"
        result["model_bytes"] = len(blob)
        result["kmodel_sha256"] = sha256(kmodel)

        info = (output / "dump/kmodel_info.txt").read_text()
        result["compiler_memory"] = {
            key: int(value)
            for key, value in re.findall(
                r"(input|output|data|MODEL|TOTAL):?\s+[^\n]*?\((\d+) B\)",
                info,
            )
        }
        if "TOTAL" not in result["compiler_memory"]:
            raise RuntimeError("compiler TOTAL memory missing")

        mapping = {}
        for backend in ("stackvm", "k210"):
            files = list(
                (output / "dump" / backend).rglob("runtime_ops.txt")
            )
            content = "\n".join(
                f.read_text() for f in files
            )
            mapping[backend] = dict(Counter(
                re.findall(r"\[([A-Za-z0-9_]+)\]", content)
            ))
        result["mapping"] = mapping
        result["kpu_conv"] = mapping["k210"].get("KPUConv2D", 0)
        result["cpu_conv"] = mapping["stackvm"].get("Conv2D", 0)

        stage = "sim"
        simulator = nncase.Simulator()
        simulator.load_model(blob)
        outputs = []
        indices = [0, 24, 56]
        for index in indices:
            x = np.ascontiguousarray(
                calibration[index:index + 1]
            )
            simulator.set_input_tensor(
                0, nncase.RuntimeTensor.from_numpy(x)
            )
            simulator.run()
            y = simulator.get_output_tensor(0).to_numpy().copy()
            if y.shape != (1, 512):
                raise RuntimeError(
                    f"unexpected output shape: {y.shape}"
                )
            if not np.isfinite(y).all():
                raise RuntimeError("non-finite simulator output")
            outputs.append(y)
        np.save(
            output / "sim_outputs.npy",
            np.concatenate(outputs),
        )
        result["sim"] = "PASS"
        result["sim_input_indices"] = indices
        result["sim_output_shape"] = [3, 512]

    except Exception as exc:
        result[stage] = "FAIL"
        result["failed_stage"] = stage
        result["error"] = str(exc)
        result["traceback"] = traceback.format_exc()

    result["graph_pass"] = (
        result.get("gencode") == "PASS"
        and result.get("sim") == "PASS"
        and result.get("kpu_conv") == 20
        and result.get("cpu_conv") == 0
    )
    result["reserve_screen_pass"] = (
        result["graph_pass"]
        and result.get("compiler_memory", {}).get(
            "TOTAL", 10**18
        ) <= 3 * 1024**2
    )
    put(output / "compile.json", result)
    print(json.dumps(result))
    raise SystemExit(0 if result["graph_pass"] else 2)

if __name__ == "__main__":
    main()
