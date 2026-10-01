from __future__ import annotations
from pathlib import Path
import json
import subprocess

import numpy as np
import torch

from .config import spec_for_stage, stage_ids
from .export import export_onnx
from .model import K210MixVPR, count_params, conv_linear_macs

K210_PYTHON = "/home/quyet/miniconda3/envs/k210/bin/python"
NNCASE_SCRIPT = Path(
    "/home/quyet/k210_lab/mixvpr/scripts/v2/nncase_compile.py"
)

def _put(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")

def structural_probe(
    cfg: dict,
    stage_id: str,
    output: Path,
) -> dict:
    """Probe graph feasibility only; no training or retrieval evaluation."""
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(int(cfg["seed"]))
    spec = spec_for_stage(cfg, stage_id)
    model = K210MixVPR(spec).eval()

    calibration_path = Path(
        cfg["data"]["calibration"]
    ).resolve()
    calibration = np.load(calibration_path, mmap_mode="r")
    sample = torch.from_numpy(
        np.array(calibration[:1], dtype=np.float32)
    )

    export = export_onnx(model, output, sample)
    log_path = output / "nncase.log"
    command = [
        K210_PYTHON,
        str(NNCASE_SCRIPT),
        "--onnx", str((output / "model.onnx").resolve()),
        "--calibration", str(calibration_path),
        "--output", str(output),
    ]
    with log_path.open("w") as log:
        process = subprocess.run(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            cwd=output,
            check=False,
        )

    compile_path = output / "compile.json"
    if not compile_path.exists():
        raise RuntimeError(
            "nncase backend did not produce compile.json; "
            f"returncode={process.returncode}"
        )
    compiled = json.loads(compile_path.read_text())
    result = {
        "stage_id": stage_id,
        "scope": "untrained structural feasibility only",
        "spec": {
            "widths": list(spec.widths),
            "mixer_hidden": spec.mixer_hidden,
            "mixer_depth": spec.mixer_depth,
            "descriptor_dim": spec.descriptor_dim,
        },
        "params": count_params(model),
        "conv_linear_macs": conv_linear_macs(model),
        "export": export,
        "nncase": compiled,
        "backend_returncode": process.returncode,
        "quality": "NOT_EVALUATED",
        "test_access": False,
    }
    _put(output / "result.json", result)
    return result

def probe_all(cfg: dict, root: Path) -> list[dict]:
    root = root.resolve()
    rows = []
    for stage_id in stage_ids(cfg):
        print("PROBE", stage_id, flush=True)
        result = structural_probe(
            cfg, stage_id, root / stage_id
        )
        rows.append({
            "stage_id": stage_id,
            "widths": result["spec"]["widths"],
            "hidden": result["spec"]["mixer_hidden"],
            "params": result["params"],
            "macs": result["conv_linear_macs"],
            "gencode": result["nncase"].get("gencode"),
            "sim": result["nncase"].get("sim"),
            "kpu_conv": result["nncase"].get("kpu_conv"),
            "cpu_conv": result["nncase"].get("cpu_conv"),
            "total_bytes": result["nncase"]
                .get("compiler_memory", {})
                .get("TOTAL"),
            "kmodel_bytes": result["nncase"].get("model_bytes"),
            "graph_pass": result["nncase"].get("graph_pass"),
            "reserve_screen_pass": result["nncase"]
                .get("reserve_screen_pass"),
            "error": result["nncase"].get("error"),
        })
        _put(root / "summary.json", {
            "status": "RUNNING",
            "rows": rows,
        })
    _put(root / "summary.json", {
        "status": "COMPLETE",
        "rows": rows,
    })
    return rows
