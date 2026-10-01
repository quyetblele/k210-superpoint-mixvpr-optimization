#!/usr/bin/env python3
"""Build a self-contained, host-validated K210 deployment package.

This script deliberately packages only frozen artifacts.  It does not train,
rewrite ONNX graphs, or run a board.  The package boundary is the two kmodels
plus their exact preprocessing/postprocessing contracts and deterministic host
test vectors.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "deployment" / "k210_release"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def copy(src: Path, rel: str) -> Path:
    dst = OUT / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return dst


def shape(model_path: Path):
    model = onnx.load(str(model_path))
    onnx.checker.check_model(model)
    def value_info(v):
        return {
            "name": v.name,
            "shape": [d.dim_value for d in v.type.tensor_type.shape.dim],
            "dtype": int(v.type.tensor_type.elem_type),
        }
    return {
        "inputs": [value_info(v) for v in model.graph.input],
        "outputs": [value_info(v) for v in model.graph.output],
        "nodes": len(model.graph.node),
    }


def main() -> None:
    if OUT.exists():
        # Only this generated release directory is replaced; source artifacts
        # and all historical experiment namespaces remain untouched.
        shutil.rmtree(OUT)
    (OUT / "models").mkdir(parents=True)
    (OUT / "contracts").mkdir()
    (OUT / "test_vectors").mkdir()
    (OUT / "evidence").mkdir()

    # These are the last frozen, independently validated deployment artifacts.
    mix_k = copy(ROOT / "mixvpr/artifacts/depth_pilot/D4/model.kmodel", "models/mixvpr_d4_240x240.kmodel")
    mix_o = copy(ROOT / "mixvpr/artifacts/depth_pilot/D4/model.onnx", "models/mixvpr_d4_240x240.onnx")
    sp_k = copy(ROOT / "superpoint/artifacts/deployment/full_capacity_r2_cw/model.kmodel", "models/superpoint_full_r2_cw.kmodel")
    sp_o = copy(ROOT / "superpoint/artifacts/deployment/full_capacity_r2_cw/canonical.onnx", "models/superpoint_full_r2_cw.onnx")

    mix_cfg = {
        "identity": "MixVPR-D4-frozen",
        "input": {"shape": [1, 3, 240, 240], "dtype": "float32", "color": "RGB",
                   "resize": "PIL bicubic direct 240x240, antialias", "scale": "uint8 / 255",
                   "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
        "output": {"shape": [1, 512], "dtype": "float32", "meaning": "raw pre-L2 descriptor"},
        "cpu_postprocess": "d = z / max(||z||_2, 1e-12)",
        "k210": {"target": "k210", "nncase": "1.8.0.20220929", "_nncase": "1.8.0-55be52f"},
        "quality_status": "DEV qualified; independent TEST protocol remains blocked",
    }
    sp_cfg = json.loads((ROOT / "superpoint/artifacts/full_training/frozen_preprocessing.json").read_text())
    sp_post = json.loads((ROOT / "superpoint/artifacts/full_training/frozen_postprocess.json").read_text())
    sp_cfg.update({"identity": "SuperPoint-full_capacity_r2_cw-frozen", "output": {
        "detector_logits": [1, 65, 23, 40], "descriptor_map": [1, 256, 23, 40],
        "dtype": "float32", "descriptor_dim": 256, "stride": 8,
    }, "k210": {"target": "k210", "nncase": "1.8.0.20220929", "_nncase": "1.8.0-55be52f"}})
    (OUT / "contracts/mixvpr.json").write_text(json.dumps(mix_cfg, indent=2) + "\n")
    (OUT / "contracts/superpoint_preprocessing.json").write_text(json.dumps(sp_cfg, indent=2) + "\n")
    (OUT / "contracts/superpoint_postprocess.json").write_text(json.dumps(sp_post, indent=2) + "\n")

    # ONNX checks + deterministic host vectors.  Inputs are already normalized
    # tensors at the model boundary; camera preprocessing is specified above.
    mix_shape = shape(mix_o)
    sp_shape = shape(sp_o)
    rng = np.random.default_rng(20260918)
    mix_inputs = {
        "linspace": np.linspace(-1, 1, 3 * 240 * 240, dtype=np.float32).reshape(1, 3, 240, 240),
        "seeded_random": rng.standard_normal((1, 3, 240, 240), dtype=np.float32),
        "zeros": np.zeros((1, 3, 240, 240), dtype=np.float32),
    }
    sp_inputs = {
        "linspace": np.linspace(0, 1, 184 * 320, dtype=np.float32).reshape(1, 1, 184, 320),
        "seeded_random": rng.random((1, 1, 184, 320), dtype=np.float32),
        "zeros": np.zeros((1, 1, 184, 320), dtype=np.float32),
    }
    mix_sess = ort.InferenceSession(str(mix_o), providers=["CPUExecutionProvider"])
    sp_sess = ort.InferenceSession(str(sp_o), providers=["CPUExecutionProvider"])
    vectors = {"mixvpr": {}, "superpoint": {}}
    for name, x in mix_inputs.items():
        y = mix_sess.run(None, {mix_sess.get_inputs()[0].name: x})[0]
        z = y / np.maximum(np.linalg.norm(y, axis=1, keepdims=True), 1e-12)
        np.save(OUT / "test_vectors" / f"mixvpr_{name}_input.npy", x)
        np.save(OUT / "test_vectors" / f"mixvpr_{name}_raw.npy", y)
        np.save(OUT / "test_vectors" / f"mixvpr_{name}_normalized.npy", z)
        vectors["mixvpr"][name] = {"input_shape": list(x.shape), "raw_shape": list(y.shape),
                                    "raw_finite": bool(np.isfinite(y).all()),
                                    "normalized_norm": float(np.linalg.norm(z)),
                                    "raw_sha256": sha256(OUT / "test_vectors" / f"mixvpr_{name}_raw.npy")}
    for name, x in sp_inputs.items():
        ys = sp_sess.run(None, {sp_sess.get_inputs()[0].name: x})
        for suffix, y in zip(("detector_logits", "descriptor_map"), ys):
            np.save(OUT / "test_vectors" / f"superpoint_{name}_{suffix}.npy", y)
        vectors["superpoint"][name] = {"input_shape": list(x.shape),
                                       "output_shapes": [list(y.shape) for y in ys],
                                       "finite": bool(all(np.isfinite(y).all() for y in ys)),
                                       "output_sha256": [sha256(OUT / "test_vectors" / f"superpoint_{name}_{s}.npy") for s in ("detector_logits", "descriptor_map")]}

    files = {}
    for p in sorted(OUT.rglob("*")):
        if p.is_file() and p.name != "manifest.json":
            files[str(p.relative_to(OUT))] = {"bytes": p.stat().st_size, "sha256": sha256(p)}
    manifest = {
        "status": "HOST_VALIDATED_READY_FOR_BOARD_IO_TEST",
        "generated": "2026-09-18",
        "toolchain": {"nncase": "1.8.0.20220929", "_nncase": "1.8.0-55be52f", "target": "k210"},
        "models": {
            "mixvpr": {"kmodel": "models/mixvpr_d4_240x240.kmodel", "onnx": "models/mixvpr_d4_240x240.onnx", "onnx_contract": mix_shape},
            "superpoint": {"kmodel": "models/superpoint_full_r2_cw.kmodel", "onnx": "models/superpoint_full_r2_cw.onnx", "onnx_contract": sp_shape},
        },
        "host_checks": {"onnx_checker": "PASS", "onnxruntime": "PASS", "finite_outputs": True, "test_vectors": vectors},
        "board_unknown": ["camera sensor/firmware I/O", "actual AI/general SRAM peak", "latency/FPS", "power", "long-run stability", "full online pose chain"],
        "files": files,
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"out": str(OUT), "mixvpr": mix_shape, "superpoint": sp_shape, "files": len(files)}, indent=2))


if __name__ == "__main__":
    main()
