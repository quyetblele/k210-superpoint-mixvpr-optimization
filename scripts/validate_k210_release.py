#!/usr/bin/env python3
"""Revalidate a generated deployment/k210_release package without a board."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort


ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "deployment" / "k210_release"


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def main() -> None:
    m = json.loads((PKG / "manifest.json").read_text())
    rows = []
    for rel, expected in m["files"].items():
        p = PKG / rel
        ok = p.is_file() and p.stat().st_size == expected["bytes"] and sha256(p) == expected["sha256"]
        rows.append({"file": rel, "sha256": ok})
        if not ok:
            raise SystemExit(f"HASH_FAIL {rel}")
    for key in ("mixvpr", "superpoint"):
        onnx_rel = m["models"][key]["onnx"]
        model = onnx.load(str(PKG / onnx_rel))
        onnx.checker.check_model(model)
        sess = ort.InferenceSession(str(PKG / onnx_rel), providers=["CPUExecutionProvider"])
        for name, spec in m["host_checks"]["test_vectors"][key].items():
            x = np.load(PKG / "test_vectors" / f"{key}_{name}_input.npy")
            out = sess.run(None, {sess.get_inputs()[0].name: x})
            if key == "mixvpr":
                assert out[0].shape == (1, 512) and np.isfinite(out[0]).all()
                z = out[0] / max(float(np.linalg.norm(out[0])), 1e-12)
                assert abs(float(np.linalg.norm(z)) - 1.0) < 1e-5
            else:
                assert [list(y.shape) for y in out] == [[1, 65, 23, 40], [1, 256, 23, 40]]
                assert all(np.isfinite(y).all() for y in out)
    result = {"status": "PASS", "files_checked": len(rows), "onnx_checker": "PASS", "onnxruntime": "PASS", "board": "NOT_RUN"}
    (PKG / "host_validation.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
