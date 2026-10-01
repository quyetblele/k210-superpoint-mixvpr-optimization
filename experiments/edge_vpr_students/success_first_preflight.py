"""Fail-closed preflight for the success-first teacher/student pipeline."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

ROOT = Path("/home/quyet/k210_lab")
REPORT = ROOT / "reports" / "success_first"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA_REQUIRED: GPU device is not exposed; CPU fallback is forbidden")
    dataset = json.loads((REPORT / "dataset_manifest.json").read_text())
    frozen_path = ROOT / "artifacts" / "edge_vpr_students" / "s512_br1_sq240_m100_c160h96" / "frozen_config.json"
    frozen = json.loads(frozen_path.read_text())
    teacher = Path("/home/quyet/.cache/torch/hub/checkpoints/model.ckpt")
    checks = {
        "cuda": True,
        "gpu": torch.cuda.get_device_name(0),
        "teacher_sha256": sha256(teacher),
        "teacher_identity_ok": sha256(teacher) == "97528606773e9920e93ca4d211daeabf1c7312480f38b45379ee5551a1b4dd24",
        "student_frozen": frozen["status"] == "FROZEN_COMPILER_FEASIBLE",
        "student_gencode": frozen["pipeline"]["gencode"] == "PASS",
        "datasets_ready": dataset["ready_for_success_first_training"],
        "gt1_content_overlap": dataset["project_target"]["gt1_exclusion"]["content_overlap"],
    }
    checks["pass"] = all([
        checks["teacher_identity_ok"], checks["student_frozen"], checks["student_gencode"],
        checks["datasets_ready"], checks["gt1_content_overlap"] == 0,
    ])
    print(json.dumps(checks, indent=2))
    if not checks["pass"]:
        raise RuntimeError("SUCCESS_FIRST_PREFLIGHT_BLOCKED")


if __name__ == "__main__":
    main()
