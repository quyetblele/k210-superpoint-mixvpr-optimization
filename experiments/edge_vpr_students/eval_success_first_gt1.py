"""One-shot frozen GT1 evaluation after all DEV-based model selection is final."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/home/quyet/k210_lab")
EDGE = Path("/home/quyet/edge_ai_project")
GT0 = ROOT / "artifacts/edge_vpr_students/gt0"
sys.path.insert(0, str(ROOT / "experiments/edge_vpr_students"))
import success_first_train_core as core


def atomic_npy(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, value)
    temporary.replace(path)


@torch.inference_mode()
def descriptors(model, size: int, paths: list[Path], dim: int, device, output: Path,
                checkpoint_hash: str) -> np.ndarray:
    metadata_path = output.with_suffix(".json")
    ordered = [str(path) for path in paths]
    expected = {"checkpoint_sha256": checkpoint_hash, "ordered_paths": ordered,
                "shape": [len(paths), dim], "input": [size, size],
                "preprocessing": "RGB direct bicubic square resize; /255; ImageNet normalization"}
    if output.exists() and metadata_path.exists() and json.loads(metadata_path.read_text()) == expected:
        cached = np.load(output)
        if cached.shape == (len(paths), dim) and np.isfinite(cached).all() and np.allclose(
            np.linalg.norm(cached, axis=1), 1.0, atol=1e-4
        ):
            return cached
    rows = []
    for start in range(0, len(paths), 16):
        batch = torch.stack([core.image_tensor(str(path), size, None) for path in paths[start:start + 16]]).to(device)
        # The final gate is FP32 inference; casting AMP outputs afterwards
        # would not make that an FP32 evaluation.
        value = model(batch.float()).float()
        rows.append(torch.nn.functional.normalize(value, dim=1).cpu().numpy().astype(np.float32))
    result = np.concatenate(rows)
    if result.shape != (len(paths), dim) or not np.isfinite(result).all():
        raise RuntimeError("invalid GT1 descriptor cache")
    atomic_npy(output, result); core.atomic_json(metadata_path, expected)
    return result


def recall(reference: np.ndarray, query: np.ndarray, refs: list[str], queries: list[str], floor: str) -> dict:
    labels = json.loads((GT0 / f"{floor}_positive_labels.json").read_text())["primary"]
    hits = {1: 0, 5: 0, 10: 0}; valid = 0
    for name, descriptor in zip(queries, query):
        positive = set(labels[name])
        if not positive:
            continue
        valid += 1
        order = np.argsort(-(reference @ descriptor), kind="mergesort")
        for k in hits:
            hits[k] += bool(positive.intersection(refs[index] for index in order[:k]))
    return {"references": len(refs), "queries": len(queries), "valid_queries": valid,
            "zero_positive": len(queries) - valid,
            "recall": {f"R@{k}": hits[k] / valid for k in hits}, "hits": hits}


def evaluate(model, size: int, dim: int, checkpoint: Path, namespace: Path, device) -> dict:
    checkpoint_hash = core.sha256(checkpoint); model = model.to(device).eval()
    floors = {}; combined_hits = {1: 0, 5: 0, 10: 0}; combined_valid = 0
    for floor in ("floor6", "floor7"):
        names = {role: (GT0 / f"{floor}_{role}.txt").read_text().splitlines()
                 for role in ("reference", "query")}
        values = {}
        for role in names:
            paths = [EDGE / "assets" / floor / name for name in names[role]]
            missing = [str(path) for path in paths if not path.is_file()]
            if missing: raise RuntimeError(f"missing GT1 image: {missing[0]}")
            values[role] = descriptors(model, size, paths, dim, device,
                namespace / f"{floor}_{role}.npy", checkpoint_hash)
        result = recall(values["reference"], values["query"], names["reference"], names["query"], floor)
        floors[floor] = result; combined_valid += result["valid_queries"]
        for k in combined_hits: combined_hits[k] += result["hits"][k]
    return {"checkpoint": str(checkpoint), "checkpoint_sha256": checkpoint_hash,
            "floors": floors, "combined": {f"R@{k}": combined_hits[k] / combined_valid for k in combined_hits}}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--student", type=Path, required=True)
    parser.add_argument("--identity", choices=("A_C144_H112", "B_C160_H96"), required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/success_first/final_gt1")
    args = parser.parse_args(); device = core.require_cuda(); args.output.mkdir(parents=True, exist_ok=True)
    teacher_state = torch.load(args.teacher, map_location="cpu", weights_only=False)
    teacher = core.build_teacher(); teacher.load_state_dict(teacher_state["model_state_dict"], strict=True)
    student_state = torch.load(args.student, map_location="cpu", weights_only=False)
    student = core.build_student(args.identity); student.load_state_dict(student_state["model_state_dict"], strict=True)
    teacher_result = evaluate(teacher, 320, 4096, args.teacher, args.output / "strong_teacher", device)
    student_result = evaluate(student, 240, 512, args.student, args.output / "student", device)
    historical = {
        "Official_T0": {"R@1": 0.5080645161, "R@5": 0.6209677419, "R@10": 0.6612903226},
        "S512_B": {"R@1": 0.2177419355, "R@5": 0.3629032258, "R@10": 0.4354838710},
        "BR1_old": {"R@1": 0.2016129032, "R@5": 0.3467741935, "R@10": 0.4112903226},
        "BR2M_110": {"R@1": 0.1370967742, "R@5": 0.2661290323, "R@10": 0.3306451613},
    }
    s = student_result["combined"]; t = teacher_result["combined"]
    report = {"status": "COMPLETE", "protocol": "frozen GT1; never used for training/tuning/selection",
              "teacher": teacher_result, "student": student_result, "historical": historical,
              "student_retention_vs_strong_teacher": {key: s[key] / t[key] for key in s},
              "student_retention_vs_official_t0": {key: s[key] / historical["Official_T0"][key] for key in s}}
    core.atomic_json(args.output / "report.json", report); print(json.dumps(report))


if __name__ == "__main__":
    main()
