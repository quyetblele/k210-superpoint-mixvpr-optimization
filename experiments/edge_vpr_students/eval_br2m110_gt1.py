"""Resumable full frozen FP32 GT1 evaluation for official BR2M-110."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/home/quyet/k210_lab")
EXP = ROOT / "experiments" / "edge_vpr_students"
sys.path.insert(0, str(EXP))
import train_s512_b_kd as base
from s512_br2m110_model import BR2M110

OFFICIAL = ROOT / "artifacts" / "edge_vpr_students" / "s512_br2m110_official"
CACHE = OFFICIAL / "gt1_fp32"
GT0 = ROOT / "artifacts" / "edge_vpr_students" / "gt0"
REPORT = ROOT / "reports" / "edge_vpr_students"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def atomic_npy(path: Path, value: np.ndarray) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, value)
    temporary.replace(path)


def extract_role(model, checkpoint_hash: str, floor: str, role: str):
    names = (GT0 / f"{floor}_{role}.txt").read_text().splitlines()
    role_dir = CACHE / f"{floor}_{role}"
    role_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = role_dir / "manifest.json"
    manifest = {
        "checkpoint_sha256": checkpoint_hash,
        "floor": floor,
        "role": role,
        "ordered_image_ids": names,
        "shape": [len(names), 512],
        "preprocessing": "native raw BGR -> RGB -> direct INTER_AREA W192xH352 -> float32/255 -> ImageNet normalize -> CHW",
        "chunk_size": 32,
    }
    if manifest_path.exists():
        assert json.loads(manifest_path.read_text()) == manifest
    else:
        atomic_json(manifest_path, manifest)
    chunks = []
    for chunk_index, start in enumerate(range(0, len(names), 32)):
        end = min(start + 32, len(names))
        chunk_path = role_dir / f"chunk_{chunk_index:03d}.npz"
        valid = False
        if chunk_path.exists():
            loaded = np.load(chunk_path, allow_pickle=False)
            values = loaded["descriptor"]
            valid = (
                loaded["image_ids"].tolist() == names[start:end]
                and loaded["checkpoint_sha256"].item() == checkpoint_hash
                and values.shape == (end - start, 512)
                and np.isfinite(values).all()
                and np.allclose(np.linalg.norm(values, axis=1), 1.0, atol=1e-4)
            )
        if not valid:
            rows = [(floor, name, None, None) for name in names[start:end]]
            with torch.inference_mode():
                values = model(
                    torch.stack([base.student_tensor(row, 192, 352) for row in rows])
                ).cpu().numpy().astype(np.float32)
            assert values.shape == (end - start, 512)
            assert np.isfinite(values).all()
            assert np.allclose(np.linalg.norm(values, axis=1), 1.0, atol=1e-4)
            temporary = chunk_path.with_suffix(".npz.tmp")
            with temporary.open("wb") as handle:
                np.savez_compressed(
                    handle,
                    descriptor=values,
                    image_ids=np.asarray(names[start:end]),
                    checkpoint_sha256=np.asarray(checkpoint_hash),
                )
            temporary.replace(chunk_path)
        loaded = np.load(chunk_path, allow_pickle=False)
        chunks.append(loaded["descriptor"].copy())
    descriptors = np.concatenate(chunks, axis=0)
    assert descriptors.shape == (len(names), 512)
    assert len(names) == len(set(names))
    assert np.isfinite(descriptors).all()
    norms = np.linalg.norm(descriptors, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-4)
    final_path = CACHE / f"{floor}_{role}_fp32.npy"
    atomic_npy(final_path, descriptors)
    return names, descriptors, {
        "count": len(names),
        "shape": list(descriptors.shape),
        "finite": True,
        "unique_image_ids": len(set(names)),
        "norm_min": float(norms.min()),
        "norm_max": float(norms.max()),
        "sha256": sha256(final_path),
    }


def recall(floor: str, reference_names, reference, query_names, query):
    labels = json.loads((GT0 / f"{floor}_positive_labels.json").read_text())["primary"]
    outcomes = []
    for name, descriptor in zip(query_names, query):
        positive = set(labels[name])
        if not positive:
            outcomes.append(None)
            continue
        ranking = np.argsort(-(reference @ descriptor), kind="mergesort")
        outcomes.append(
            {k: bool(positive.intersection(reference_names[index] for index in ranking[:k])) for k in (1, 5, 10)}
        )
    valid = [row for row in outcomes if row is not None]
    return {
        "reference_count": len(reference_names),
        "query_count": len(query_names),
        "zero_positive": len(outcomes) - len(valid),
        "valid_queries": len(valid),
        "recall": {f"R@{k}": float(np.mean([row[k] for row in valid])) for k in (1, 5, 10)},
    }


def main() -> None:
    torch.set_num_threads(4)
    checkpoint = OFFICIAL / "final.pt"
    state = torch.load(checkpoint, weights_only=False)
    assert (state["current_epoch"], state["next_batch_index"], state["global_step"]) == (5, 0, 192)
    model = BR2M110().eval()
    model.load_state_dict(state["model_state_dict"])
    checkpoint_hash = sha256(checkpoint)
    cache_audit = {}
    results = {}
    total_hits = {1: 0, 5: 0, 10: 0}
    total_valid = 0
    for floor in ("floor6", "floor7"):
        reference_names, reference, reference_audit = extract_role(model, checkpoint_hash, floor, "reference")
        query_names, query, query_audit = extract_role(model, checkpoint_hash, floor, "query")
        cache_audit[floor] = {"reference": reference_audit, "query": query_audit}
        result = recall(floor, reference_names, reference, query_names, query)
        results[floor] = result
        total_valid += result["valid_queries"]
        for k in (1, 5, 10):
            total_hits[k] += round(result["recall"][f"R@{k}"] * result["valid_queries"])
    combined = {f"R@{k}": total_hits[k] / total_valid for k in (1, 5, 10)}
    t0 = {"R@1": 0.5080645161, "R@5": 0.6209677419, "R@10": 0.6612903226}
    b = {"R@1": 0.2177419355, "R@5": 0.3629032258, "R@10": 0.4354838710}
    br1_data = json.loads((REPORT / "s512_br1_gt1_results.json").read_text())
    br1 = br1_data["combined"]
    comparison = {
        "T0": t0,
        "S512-B": b,
        "S512-BR1": br1,
        "S512-BR2M-110": combined,
    }
    output = {
        "status": "PASS",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_hash,
        "protocol": "frozen GT1: center distance <=1.0m AND view angle <=60deg",
        "cache_audit": cache_audit,
        "floor6": results["floor6"],
        "floor7": results["floor7"],
        "combined": combined,
        "comparison": comparison,
        "retention_vs_t0": {key: combined[key] / t0[key] for key in t0},
        "absolute_delta_vs_t0": {key: combined[key] - t0[key] for key in t0},
        "absolute_gain_vs_br1": {key: combined[key] - br1[key] for key in br1},
        "relative_gain_vs_br1": {
            key: (combined[key] - br1[key]) / br1[key] for key in br1
        },
    }
    atomic_json(REPORT / "s512_br2m110_gt1_results.json", output)
    print(json.dumps(output))


if __name__ == "__main__":
    main()
