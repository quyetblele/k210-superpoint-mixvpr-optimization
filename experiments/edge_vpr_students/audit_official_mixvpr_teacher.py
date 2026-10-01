"""Audit official MixVPR against the project's current runtime T0.

This is a baseline/provenance audit only.  GT1 is not used for model selection,
fine-tuning, early stopping, or hyperparameter choices.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T

ROOT = Path("/home/quyet/k210_lab")
EDGE = Path("/home/quyet/edge_ai_project")
HUB = Path("/home/quyet/.cache/torch/hub/jarvisyjw_MixVPR_main")
CHECKPOINT = Path("/home/quyet/.cache/torch/hub/checkpoints/model.ckpt")
GT0 = ROOT / "artifacts" / "edge_vpr_students" / "gt0"
RUNTIME_T0 = ROOT / "artifacts" / "edge_vpr_students" / "gt1_t0"
CACHE = ROOT / "artifacts" / "edge_vpr_students" / "official_mixvpr_t0_gt1"
REPORTS = ROOT / "reports" / "edge_vpr_students"
sys.path.insert(0, str(HUB))
from vpr_model import VPRModel


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def model_and_load_audit():
    model = VPRModel(
        backbone_arch="resnet50",
        pretrained=False,
        layers_to_freeze=2,
        layers_to_crop=[4],
        agg_arch="MixVPR",
        agg_config={
            "in_channels": 1024,
            "in_h": 20,
            "in_w": 20,
            "out_channels": 1024,
            "mix_depth": 4,
            "mlp_ratio": 1,
            "out_rows": 4,
        },
        faiss_gpu=False,
    )
    state = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    incompatible = model.load_state_dict(state, strict=True)
    assert not incompatible.missing_keys and not incompatible.unexpected_keys
    return model.eval(), {
        "checkpoint": str(CHECKPOINT),
        "checkpoint_sha256": sha256(CHECKPOINT),
        "state_tensors": len(state),
        "missing_keys": list(incompatible.missing_keys),
        "unexpected_keys": list(incompatible.unexpected_keys),
        "parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters_as_constructed": sum(p.numel() for p in model.parameters() if p.requires_grad),
    }


OFFICIAL_TRANSFORM = T.Compose(
    [
        T.Resize((320, 320), interpolation=T.InterpolationMode.BICUBIC, antialias=True),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


def image_tensor(path: Path):
    with Image.open(path) as image:
        return OFFICIAL_TRANSFORM(image.convert("RGB"))


def extract(model, floor: str, role: str, checkpoint_hash: str):
    names = (GT0 / f"{floor}_{role}.txt").read_text().splitlines()
    directory = CACHE / f"{floor}_{role}"
    directory.mkdir(parents=True, exist_ok=True)
    manifest = {
        "checkpoint_sha256": checkpoint_hash,
        "ordered_names": names,
        "shape": [len(names), 4096],
        "preprocess": "PIL RGB -> direct Resize((320,320), BICUBIC, antialias=True) -> ToTensor -> ImageNet normalize",
        "chunk_size": 16,
    }
    manifest_path = directory / "manifest.json"
    if manifest_path.exists():
        assert json.loads(manifest_path.read_text()) == manifest
    else:
        atomic_json(manifest_path, manifest)
    chunks = []
    for index, start in enumerate(range(0, len(names), 16)):
        end = min(start + 16, len(names))
        path = directory / f"chunk_{index:03d}.npz"
        valid = False
        if path.exists():
            item = np.load(path, allow_pickle=False)
            descriptor = item["descriptor"]
            valid = (
                item["names"].tolist() == names[start:end]
                and item["checkpoint_sha256"].item() == checkpoint_hash
                and descriptor.shape == (end - start, 4096)
                and np.isfinite(descriptor).all()
                and np.allclose(np.linalg.norm(descriptor, axis=1), 1, atol=1e-5)
            )
        if not valid:
            batch = torch.stack([image_tensor(EDGE / "assets" / floor / name) for name in names[start:end]])
            with torch.inference_mode():
                descriptor = model(batch).cpu().numpy().astype(np.float32)
            assert descriptor.shape == (end - start, 4096)
            assert np.isfinite(descriptor).all()
            assert np.allclose(np.linalg.norm(descriptor, axis=1), 1, atol=1e-5)
            tmp = path.with_suffix(".npz.tmp")
            with tmp.open("wb") as handle:
                np.savez_compressed(handle, descriptor=descriptor, names=np.asarray(names[start:end]), checkpoint_sha256=np.asarray(checkpoint_hash))
            tmp.replace(path)
        chunks.append(np.load(path, allow_pickle=False)["descriptor"].copy())
    descriptors = np.concatenate(chunks)
    final = CACHE / f"{floor}_{role}_fp32.npy"
    tmp = final.with_suffix(".npy.tmp")
    with tmp.open("wb") as handle:
        np.save(handle, descriptors)
    tmp.replace(final)
    return names, descriptors


def evaluate(floor, references, reference_descriptor, queries, query_descriptor):
    labels = json.loads((GT0 / f"{floor}_positive_labels.json").read_text())["primary"]
    rows = []
    for name, descriptor in zip(queries, query_descriptor):
        positive = set(labels[name])
        if not positive:
            rows.append(None)
            continue
        order = np.argsort(-(reference_descriptor @ descriptor), kind="mergesort")
        rows.append({k: bool(positive.intersection(references[i] for i in order[:k])) for k in (1, 5, 10)})
    valid = [row for row in rows if row is not None]
    return {
        "references": len(references),
        "queries": len(queries),
        "zero_positive": len(rows) - len(valid),
        "valid_queries": len(valid),
        "recall": {f"R@{k}": float(np.mean([row[k] for row in valid])) for k in (1, 5, 10)},
    }


def main():
    torch.set_num_threads(4)
    model, load_audit = model_and_load_audit()
    results = {}
    comparison = {}
    hits = {1: 0, 5: 0, 10: 0}
    valid_total = 0
    for floor in ("floor6", "floor7"):
        reference_names, references = extract(model, floor, "reference", load_audit["checkpoint_sha256"])
        query_names, queries = extract(model, floor, "query", load_audit["checkpoint_sha256"])
        result = evaluate(floor, reference_names, references, query_names, queries)
        results[floor] = result
        valid_total += result["valid_queries"]
        for k in hits:
            hits[k] += round(result["recall"][f"R@{k}"] * result["valid_queries"])
        current_ref = np.load(RUNTIME_T0 / f"{floor}_reference_fp32.npy")
        current_query = np.load(RUNTIME_T0 / f"{floor}_query_fp32.npy")
        cosine = np.concatenate([np.sum(references * current_ref, axis=1), np.sum(queries * current_query, axis=1)])
        comparison[floor] = {
            "images": len(cosine),
            "official_vs_runtime_t0_cosine": {
                "min": float(cosine.min()),
                "p10": float(np.quantile(cosine, 0.1)),
                "median": float(np.median(cosine)),
                "p90": float(np.quantile(cosine, 0.9)),
                "max": float(cosine.max()),
            },
        }
    combined = {f"R@{k}": hits[k] / valid_total for k in hits}
    report = {
        "status": "PASS",
        "identity": "official pretrained MixVPR ResNet50-layer3 + original MixVPR aggregator",
        "load_audit": load_audit,
        "architecture": {
            "input": [1, 3, 320, 320],
            "backbone": "ResNet50 cropped before layer4",
            "backbone_output": [1, 1024, 20, 20],
            "aggregator": "MixVPR in_channels=1024, 20x20, mix_depth=4, mlp_ratio=1, out_channels=1024, out_rows=4",
            "descriptor": [1, 4096],
            "l2_normalized": True,
        },
        "official_preprocessing": "direct PIL RGB bicubic stretch 320x320 with antialias, ToTensor, ImageNet normalize",
        "project_runtime_t0_preprocessing": "native -> aspect-preserving INTER_AREA cap 384x480 -> RGB tensor -> ImageNet normalize -> torch bilinear stretch 320x320",
        "preprocessing_equivalent": False,
        "descriptor_comparison": comparison,
        "gt1_role": "baseline measurement only; not used for selection or tuning",
        "floor6": results["floor6"],
        "floor7": results["floor7"],
        "combined": combined,
    }
    atomic_json(REPORTS / "official_mixvpr_teacher_audit.json", report)
    print(json.dumps(report))


if __name__ == "__main__":
    main()
