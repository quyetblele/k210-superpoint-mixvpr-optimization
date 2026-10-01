"""Frozen teacher-vs-student Pitts30k-test comparison.
Adds only the actual frozen 4096D KD teacher; reuses the already locked
D4 FP32/INT8 Pitts30k results without retraining or retuning.
"""
from __future__ import annotations
import os
os.environ.update(OMP_NUM_THREADS="2", MKL_NUM_THREADS="2",
                  OPENBLAS_NUM_THREADS="2", PYTHONDONTWRITEBYTECODE="1")
from pathlib import Path
import argparse, hashlib, importlib.util, json, sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "datasets/pitts30k_test"
BASE = Path("/mnt/d/k210_benchmarks/pitts30k_test/artifacts")
OUT = Path("/mnt/d/k210_benchmarks/pitts30k_test/teacher_student_artifacts")
VENDOR = ROOT / "mixvpr/vendor/MixVPR"
TEACHER = Path("/home/quyet/training_recovery/full/teacher_frozen/teacher.pt")
TEACHER_MANIFEST = TEACHER.with_name("manifest.json")
BASE_BENCH = ROOT / "mixvpr/pitts30k_benchmark.py"

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()

def put(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)

def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

BASEMOD = module("pitts_base", BASE_BENCH)

def teacher_model():
    import torch
    import torch.nn as nn
    ResNet = module("teacher_resnet", VENDOR / "models/backbones/resnet.py").ResNet
    MixVPR = module("teacher_mixvpr", VENDOR / "models/aggregators/mixvpr.py").MixVPR
    class Teacher(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = ResNet("resnet50", pretrained=False, layers_to_crop=[4])
            self.aggregator = MixVPR(1024, 20, 20, 1024, 4, 1, 4)
        def forward(self, x):
            return self.aggregator(self.backbone(x))
    ckpt = torch.load(TEACHER, map_location="cpu", weights_only=False)
    model = Teacher()
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    assert ckpt["identity"] == "Official_ResNet50_MixVPR"
    return model.eval()

def teacher_image(path):
    from PIL import Image
    with Image.open(path) as im:
        im = im.convert("RGB").resize((320, 320), Image.Resampling.BICUBIC)
        x = np.array(im, dtype=np.float32).transpose(2, 0, 1) / np.float32(255)
    mean = np.array([.485, .456, .406], np.float32)[:, None, None]
    std = np.array([.229, .224, .225], np.float32)[:, None, None]
    return np.ascontiguousarray((x - mean) / std)

def lock():
    OUT.mkdir(parents=True, exist_ok=True)
    base_protocol = json.loads((BASE / "protocol.json").read_text())
    base_results = json.loads((BASE / "results.json").read_text())
    manifest = json.loads(TEACHER_MANIFEST.read_text())
    assert base_results["status"] == "BENCHMARK_COMPLETE"
    assert base_protocol["benchmark"] == "Pitts30k-test"
    assert manifest["status"] == "FROZEN"
    assert sha(TEACHER) == manifest["sha256"]
    assert manifest["architecture"].endswith("320x320, 4096D")
    model = teacher_model()
    import torch
    with torch.inference_mode():
        y = model(torch.zeros(1, 3, 320, 320))
    assert tuple(y.shape) == (1, 4096)
    assert torch.isfinite(y).all() and abs(float(y.norm()) - 1.0) < 1e-5
    protocol = {
        "status": "LOCKED_BEFORE_TEACHER_TEST_INFERENCE",
        "benchmark": "Pitts30k-test",
        "base_protocol_sha256": sha(BASE / "protocol.json"),
        "base_results_sha256": sha(BASE / "results.json"),
        "teacher_checkpoint": str(TEACHER),
        "teacher_checkpoint_sha256": sha(TEACHER),
        "teacher_manifest_sha256": sha(TEACHER_MANIFEST),
        "teacher_architecture": manifest["architecture"],
        "teacher_preprocessing": manifest["preprocessing"],
        "teacher_descriptor_dim": 4096,
        "models_compared": ["teacher_frozen_4096D", "D4_FP32", "D4_INT8"],
        "student_results_reused_from_locked_base_benchmark": True,
        "no_training_tuning_calibration_or_checkpoint_change": True,
        "script_sha256": sha(__file__),
    }
    put(OUT / "protocol.json", protocol)
    print("LOCKED", sha(OUT / "protocol.json"))

def verify_protocol():
    p = json.loads((OUT / "protocol.json").read_text())
    assert p["script_sha256"] == sha(__file__)
    assert p["teacher_checkpoint_sha256"] == sha(TEACHER)
    assert p["teacher_manifest_sha256"] == sha(TEACHER_MANIFEST)
    assert p["base_protocol_sha256"] == sha(BASE / "protocol.json")
    assert p["base_results_sha256"] == sha(BASE / "results.json")
    return p

def infer():
    import torch
    torch.set_num_threads(2)
    assert torch.cuda.is_available()
    verify_protocol()
    split, names, _ = BASEMOD.split()
    model = teacher_model().cuda()
    folder = OUT / "teacher"
    folder.mkdir(parents=True, exist_ok=True)
    batch_size = 8
    block_size = 128
    with torch.inference_mode():
        for start in range(0, len(names), block_size):
            ids = list(range(start, min(start + block_size, len(names))))
            dest = folder / f"block{start:05d}.npy"
            if dest.exists():
                arr = np.load(dest)
                assert arr.shape == (len(ids), 4096) and np.isfinite(arr).all()
                continue
            values = []
            for j in range(0, len(ids), batch_size):
                x = np.stack([teacher_image(DATA / "images" / names[i])
                              for i in ids[j:j + batch_size]])
                y = model(torch.from_numpy(x).cuda()).float().cpu().numpy()
                assert y.shape[1] == 4096 and np.isfinite(y).all()
                values.append(y)
            arr = np.concatenate(values, axis=0).astype(np.float32, copy=False)
            tmp = dest.with_suffix(".tmp")
            with tmp.open("wb") as f:
                np.save(f, arr)
            tmp.replace(dest)
            print("teacher", len(ids) + start, "/", len(names), flush=True)

    put(folder / "complete.json", {
        "protocol_sha256": sha(OUT / "protocol.json"),
        "blocks": {p.name: sha(p) for p in sorted(folder.glob("block*.npy"))}
    })

def score():
    verify_protocol()
    s, names, gt = BASEMOD.split()
    folder = OUT / "teacher"
    complete = json.loads((folder / "complete.json").read_text())
    assert complete["protocol_sha256"] == sha(OUT / "protocol.json")
    values = np.empty((len(names), 4096), np.float32)
    cursor = 0
    for path in sorted(folder.glob("block*.npy")):
        assert sha(path) == complete["blocks"][path.name]
        arr = np.load(path)
        values[cursor:cursor + len(arr)] = arr
        cursor += len(arr)
    assert cursor == len(names) and np.isfinite(values).all()
    teacher_recalls, teacher_hits, teacher_pred = BASEMOD.recalls(
        values[:s.numDb], values[s.numDb:], gt)
    np.savez(OUT / "teacher_predictions.npz",
             predictions=teacher_pred, hits=teacher_hits)

    base = json.loads((BASE / "results.json").read_text())["results"]
    student = base["D4"]["recalls"]
    int8 = base["INT8"]["recalls"]
    retention = {}
    for key in ("R@1", "R@5", "R@10"):
        t = teacher_recalls[key]
        retention[key] = {
            "student_fp32_vs_teacher": student[key] / t,
            "student_int8_vs_teacher": int8[key] / t,
            "student_fp32_delta_pp": (student[key] - t) * 100.0,
            "student_int8_delta_pp": (int8[key] - t) * 100.0,
            "int8_vs_student_fp32_delta_pp": (int8[key] - student[key]) * 100.0,
        }

    result = {
        "status": "TEACHER_STUDENT_TEST_COMPLETE",
        "protocol_sha256": sha(OUT / "protocol.json"),
        "teacher_frozen_4096D": {
            "recalls": teacher_recalls,
            "correct_counts": teacher_hits.sum(0).tolist(),
            "queries": s.numQ,
            "descriptor": 4096,
        },
        "student_D4_FP32": base["D4"],
        "student_D4_INT8": base["INT8"],
        "official_512_reference": base["official"],
        "retention_and_deltas": retention,
        "note": "Teacher is the frozen 4096D KD teacher used by this project. "
                "Official-512 is context only, not the KD teacher."
    }
    put(OUT / "results.json", result)
    print(json.dumps(result, indent=2))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["lock", "infer", "score", "all"])
    args = parser.parse_args()
    if args.stage in ("lock", "all"):
        lock()
    if args.stage in ("infer", "all"):
        infer()
    if args.stage in ("score", "all"):
        score()

if __name__ == "__main__":
    main()
