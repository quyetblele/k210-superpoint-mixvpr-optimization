from __future__ import annotations
from pathlib import Path
import json

import numpy as np
import torch
import torch.nn.functional as F

from .data import load_bundle, make_plan, training_batch
from .losses import retrieval_loss_with_stats, ranking_kd
from .trainer import initialize_first_stage

def training_smoke(cfg: dict, output: Path) -> dict:
    """One optimizer update. No DEV selection, no checkpoint promotion."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA_REQUIRED_NO_CPU_FALLBACK")
    output.mkdir(parents=True, exist_ok=True)

    seed = int(cfg["seed"])
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda")

    data = load_bundle(cfg)
    plan = make_plan(
        data, 1, seed, sampler=cfg.get("sampler")
    )
    bank = np.load(data["teacher_bank_path"], mmap_mode="r")
    offset = {
        int(image_id): index
        for index, image_id in enumerate(
            data["teacher_bank_manifest"]["ids"]
        )
    }

    model, _parent, init_device, init_report = initialize_first_stage(
        cfg, "00_large", output / "init"
    )
    if init_device != device:
        raise RuntimeError("smoke initialization device mismatch")
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["training"]["lr"]),
        weight_decay=float(cfg["training"]["weight_decay"]),
    )
    optimizer.zero_grad(set_to_none=True)

    totals = {"task": 0.0, "teacher_kd": 0.0}
    micro_reports = []
    accumulation = int(cfg["training"]["accumulation"])
    for micro, item in enumerate(plan[0]):
        x, labels = training_batch(
            item, data["rows"], 240, device
        )
        teacher_values = np.stack([
            bank[offset[int(image_id)]].astype(np.float32)
            for image_id in item[:, 0]
        ])
        teacher = F.normalize(
            torch.from_numpy(teacher_values).to(device),
            dim=1,
        )

        with torch.amp.autocast("cuda", dtype=torch.float16):
            student = model(x)
        student = F.normalize(student.float(), dim=1)

        task, miner_stats = retrieval_loss_with_stats(
            student, labels
        )
        kd = ranking_kd(student, teacher, labels)
        loss = (task + kd) / accumulation
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite smoke loss")
        loss.backward()

        totals["task"] += task.item() / accumulation
        totals["teacher_kd"] += kd.item() / accumulation
        micro_reports.append({
            "micro": micro,
            "student_shape": list(student.shape),
            "teacher_shape": list(teacher.shape),
            "task_loss": task.item(),
            "teacher_kd": kd.item(),
            "mined_positive_pairs": miner_stats["positive_pairs"],
            "mined_negative_pairs": miner_stats["negative_pairs"],
        })

    grad_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), 10.0
    )
    if not torch.isfinite(grad_norm):
        raise RuntimeError("non-finite smoke gradient")
    optimizer.step()

    report = {
        "status": "PASS",
        "scope": "single update smoke only; not training evidence",
        "student_stage": "00_large",
        "initialization": init_report,
        "student_descriptor": 512,
        "teacher_descriptor": int(teacher.shape[1]),
        "microbatches": micro_reports,
        "mean_losses": totals,
        "grad_norm": float(grad_norm),
    }
    (output / "training_smoke.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    return report
