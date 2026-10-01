from __future__ import annotations
from pathlib import Path
import json
import math

import numpy as np
import torch
import torch.nn.functional as F

from .config import stage_ids
from .data import image_tensor, training_batch
from .losses import retrieval_loss_with_stats, ranking_kd
from .trainer import (
    _teacher_bank,
    _teacher_batch,
    initialize_first_stage,
    master_plan,
)


def _clean_batch(item, rows, device):
    images = torch.stack([
        image_tensor(rows[int(i)]["path"], 240, None)
        for i, _label, _seed in item
    ])
    return images.to(device, non_blocking=True)


def _grad_geometry(task, kd, params):
    task_grads = torch.autograd.grad(
        task, params, retain_graph=True, allow_unused=True
    )
    kd_grads = torch.autograd.grad(
        kd, params, retain_graph=False, allow_unused=True
    )
    dot = task.new_zeros(())
    task_sq = task.new_zeros(())
    kd_sq = task.new_zeros(())
    for gt, gk in zip(task_grads, kd_grads):
        if gt is not None:
            task_sq += gt.float().square().sum()
        if gk is not None:
            kd_sq += gk.float().square().sum()
        if gt is not None and gk is not None:
            dot += (gt.float() * gk.float()).sum()
    task_norm = float(task_sq.sqrt())
    kd_norm = float(kd_sq.sqrt())
    denom = math.sqrt(float(task_sq) * float(kd_sq))
    cosine = float(dot) / denom if denom > 0.0 else float("nan")
    return task_norm, kd_norm, cosine


def _teacher_margin(teacher, labels):
    sim = teacher @ teacher.T
    diagonal = torch.eye(
        len(labels), dtype=torch.bool, device=labels.device
    )
    same = labels[:, None].eq(labels[None, :])
    positive = same & ~diagonal
    negative = ~same
    pos = sim.masked_fill(~positive, -1e4).max(1).values
    neg = sim.masked_fill(~negative, -1e4).max(1).values
    margin = pos - neg
    return {
        "mean": float(margin.mean()),
        "min": float(margin.min()),
        "positive_fraction": float((margin > 0).float().mean()),
    }


def _aggregate(records, key):
    values = [
        float(x[key]) for x in records
        if x.get(key) is not None
        and math.isfinite(float(x[key]))
    ]
    if not values:
        return {
            "count": 0,
            "mean": None,
            "min": None,
            "max": None,
        }
    return {
        "count": len(values),
        "mean": float(np.mean(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def _spread_select(pool, count):
    if count <= 0:
        return []
    if not pool:
        raise ValueError("cannot sample an empty diagnostic pool")
    indices = np.linspace(
        0, len(pool) - 1, num=count, dtype=np.int64
    )
    return [pool[int(i)] for i in indices]


def _stratified_positions(plan, rows, parent_steps, batches):
    micros_per_step = len(plan[0])
    pools = {"indoor": [], "project": []}
    for flat_index in range(parent_steps * micros_per_step):
        step_index, micro_index = divmod(flat_index, micros_per_step)
        item = plan[step_index, micro_index]
        domain = rows[int(item[0, 0])]["domain"]
        pools.setdefault(domain, []).append(flat_index)
    total = sum(len(x) for x in pools.values())
    project_count = 0
    if pools.get("project") and pools.get("indoor") and batches > 1:
        project_count = max(
            1, int(round(batches * len(pools["project"]) / total))
        )
        project_count = min(project_count, batches - 1)
    indoor_count = batches - project_count
    selected = (
        _spread_select(pools["indoor"], indoor_count)
        + _spread_select(pools["project"], project_count)
    )
    return sorted(selected)


def kd_diagnostic(cfg: dict, run_root: Path, batches: int = 8):
    if batches < 1:
        raise ValueError("batches must be >= 1")
    output = run_root / "diagnostics" / "kd_gradient"
    output.mkdir(parents=True, exist_ok=True)
    plan, data = master_plan(cfg, run_root)
    stage_id = stage_ids(cfg)[0]
    model, _parent, device, init_report = initialize_first_stage(
        cfg, stage_id, output / "init"
    )
    model.train()
    bank, bank_offset = _teacher_bank(data)
    params = [p for p in model.parameters() if p.requires_grad]
    rows = data["rows"]
    task_weight = float(cfg["training"]["task_weight"])
    kd_weight = float(cfg["training"]["teacher_kd_weight"])
    records = []
    parent_steps = int(cfg["training"]["parent_max_updates"])
    micros_per_step = len(plan[0])
    flat_positions = _stratified_positions(
        plan, rows, parent_steps, batches
    )

    for flat_index in flat_positions:
        step_index = int(flat_index) // micros_per_step
        micro_index = int(flat_index) % micros_per_step
        item = plan[step_index, micro_index]
        x, labels = training_batch(item, rows, 240, device)
        teacher = _teacher_batch(
            item, bank, bank_offset, device
        )
        clean_x = _clean_batch(item, rows, device)

        with torch.amp.autocast("cuda", dtype=torch.float16):
            student_aug = model(x)
        student_aug = F.normalize(student_aug.float(), dim=1)
        task, miner_stats = retrieval_loss_with_stats(
            student_aug, labels
        )
        kd_aug = ranking_kd(student_aug, teacher, labels)

        with torch.inference_mode():
            with torch.amp.autocast("cuda", dtype=torch.float16):
                student_clean = model(clean_x)
            student_clean = F.normalize(
                student_clean.float(), dim=1
            )
            kd_clean = ranking_kd(
                student_clean, teacher, labels
            )
            same_image_cos = (
                student_aug.detach() * student_clean
            ).sum(1).mean()

        task_norm, kd_norm, grad_cos = _grad_geometry(
            task, kd_aug, params
        )
        task_active = task_norm > 1e-12
        gradient_pair_active = (
            task_active
            and kd_norm > 1e-12
            and math.isfinite(grad_cos)
        )
        weighted_ratio = None
        valid_grad_cos = None
        if gradient_pair_active:
            weighted_ratio = (
                kd_weight * kd_norm
            ) / (task_weight * task_norm)
            valid_grad_cos = grad_cos

        margin = _teacher_margin(teacher, labels)
        records.append({
            "step": step_index,
            "micro": micro_index,
            "anchor_domain": rows[int(item[0, 0])]["domain"],
            "task_loss": float(task.detach()),
            "teacher_kd_aug": float(kd_aug.detach()),
            "teacher_kd_clean": float(kd_clean.detach()),
            "task_grad_norm": task_norm,
            "teacher_kd_grad_norm": kd_norm,
            "task_active": task_active,
            "gradient_pair_active": gradient_pair_active,
            "mined_positive_pairs": miner_stats["positive_pairs"],
            "mined_negative_pairs": miner_stats["negative_pairs"],
            "weighted_kd_to_task_grad_norm": weighted_ratio,
            "grad_cosine_task_vs_teacher_kd": valid_grad_cos,
            "student_clean_vs_aug_cosine": float(same_image_cos),
            "teacher_margin_mean": margin["mean"],
            "teacher_margin_min": margin["min"],
            "teacher_margin_positive_fraction": margin[
                "positive_fraction"
            ],
        })

    if len(records) != batches:
        raise RuntimeError("not enough diagnostic batches")

    summary_keys = [
        "task_loss",
        "teacher_kd_aug",
        "teacher_kd_clean",
        "task_grad_norm",
        "teacher_kd_grad_norm",
        "student_clean_vs_aug_cosine",
        "teacher_margin_mean",
        "teacher_margin_positive_fraction",
        "mined_positive_pairs",
        "mined_negative_pairs",
    ]
    summary = {
        key: _aggregate(records, key)
        for key in summary_keys
    }
    task_active_records = [
        x for x in records if x["task_active"]
    ]
    gradient_active_records = [
        x for x in records if x["gradient_pair_active"]
    ]
    active_gradient_summary = {
        "weighted_kd_to_task_grad_norm": _aggregate(
            gradient_active_records,
            "weighted_kd_to_task_grad_norm",
        ),
        "grad_cosine_task_vs_teacher_kd": _aggregate(
            gradient_active_records,
            "grad_cosine_task_vs_teacher_kd",
        ),
    }
    task_active_fraction = len(task_active_records) / len(records)
    mean_cos = active_gradient_summary[
        "grad_cosine_task_vs_teacher_kd"
    ]["mean"]
    mean_ratio = active_gradient_summary[
        "weighted_kd_to_task_grad_norm"
    ]["mean"]
    kd_aug_mean = summary["teacher_kd_aug"]["mean"]
    kd_clean_mean = summary["teacher_kd_clean"]["mean"]
    report = {
        "status": "MEASURED",
        "diagnostic_version": 2,
        "stage_id": stage_id,
        "batches": batches,
        "weights": {
            "task": task_weight,
            "teacher_kd": kd_weight,
        },
        "sample_domains": {
            domain: sum(
                1 for item in records
                if item["anchor_domain"] == domain
            )
            for domain in sorted({
                item["anchor_domain"] for item in records
            })
        },
        "sample_flat_positions": [
            int(x) for x in flat_positions
        ],
        "initialization": init_report,
        "summary": summary,
        "coverage": {
            "task_active_batches": len(task_active_records),
            "task_inactive_batches": len(records) - len(task_active_records),
            "task_active_fraction": task_active_fraction,
            "gradient_pair_active_batches": len(gradient_active_records),
        },
        "active_gradient_summary": active_gradient_summary,
        "heuristics": {
            "task_coverage_low": task_active_fraction < 0.50,
            "mean_gradient_conflict": (
                mean_cos is not None and mean_cos < -0.10
            ),
            "kd_gradient_dominant": (
                mean_ratio is not None and mean_ratio > 3.0
            ),
            "kd_gradient_too_weak": (
                mean_ratio is not None and mean_ratio < 0.10
            ),
            "augmentation_signal_shift": (
                kd_aug_mean is not None
                and kd_clean_mean is not None
                and kd_aug_mean > 1.5 * kd_clean_mean
            ),
        },
        "records": records,
        "notes": [
            "Heuristics are diagnostics, not automatic training decisions.",
            "Negative gradient cosine means task and teacher KD oppose locally.",
            "Weighted gradient ratio compares configured loss contributions.",
        ],
    }
    payload = json.dumps(report, indent=2) + "\n"
    versioned = output / f"report_b{batches}_v2.json"
    latest = output / "report.json"
    versioned.write_text(payload)
    latest.write_text(payload)
    return report
