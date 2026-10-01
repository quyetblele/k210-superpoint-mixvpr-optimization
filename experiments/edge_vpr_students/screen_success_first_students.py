"""Controlled 1500-step DEV screening for the two frozen SQ240 students."""
from __future__ import annotations

import argparse
import json
import math
import random
import sqlite3
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd.graph import save_on_cpu

ROOT = Path("/home/quyet/k210_lab")
sys.path.insert(0, str(ROOT / "experiments" / "edge_vpr_students"))
import success_first_train_core as core
from per_image_teacher import PerImageTeacher, POLICY

CANDIDATES = ("A_C144_H112", "B_C160_H96")


def load_bank(bank_dir: Path):
    manifest = json.loads((bank_dir / "manifest.json").read_text())
    database = sqlite3.connect(manifest["bank"])
    descriptors = np.load(manifest["descriptors"], mmap_mode="r")
    descriptor_rows = database.execute("SELECT place_id,path,offset FROM descriptors").fetchall()
    offsets = {path: offset for _, path, offset in descriptor_rows}
    place_offsets = {place: offset for place, _, offset in descriptor_rows}
    neighbors = {}
    for place, _, negative in database.execute("SELECT place_id,rank,negative_place_id FROM hard_neighbors ORDER BY place_id,rank"):
        neighbors.setdefault(place, []).append(negative)
    database.close()
    return manifest, descriptors, offsets, place_offsets, neighbors


def shared_plan(index, neighbors, output: Path, steps: int, accumulation: int,
                physical_places: int):
    path = output / f"shared_plan_{steps}steps_p{physical_places}_a{accumulation}.pt"
    if not path.exists():
        plan = core.make_plan(index, steps * accumulation, core.SEED + 811,
                              places_per_step=physical_places,
                              hard_neighbors=neighbors)
        core.atomic_torch(path, {"schema_version": 1, "seed": core.SEED + 811, "steps": steps,
                                 "accumulation": accumulation,
                                 "physical_places": physical_places, "plan": plan})
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if (payload["steps"] != steps or payload["accumulation"] != accumulation
            or payload.get("physical_places") != physical_places):
        raise RuntimeError("shared screening plan mismatch")
    return path, payload["plan"]


def train_candidate(identity: str, plan_path: Path, plan: list, bank, index, output: Path,
                    steps: int, accumulation: int, device):
    directory = output / identity; directory.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(core.SEED); torch.cuda.manual_seed_all(core.SEED); np.random.seed(core.SEED); random.seed(core.SEED)
    model = core.build_student(identity).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps, eta_min=3e-5)
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError('BF16 GPU support required for this controlled screening')
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    latest = directory / "latest.pt"; start = 0; best_key = [-1.0, -1.0, -1e9]; best_metrics = None; trace = []
    if latest.exists():
        checkpoint = torch.load(latest, map_location="cpu", weights_only=False)
        if checkpoint.get('precision') != 'bf16_forward_fp32_norm_loss':
            raise RuntimeError('precision changed: use a fresh screening namespace for both candidates')
        if checkpoint.get('teacher_target_policy') != POLICY:
            raise RuntimeError('incompatible old KD checkpoint; use new experiment namespace')
        model.load_state_dict(checkpoint["model_state_dict"]); optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"]); scaler.load_state_dict(checkpoint["scaler_state_dict"])
        start = checkpoint["optimizer_step"]; best_key = checkpoint["best_key"]
        best_metrics = checkpoint["best_metrics"]; trace = checkpoint["trace"]; core.restore_rng(checkpoint["rng_state"])
    descriptors, offsets, place_offsets = bank
    targets = PerImageTeacher(core.ROOT/'artifacts/success_first/per_image_teacher', device)
    model.train()
    for step in range(start, steps):
        optimizer.zero_grad(set_to_none=True); task_sum = kd_sum = relational_sum = confidence_sum = margin_sum = 0.0
        for micro in range(accumulation):
            item = plan[step * accumulation + micro]
            images, labels = core.batch_from_plan(item, 240, device)
            teacher = targets.targets(item['samples'])
            with save_on_cpu(pin_memory=False):
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    student = model.raw(images)
                with torch.amp.autocast('cuda', enabled=False):
                    student = F.normalize(student.float(),dim=1)
                    task = core.retrieval_loss(student, labels)
                    kd, confidence, margin = core.ranking_kd(student, teacher, labels)
                    relational = F.mse_loss(student @ student.T, teacher @ teacher.T)
                    total = (task + kd + 2.0 * relational) / accumulation
            if not torch.isfinite(total): raise RuntimeError(f"{identity} non-finite loss at step {step}")
            scaler.scale(total).backward()
            task_sum += float(task.detach()); kd_sum += float(kd.detach()); relational_sum += float(relational.detach())
            confidence_sum += float(confidence.detach()); margin_sum += float(margin.detach())
        scaler.unscale_(optimizer); grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        if not torch.isfinite(grad): raise RuntimeError(f"{identity} non-finite gradient at step {step}")
        scaler.step(optimizer); scaler.update(); scheduler.step()
        record = {"step": step + 1, "task": task_sum / accumulation, "ranking_kd": kd_sum / accumulation,
                  "relational_kd": relational_sum / accumulation, "teacher_confidence": confidence_sum / accumulation,
                  "batch_margin": margin_sum / accumulation, "grad_norm": float(grad), "lr": optimizer.param_groups[0]["lr"]}
        trace.append(record)
        if (step + 1) % 250 == 0 or step + 1 == steps:
            metrics = core.dev_metrics(model, index, 240, device, max_places=1000)
            key = [metrics["R@1"], metrics["R@5"], metrics["positive_hardest_negative_margin"]]
            if key > best_key:
                best_key = key; best_metrics = metrics
                core.atomic_torch(directory / "best.pt", {"model_state_dict": model.state_dict(),
                                  "identity": identity, "teacher_target_policy": POLICY,
                                  "teacher_contract": targets.contract,
                                  "optimizer_step": step + 1, "dev_metrics": metrics})
            model.train()
        checkpoint = {"schema_version": 2, "identity": identity, "teacher_target_policy": POLICY,
            "precision": "bf16_forward_fp32_norm_loss",
            "teacher_contract": targets.contract, "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(), "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(), "optimizer_step": step + 1, "best_key": best_key,
            "best_metrics": best_metrics, "trace": trace, "rng_state": core.rng_state(),
            "shared_plan": str(plan_path), "shared_plan_sha256": core.sha256(plan_path), "gt1_used": False}
        core.atomic_torch(latest, checkpoint)
        print(json.dumps({'identity':identity, **record}), flush=True)
    result = {"identity": identity, "steps": steps, "best_metrics": best_metrics,
              "teacher_contract": targets.contract,
              "best_checkpoint": str(directory / "best.pt"), "best_checkpoint_sha256": core.sha256(directory / "best.pt"),
              "stability": {"all_losses_finite": True, "max_grad_norm": max(x["grad_norm"] for x in trace),
                            "final_task": trace[-1]["task"], "final_ranking_kd": trace[-1]["ranking_kd"]}}
    core.atomic_json(directory / "report.json", result); return result


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--bank", type=Path, default=ROOT / "artifacts/success_first/strong_teacher_bank")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/success_first/student_screening")
    parser.add_argument("--physical-places", type=int, default=4)
    parser.add_argument("--accumulation", type=int, default=8)
    args = parser.parse_args(); device = core.require_cuda(); args.output.mkdir(parents=True, exist_ok=True)
    manifest, descriptors, offsets, place_offsets, neighbors = load_bank(args.bank)
    index = core.PlaceIndex()
    plan_path, plan = shared_plan(index, neighbors, args.output, args.steps,
                                  args.accumulation, args.physical_places)
    results = [train_candidate(identity, plan_path, plan, (descriptors, offsets, place_offsets), index, args.output,
                               args.steps, args.accumulation, device) for identity in CANDIDATES]
    a, b = results; am, bm = a["best_metrics"], b["best_metrics"]
    near = abs(am["R@1"] - bm["R@1"]) <= .005 and abs(am["R@5"] - bm["R@5"]) <= .005 and abs(am["positive_hardest_negative_margin"] - bm["positive_hardest_negative_margin"]) <= .01
    if near: winner = "A_C144_H112"
    else: winner = max(results, key=lambda r: (r["best_metrics"]["R@1"], r["best_metrics"]["R@5"], r["best_metrics"]["positive_hardest_negative_margin"]))["identity"]
    report = {"status": "COMPLETE", "strong_teacher_sha256": manifest["teacher_checkpoint_sha256"],
              "shared_plan_sha256": core.sha256(plan_path), "results": results, "winner": winner,
              "near_tie_headroom_rule_applied": near, "gt1_used": False}
    core.atomic_json(args.output / "screening_report.json", report); print(json.dumps(report))


if __name__ == "__main__": main()
