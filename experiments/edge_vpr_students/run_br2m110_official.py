"""Trusted schema-v2 smoke, official KD training, resume, and final audit."""
from __future__ import annotations

import hashlib
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
from pytorch_metric_learning import losses, miners

ROOT = Path("/home/quyet/k210_lab")
EXP = ROOT / "experiments" / "edge_vpr_students"
sys.path.insert(0, str(EXP))
import train_s512_b_kd as base
from s512_br2m110_model import BR2M110, FROZEN_CONFIG
from s512_checkpoint_v2 import restore_rng_state, save

OFFICIAL = ROOT / "artifacts" / "edge_vpr_students" / "s512_br2m110_official"
SMOKE = ROOT / "artifacts" / "edge_vpr_students" / "s512_br2m110_smoke"
REPORT = ROOT / "reports" / "edge_vpr_students"
TRACE = OFFICIAL / "official_trace.jsonl"
SEED = base.SEED


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text)
    temporary.replace(path)


def digest_object(value) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def set_seed() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)


def make_objects():
    model = BR2M110()
    assert sum(p.numel() for p in model.parameters()) == 691300
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=4, eta_min=3e-5
    )
    return model, optimizer, scheduler


def complete_plans(rows, positive, valid):
    plans = {}
    for epoch in range(1, 5):
        plan = base.epoch_plan(epoch, positive, valid)
        enriched = []
        for batch in plan:
            ids = batch["row_indices"]
            image_ids = [f"{rows[i][0]}/{rows[i][1]}" for i in ids]
            enriched.append(
                {
                    "row_indices": ids,
                    "labels": batch["labels"],
                    "image_ids": image_ids,
                    "pair_ids": [
                        {"anchor": image_ids[i], "positive": image_ids[i + 1]}
                        for i in range(0, len(ids), 2)
                    ],
                }
            )
        assert len(enriched) == 48
        plans[epoch] = enriched
    return plans


def execute_batch(model, optimizer, task_loss, miner, rows, teachers, entry):
    ids = entry["row_indices"]
    labels = torch.tensor(entry["labels"], dtype=torch.long)
    inputs = torch.stack([base.student_tensor(rows[i], 192, 352) for i in ids])
    teacher = torch.from_numpy(teachers[ids])
    output = model(inputs)
    assert list(inputs.shape) == [16, 3, 352, 192]
    assert list(output.shape) == [16, 512]
    task = task_loss(output, labels, miner(output, labels))
    kd = torch.nn.functional.mse_loss(output @ output.T, teacher @ teacher.T)
    total = task + 2.0 * kd
    assert torch.isfinite(task) and torch.isfinite(kd) and torch.isfinite(total)
    optimizer.zero_grad()
    total.backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)
    gradient_sum = sum(g.abs().sum().item() for g in gradients)
    assert math.isfinite(gradient_sum) and gradient_sum > 0
    optimizer.step()
    return {
        "task_loss": float(task.detach()),
        "kd_loss": float(kd.detach()),
        "total_loss": float(total.detach()),
        "lr": float(optimizer.param_groups[0]["lr"]),
        "grad_parameter_count": len(gradients),
        "grad_abs_sum": gradient_sum,
    }


def run_smoke(rows, positive, valid, teachers) -> None:
    set_seed()
    plans = complete_plans(rows, positive, valid)
    model, optimizer, scheduler = make_objects()
    before = [p.detach().clone() for p in model.parameters()]
    task_loss = losses.MultiSimilarityLoss(alpha=1.0, beta=50.0, base=0.0)
    miner = miners.MultiSimilarityMiner(epsilon=0.1)
    measured = execute_batch(
        model, optimizer, task_loss, miner, rows, teachers, plans[1][0]
    )
    update = max(float((p.detach() - old).abs().max()) for p, old in zip(model.parameters(), before))
    assert update > 0
    path = SMOKE / "smoke_v2.pt"
    save(
        path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        current_epoch=1,
        next_batch_index=1,
        global_step=1,
        training_config=FROZEN_CONFIG,
        epoch_plan=plans[1],
        all_epoch_plans=plans,
        best_metric_name="none_during_training",
        best_metric_value=float("-inf"),
        best_checkpoint_path=None,
    )
    loaded = torch.load(path, weights_only=False)
    fresh_model, fresh_optimizer, fresh_scheduler = make_objects()
    fresh_model.load_state_dict(loaded["model_state_dict"])
    fresh_optimizer.load_state_dict(loaded["optimizer_state_dict"])
    fresh_scheduler.load_state_dict(loaded["scheduler_state_dict"])
    assert (loaded["current_epoch"], loaded["next_batch_index"], loaded["global_step"]) == (1, 1, 1)
    report = {
        "status": "PASS",
        "checkpoint": str(path),
        "cursor": [1, 1, 1],
        "input": [16, 3, 352, 192],
        "student": [16, 512],
        "teacher": [16, 4096],
        "max_parameter_update": update,
        **measured,
    }
    atomic_text(REPORT / "s512_br2m110_smoke.json", json.dumps(report, indent=2) + "\n")


def initialize_official(rows, positive, valid) -> Path:
    initial = OFFICIAL / "epoch1_next0.pt"
    if initial.exists():
        return initial
    set_seed()
    plans = complete_plans(rows, positive, valid)
    model, optimizer, scheduler = make_objects()
    OFFICIAL.mkdir(parents=True, exist_ok=True)
    config = {
        **FROZEN_CONFIG,
        "seed": SEED,
        "training_images": 1444,
        "epochs": 4,
        "steps_per_epoch": 48,
        "batch_images": 16,
        "task_loss": "MultiSimilarityLoss(alpha=1,beta=50,base=0) + MultiSimilarityMiner(epsilon=0.1)",
        "kd_loss": "MSE(student_similarity_matrix, teacher_similarity_matrix)",
        "loss_weights": {"task": 1.0, "kd": 2.0},
        "optimizer": "AdamW(lr=3e-4,weight_decay=1e-4)",
        "scheduler": "CosineAnnealingLR(T_max=4,eta_min=3e-5)",
        "preprocessing": "raw BGR -> RGB -> direct INTER_AREA 192x352 -> /255 -> ImageNet normalize -> CHW",
        "all_epoch_plans_sha256": digest_object(plans),
    }
    atomic_text(OFFICIAL / "frozen_config.json", json.dumps(config, indent=2) + "\n")
    atomic_text(TRACE, "")
    save(
        initial,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        current_epoch=1,
        next_batch_index=0,
        global_step=0,
        training_config=config,
        epoch_plan=plans[1],
        all_epoch_plans=plans,
        trace_records=[],
        best_metric_name="none_during_training",
        best_metric_value=float("-inf"),
        best_checkpoint_path=None,
    )
    return initial


def train_to_completion(start: Path, rows, teachers) -> Path:
    checkpoint = start
    task_loss = losses.MultiSimilarityLoss(alpha=1.0, beta=50.0, base=0.0)
    miner = miners.MultiSimilarityMiner(epsilon=0.1)
    while True:
        state = torch.load(checkpoint, weights_only=False)
        if state["global_step"] == 192:
            break
        assert state["schema_version"] == 2
        model, optimizer, scheduler = make_objects()
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        scheduler.load_state_dict(state["scheduler_state_dict"])
        restore_rng_state(state["rng_state"])
        epoch = state["current_epoch"]
        start_batch = state["next_batch_index"]
        global_step = state["global_step"]
        plans = state["all_epoch_plans"]
        trace = state.get("trace_records", [])
        # Checkpoint is the transaction source of truth; repair a stale external
        # trace after an interruption without repeating optimizer work.
        atomic_text(TRACE, "".join(json.dumps(record) + "\n" for record in trace))
        end_batch = min(48, start_batch + 4)
        for batch_index in range(start_batch, end_batch):
            assert (epoch, batch_index) not in {(r["epoch"], r["batch_index"]) for r in trace}
            entry = plans[epoch][batch_index]
            measured = execute_batch(model, optimizer, task_loss, miner, rows, teachers, entry)
            global_step += 1
            trace.append(
                {
                    "epoch": epoch,
                    "batch_index": batch_index,
                    "global_step": global_step,
                    "row_indices": entry["row_indices"],
                    "pair_ids": entry["pair_ids"],
                    "image_ids": entry["image_ids"],
                    "labels": entry["labels"],
                    **measured,
                }
            )
        if end_batch == 48:
            scheduler.step()
            next_epoch, next_batch = epoch + 1, 0
            next_plan = plans[next_epoch] if next_epoch <= 4 else []
        else:
            next_epoch, next_batch, next_plan = epoch, end_batch, plans[epoch]
        next_path = OFFICIAL / f"epoch{next_epoch}_next{next_batch}.pt"
        save(
            next_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            current_epoch=next_epoch,
            next_batch_index=next_batch,
            global_step=global_step,
            training_config=state["training_config"],
            epoch_plan=next_plan,
            all_epoch_plans=plans,
            trace_records=trace,
            best_metric_name=state["best_metric_name"],
            best_metric_value=state["best_metric_value"],
            best_checkpoint_path=state["best_checkpoint_path"],
        )
        atomic_text(TRACE, "".join(json.dumps(record) + "\n" for record in trace))
        checkpoint = next_path
    final_state = torch.load(checkpoint, weights_only=False)
    final_path = OFFICIAL / "final.pt"
    temporary = final_path.with_suffix(".pt.tmp")
    torch.save(final_state, temporary)
    temporary.replace(final_path)
    return final_path


def audit(final_path: Path) -> dict:
    state = torch.load(final_path, weights_only=False)
    trace = state["trace_records"]
    assert state["schema_version"] == 2
    assert (state["current_epoch"], state["next_batch_index"], state["global_step"]) == (5, 0, 192)
    assert len(trace) == 192
    keys = [(r["epoch"], r["batch_index"]) for r in trace]
    expected = [(epoch, batch) for epoch in range(1, 5) for batch in range(48)]
    assert keys == expected and len(set(keys)) == 192
    assert [r["global_step"] for r in trace] == list(range(1, 193))
    for record in trace:
        planned = state["all_epoch_plans"][record["epoch"]][record["batch_index"]]
        assert record["row_indices"] == planned["row_indices"]
        assert record["pair_ids"] == planned["pair_ids"]
        assert record["image_ids"] == planned["image_ids"]
        assert record["labels"] == planned["labels"]
    summaries = {}
    for epoch in range(1, 5):
        rows = [r for r in trace if r["epoch"] == epoch]
        summaries[str(epoch)] = {
            "steps": len(rows),
            "mean_task_loss": float(np.mean([r["task_loss"] for r in rows])),
            "mean_kd_loss": float(np.mean([r["kd_loss"] for r in rows])),
            "mean_total_loss": float(np.mean([r["total_loss"] for r in rows])),
            "lr_start": rows[0]["lr"],
            "lr_end": rows[-1]["lr"],
        }
    fresh_model, fresh_optimizer, fresh_scheduler = make_objects()
    fresh_model.load_state_dict(state["model_state_dict"])
    fresh_optimizer.load_state_dict(state["optimizer_state_dict"])
    fresh_scheduler.load_state_dict(state["scheduler_state_dict"])
    result = {
        "status": "PASS",
        "checkpoint": str(final_path),
        "checkpoint_sha256": hashlib.sha256(final_path.read_bytes()).hexdigest(),
        "cursor": [5, 0, 192],
        "trace_path": str(TRACE),
        "trace_records": 192,
        "duplicates": 0,
        "missing": 0,
        "global_steps_exact": True,
        "plan_consistency": True,
        "reload_sanity": "PASS",
        "epochs": summaries,
    }
    atomic_text(REPORT / "s512_br2m110_training_final.json", json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    torch.set_num_threads(4)
    rows, positive, valid, split_audit = base.build_split()
    assert len(rows) == 1444
    teachers = base.teacher_cache(rows)
    run_smoke(rows, positive, valid, teachers)
    initial = initialize_official(rows, positive, valid)
    final_path = train_to_completion(initial, rows, teachers)
    result = audit(final_path)
    result["split_audit"] = split_audit
    print(json.dumps(result))


if __name__ == "__main__":
    main()
