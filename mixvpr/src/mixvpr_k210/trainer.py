from __future__ import annotations
from pathlib import Path
import json
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from .config import (
    ROOT, spec_for_stage, stage_config, stage_ids, validate_progression
)
from .data import (
    load_bundle, make_plan, training_batch,
    encode_dev, dev_metrics, sha256
)
from .losses import retrieval_loss_with_stats, ranking_kd
from .model import K210MixVPR
from .legacy import load_legacy_d4_state, widen_preserving_function
from .pruning import prune_one_axis

def _put(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)

def _save(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, tmp)
    tmp.replace(path)

def _rng_state() -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all(),
    }

def _restore_rng(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    torch.cuda.set_rng_state_all(state["torch_cuda"])

def _metric_key(metrics: dict):
    x = metrics["Overall"]
    return (x["R@1"], x["R@5"], x["R@10"], x["margin"])

def _setup(seed: int):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA_REQUIRED_NO_CPU_FALLBACK")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(2)
    return torch.device("cuda")
def master_plan(cfg: dict, run_root: Path):
    validate_progression(cfg)
    parent_steps = int(cfg["training"]["parent_max_updates"])
    recovery_steps = int(cfg["training"]["recovery_max_updates"])
    total = parent_steps + recovery_steps * (len(cfg["stages"]) - 1)
    path = run_root / "shared" / "train_plan.npy"
    meta = path.with_suffix(".json")
    data = load_bundle(cfg)
    if path.exists():
        plan = np.load(path)
        if len(plan) != total:
            raise RuntimeError("existing master plan length does not match config")
        if sha256(path) != json.loads(meta.read_text())["sha256"]:
            raise RuntimeError("master plan hash mismatch")
        return plan, data
    path.parent.mkdir(parents=True, exist_ok=True)
    plan = make_plan(
        data,
        total,
        int(cfg["seed"]),
        sampler=cfg.get("sampler"),
    )
    np.save(path, plan)
    _put(meta, {
        "steps": total,
        "parent_steps": parent_steps,
        "recovery_steps_per_stage": recovery_steps,
        "sha256": sha256(path),
        "seed": int(cfg["seed"]),
        "sampler": dict(cfg.get("sampler", {"mode": "p2k4"})),
        "rule": "non-overlapping contiguous slices across one student lineage",
    })
    return plan, data

def plan_slice(cfg: dict, stage_id: str) -> tuple[int, int]:
    ids = stage_ids(cfg)
    index = ids.index(stage_id)
    parent_steps = int(cfg["training"]["parent_max_updates"])
    recovery = int(cfg["training"]["recovery_max_updates"])
    if index == 0:
        return 0, parent_steps
    start = parent_steps + (index - 1) * recovery
    return start, start + recovery

def _teacher_bank(data: dict):
    manifest = data["teacher_bank_manifest"]
    bank = np.load(data["teacher_bank_path"], mmap_mode="r")
    offset = {
        int(image_id): index
        for index, image_id in enumerate(manifest["ids"])
    }
    return bank, offset

def _teacher_batch(item, bank, offset, device):
    values = []
    for image_id in item[:, 0]:
        image_id = int(image_id)
        if image_id not in offset:
            raise RuntimeError(
                f"teacher descriptor missing for image {image_id}"
            )
        values.append(bank[offset[image_id]].astype(np.float32))
    tensor = torch.from_numpy(np.stack(values)).to(device)
    return F.normalize(tensor, dim=1)

def evaluate(model, data, device):
    _ids, offset, values = encode_dev(model, data, device)
    return dev_metrics(values, offset, data)
def initialize_first_stage(cfg: dict, stage_id: str, output: Path):
    seed = int(cfg["seed"])
    device = _setup(seed)
    init_mode = cfg.get("initialization", {}).get(
        "mode", "function_preserving_widen"
    )

    if init_mode == "random":
        model = K210MixVPR(spec_for_stage(cfg, stage_id))
        report = {
            "method": "random_init",
            "seed": seed,
            "source_checkpoint": None,
            "note": (
                "Dense large student is initialized directly; "
                "no smaller-student weights are copied."
            ),
        }
        _put(output / "initialization_report.json", report)
        _save(output / "init.pt", {
            "model_state_dict": model.state_dict(),
            "stage_id": stage_id,
            "seed": seed,
            "initialization": report["method"],
        })
        return model.to(device), None, device, report

    if init_mode == "checkpoint":
        checkpoint_path = Path(cfg["initialization"]["checkpoint"]).resolve()
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
        model = K210MixVPR(spec_for_stage(cfg, stage_id))
        model.load_state_dict(
            checkpoint["model_state_dict"], strict=True
        )
        checkpoint_hash = sha256(checkpoint_path)
        report = {
            "method": "checkpoint_finetune",
            "seed": seed,
            "source_checkpoint": str(checkpoint_path),
            "source_checkpoint_sha256": checkpoint_hash,
            "source_local_step": checkpoint.get("local_step"),
            "note": (
                "Fine-tune the same dense large-student architecture "
                "from the selected DEV checkpoint."
            ),
        }
        _put(output / "initialization_report.json", report)
        _save(output / "init.pt", {
            "model_state_dict": model.state_dict(),
            "stage_id": stage_id,
            "seed": seed,
            "initialization": report["method"],
            "source_checkpoint_sha256": checkpoint_hash,
        })
        return model.to(device), None, device, report

    if init_mode != "function_preserving_widen":
        raise ValueError(f"unknown initialization mode: {init_mode}")

    checkpoint_path = Path(
        cfg["protected_baseline"]["frozen_d4_checkpoint"]
    )
    freeze_path = Path(
        cfg["protected_baseline"]["freeze_manifest"]
    )
    if not freeze_path.is_absolute():
        freeze_path = ROOT / freeze_path
    freeze = json.loads(freeze_path.read_text())
    expected_hash = freeze["checkpoint_sha256"]
    actual_hash = sha256(checkpoint_path)
    if actual_hash != expected_hash:
        raise RuntimeError("frozen D4 checkpoint hash mismatch")

    compact_id = stage_ids(cfg)[-1]
    compact = K210MixVPR(spec_for_stage(cfg, compact_id)).eval()
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    load_legacy_d4_state(
        compact, checkpoint["model_state_dict"]
    )

    model = K210MixVPR(spec_for_stage(cfg, stage_id)).eval()
    transfer = widen_preserving_function(model, compact)

    generator = torch.Generator().manual_seed(seed + 1)
    sample = torch.randn(
        2, 3, *model.spec.input_hw, generator=generator
    )
    with torch.inference_mode():
        compact_raw = compact.raw(sample)
        model_raw = model.raw(sample)
        compact_norm = compact(sample)
        model_norm = model(sample)
    torch.testing.assert_close(
        model_raw, compact_raw, rtol=1e-5, atol=1e-6
    )
    torch.testing.assert_close(
        model_norm, compact_norm, rtol=1e-5, atol=1e-6
    )
    report = {
        "method": "function_preserving_widen_from_frozen_d4",
        "source_stage": compact_id,
        "source_checkpoint": str(checkpoint_path),
        "source_checkpoint_sha256": actual_hash,
        "raw_max_abs": float((model_raw - compact_raw).abs().max()),
        "normalized_max_abs": float(
            (model_norm - compact_norm).abs().max()
        ),
        "transfer": transfer,
    }
    _put(output / "initialization_report.json", report)
    _save(output / "init.pt", {
        "model_state_dict": model.state_dict(),
        "stage_id": stage_id,
        "seed": seed,
        "initialization": report["method"],
        "source_checkpoint_sha256": actual_hash,
    })
    return model.to(device), None, device, report

def initialize_pruned_stage(
    cfg: dict,
    parent_checkpoint: Path,
    parent_stage_id: str,
    child_stage_id: str,
    output: Path,
):
    device = _setup(int(cfg["seed"]))
    parent = K210MixVPR(
        spec_for_stage(cfg, parent_stage_id)
    ).to(device)
    checkpoint = torch.load(
        parent_checkpoint, map_location="cpu", weights_only=False
    )
    parent.load_state_dict(
        checkpoint["model_state_dict"], strict=True
    )
    parent.eval()
    parent.requires_grad_(False)

    child = K210MixVPR(
        spec_for_stage(cfg, child_stage_id)
    ).to(device)
    action = stage_config(cfg, child_stage_id)["action"]
    report = prune_one_axis(parent, child, action)
    _save(output / "init.pt", {
        "model_state_dict": child.state_dict(),
        "stage_id": child_stage_id,
        "parent_stage_id": parent_stage_id,
        "parent_checkpoint": str(parent_checkpoint),
        "parent_checkpoint_sha256": sha256(parent_checkpoint),
    })
    _put(output / "pruning_report.json", report)
    return child, parent, device, report

def _checkpoint(path, model, optimizer, scheduler, scaler, state):
    _save(path, {
        "model_state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "state": state,
        "rng": _rng_state(),
    })

def train_stage(
    cfg: dict,
    stage_id: str,
    model,
    parent_model,
    device,
    data,
    plan,
    start: int,
    end: int,
    output: Path,
    initial_metrics: dict | None = None,
    stop_after_local_step: int | None = None,
):
    train_cfg = cfg["training"]
    bank, bank_offset = _teacher_bank(data)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg["lr"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=end - start,
        eta_min=float(train_cfg["eta_min"]),
    )
    scaler = torch.amp.GradScaler("cuda", init_scale=1024)
    state = {
        "stage_id": stage_id,
        "local_step": 0,
        "global_start": start,
        "global_end": end,
        "best_step": 0,
        "best_metrics": None,
        "evaluations": [],
        "trace": [],
        "status": "RUNNING",
    }
    if initial_metrics is not None:
        initial_kind = (
            "first_stage_initialization"
            if parent_model is None
            else "post_prune_pre_recovery"
        )
        state["best_metrics"] = initial_metrics
        state["evaluations"].append({
            "local_step": 0,
            "global_step": start,
            "metrics": initial_metrics,
            "kind": initial_kind,
        })
        _save(output / "best.pt", {
            "model_state_dict": model.state_dict(),
            "stage_id": stage_id,
            "local_step": 0,
            "global_step": start,
            "dev_metrics": initial_metrics,
            "selection_reason": initial_kind,
        })

    latest = output / "latest.pt"
    if latest.exists():
        checkpoint = torch.load(
            latest, map_location="cpu", weights_only=False
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        state = checkpoint["state"]
        _restore_rng(checkpoint["rng"])

    stage_total = end - start
    target_local_step = stage_total
    if stop_after_local_step is not None:
        stop_after_local_step = int(stop_after_local_step)
        if not 0 < stop_after_local_step <= stage_total:
            raise ValueError("stop_after_local_step must be within the stage budget")
        if state["local_step"] > stop_after_local_step:
            raise RuntimeError("checkpoint is already beyond requested probe stop")
        target_local_step = stop_after_local_step
    state["status"] = "RUNNING"

    rows = data["rows"]
    accumulation = int(train_cfg["accumulation"])
    started = time.time()
    while state["local_step"] < target_local_step:
        global_step = start + state["local_step"]
        model.train()
        microbatches = plan[global_step]
        if len(microbatches) != accumulation:
            raise RuntimeError("train plan accumulation mismatch")

        optimizer.zero_grad(set_to_none=True)
        sums = {
            "task": 0.0,
            "teacher_kd": 0.0,
            "parent_kd": 0.0,
            "task_active_microbatches": 0,
            "mined_positive_pairs": 0,
            "mined_negative_pairs": 0,
        }
        for item in microbatches:
            x, labels = training_batch(item, rows, 240, device)
            teacher = _teacher_batch(
                item, bank, bank_offset, device
            )
            with torch.amp.autocast("cuda", dtype=torch.float16):
                student = model(x)
            student = F.normalize(student.float(), dim=1)
            task, miner_stats = retrieval_loss_with_stats(
                student, labels
            )
            teacher_kd = ranking_kd(
                student, teacher, labels
            )
            parent_kd = student.new_zeros(())
            if parent_model is not None:
                with torch.inference_mode():
                    parent_desc = parent_model(x).float()
                parent_kd = ranking_kd(
                    student, parent_desc, labels
                )

            loss = (
                float(train_cfg["task_weight"]) * task
                + float(train_cfg["teacher_kd_weight"]) * teacher_kd
                + float(train_cfg["parent_kd_weight"]) * parent_kd
            ) / accumulation
            scaler.scale(loss).backward()
            sums["task"] += task.item() / accumulation
            sums["teacher_kd"] += teacher_kd.item() / accumulation
            sums["parent_kd"] += parent_kd.item() / accumulation
            sums["task_active_microbatches"] += int(
                task.detach().item() > 0.0
            )
            sums["mined_positive_pairs"] += int(
                miner_stats["positive_pairs"]
            )
            sums["mined_negative_pairs"] += int(
                miner_stats["negative_pairs"]
            )

        sums["task_active_fraction"] = (
            sums["task_active_microbatches"] / accumulation
        )

        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), 10.0
        )
        if not torch.isfinite(grad_norm):
            raise RuntimeError("non-finite gradient norm")
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        state["local_step"] += 1
        state["trace"].append({
            "local_step": state["local_step"],
            "global_step": global_step + 1,
            **sums,
            "grad_norm": float(grad_norm),
            "lr": optimizer.param_groups[0]["lr"],
        })

        if state["local_step"] % 50 == 0:
            _checkpoint(
                latest, model, optimizer,
                scheduler, scaler, state
            )
            window = state["trace"][-50:]
            def _window_mean(key):
                return sum(float(x[key]) for x in window) / len(window)
            print(
                "TRAIN", stage_id, state["local_step"],
                "task", round(_window_mean("task"), 6),
                "teacher_kd", round(_window_mean("teacher_kd"), 6),
                "parent_kd", round(_window_mean("parent_kd"), 6),
                "task_active", round(
                    _window_mean("task_active_fraction"), 4
                ),
                "pos_pairs", round(
                    _window_mean("mined_positive_pairs"), 2
                ),
                "neg_pairs", round(
                    _window_mean("mined_negative_pairs"), 2
                ),
                "grad", round(_window_mean("grad_norm"), 4),
                "lr", optimizer.param_groups[0]["lr"],
                flush=True,
            )

        if state["local_step"] % int(train_cfg["eval_interval"]) == 0:
            metrics = evaluate(model, data, device)
            state["evaluations"].append({
                "local_step": state["local_step"],
                "global_step": global_step + 1,
                "metrics": metrics,
            })
            if (
                state["best_metrics"] is None
                or _metric_key(metrics) > _metric_key(state["best_metrics"])
            ):
                state["best_step"] = state["local_step"]
                state["best_metrics"] = metrics
                _save(output / "best.pt", {
                    "model_state_dict": model.state_dict(),
                    "stage_id": stage_id,
                    "local_step": state["local_step"],
                    "global_step": global_step + 1,
                    "dev_metrics": metrics,
                })
            _put(output / "progress.json", state)
            print(
                "DEV", stage_id, state["local_step"],
                metrics["Indoor"]["R@1"],
                metrics["Project"]["R@1"],
                flush=True,
            )

    state["last_elapsed_seconds"] = time.time() - started
    if target_local_step < stage_total:
        state["status"] = "PROBE_COMPLETE"
        state["probe_target_step"] = target_local_step
        _checkpoint(
            latest, model, optimizer,
            scheduler, scaler, state
        )
        _put(output / "progress.json", state)
        return state

    state["status"] = "COMPLETE"
    state["elapsed_seconds"] = state["last_elapsed_seconds"]
    _put(output / "training.json", state)
    if not (output / "best.pt").exists():
        metrics = evaluate(model, data, device)
        _save(output / "best.pt", {
            "model_state_dict": model.state_dict(),
            "stage_id": stage_id,
            "local_step": state["local_step"],
            "global_step": end,
            "dev_metrics": metrics,
        })
    return state

def run_stage(
    cfg: dict,
    run_root: Path,
    stage_id: str,
    stop_after_local_step: int | None = None,
):
    validate_progression(cfg)
    ids = stage_ids(cfg)
    index = ids.index(stage_id)
    output = run_root / "stages" / stage_id
    output.mkdir(parents=True, exist_ok=True)
    plan, data = master_plan(cfg, run_root)
    start, end = plan_slice(cfg, stage_id)

    pre_recovery_metrics = None
    parent_metrics = None
    pruning_delta_pp = None

    if index == 0:
        model, parent, device, pruning = initialize_first_stage(
            cfg, stage_id, output
        )
        parent_stage_id = None
        parent_checkpoint = None
        pre_recovery_metrics = evaluate(model, data, device)
        _put(output / "initial_metrics.json", {
            "stage_id": stage_id,
            "kind": pruning.get("method", "first_stage_initialization"),
            "metrics": pre_recovery_metrics,
        })
    else:
        parent_stage_id = ids[index - 1]
        parent_checkpoint = (
            run_root / "stages" / parent_stage_id / "best.pt"
        )
        if not parent_checkpoint.is_file():
            raise RuntimeError(
                f"previous stage not complete: {parent_checkpoint}"
            )
        model, parent, device, pruning = initialize_pruned_stage(
            cfg,
            parent_checkpoint,
            parent_stage_id,
            stage_id,
            output,
        )
        parent_record = torch.load(
            parent_checkpoint,
            map_location="cpu",
            weights_only=False,
        )
        parent_metrics = parent_record.get("dev_metrics")
        if parent_metrics is None:
            raise RuntimeError(
                "parent best checkpoint is missing DEV metrics"
            )
        pre_recovery_metrics = evaluate(
            model, data, device
        )
        pruning_delta_pp = {
            domain: 100.0 * (
                pre_recovery_metrics[domain]["R@1"]
                - parent_metrics[domain]["R@1"]
            )
            for domain in ("Indoor", "Project", "Overall")
        }
        _put(output / "pre_recovery_metrics.json", {
            "parent_stage_id": parent_stage_id,
            "child_stage_id": stage_id,
            "parent_metrics": parent_metrics,
            "post_prune_metrics": pre_recovery_metrics,
            "delta_r1_pp": pruning_delta_pp,
        })

    frozen_evidence = {
        "train_plan_sha256": sha256(
            run_root / "shared" / "train_plan.npy"
        ),
        "metadata_sha256": sha256(cfg["data"]["metadata"]),
        "teacher_manifest_sha256": sha256(
            cfg["data"]["teacher_bank_manifest"]
        ),
        "teacher_bank_sha256": data[
            "teacher_bank_manifest"
        ]["descriptors_sha256"],
    }
    if cfg.get("initialization", {}).get(
        "mode", "function_preserving_widen"
    ) == "function_preserving_widen":
        frozen_evidence["frozen_d4_checkpoint_sha256"] = sha256(
            cfg["protected_baseline"]["frozen_d4_checkpoint"]
        )
    elif cfg.get("initialization", {}).get("mode") == "checkpoint":
        source = Path(cfg["initialization"]["checkpoint"]).resolve()
        frozen_evidence["initial_checkpoint_sha256"] = sha256(source)

    protocol = {
        "stage_id": stage_id,
        "stage_index": index,
        "hypothesis": cfg["hypotheses"][stage_id],
        "seed": int(cfg["seed"]),
        "frozen_evidence": frozen_evidence,
        "training_config": dict(cfg["training"]),
        "sampler_config": dict(cfg.get("sampler", {"mode": "p2k4"})),
        "compression_config": dict(cfg["compression"]),
        "model_spec": {
            "widths": list(model.spec.widths),
            "mixer_hidden": model.spec.mixer_hidden,
            "mixer_depth": model.spec.mixer_depth,
            "projection": model.spec.projection,
            "descriptor_dim": model.spec.descriptor_dim,
            "input_hw": list(model.spec.input_hw),
        },
        "action": stage_config(cfg, stage_id)["action"],
        "global_plan_slice": [start, end],
        "parent_stage_id": parent_stage_id,
        "parent_checkpoint": (
            str(parent_checkpoint) if parent_checkpoint else None
        ),
        "parent_checkpoint_sha256": (
            sha256(parent_checkpoint) if parent_checkpoint else None
        ),
        "teacher_signal": "frozen 4096D teacher ranking bank",
        "local_parent_signal": (
            "previous checkpoint online ranking"
            if parent is not None else None
        ),
        "descriptor_interface": "512D invariant",
        "mixer_depth": "D4 invariant",
        "test_access": False,
        "initialization": pruning if index == 0 else None,
        "pruning": pruning if index > 0 else None,
        "initial_metrics": pre_recovery_metrics if index == 0 else None,
        "pre_recovery_metrics": pre_recovery_metrics if index > 0 else None,
        "pruning_delta_r1_pp": pruning_delta_pp,
    }
    _put(output / "protocol.json", protocol)

    return train_stage(
        cfg=cfg,
        stage_id=stage_id,
        model=model,
        parent_model=parent,
        device=device,
        data=data,
        plan=plan,
        start=start,
        end=end,
        output=output,
        initial_metrics=pre_recovery_metrics,
        stop_after_local_step=stop_after_local_step,
    )

def next_stage(cfg: dict, run_root: Path) -> str | None:
    for stage_id in stage_ids(cfg):
        best = run_root / "stages" / stage_id / "best.pt"
        training = run_root / "stages" / stage_id / "training.json"
        if not best.exists() or not training.exists():
            return stage_id
        if json.loads(training.read_text()).get("status") != "COMPLETE":
            return stage_id
    return None

def lineage_status(cfg: dict, run_root: Path) -> list[dict]:
    rows = []
    for stage_id in stage_ids(cfg):
        folder = run_root / "stages" / stage_id
        training_path = folder / "training.json"
        best_path = folder / "best.pt"
        row = {
            "stage": stage_id,
            "action": stage_config(cfg, stage_id)["action"],
            "spec": {
                "widths": list(spec_for_stage(cfg, stage_id).widths),
                "hidden": spec_for_stage(cfg, stage_id).mixer_hidden,
                "depth": spec_for_stage(cfg, stage_id).mixer_depth,
                "descriptor": spec_for_stage(cfg, stage_id).descriptor_dim,
            },
            "status": "NOT_STARTED",
        }
        if training_path.exists():
            state = json.loads(training_path.read_text())
            row["status"] = state.get("status", "UNKNOWN")
            row["best_step"] = state.get("best_step")
            row["best_metrics"] = state.get("best_metrics")
        elif (folder / "progress.json").exists():
            state = json.loads((folder / "progress.json").read_text())
            row["status"] = state.get("status", "RUNNING")
            row["local_step"] = state.get("local_step")
        if best_path.exists():
            row["best_checkpoint_sha256"] = sha256(best_path)
        rows.append(row)
    return rows
