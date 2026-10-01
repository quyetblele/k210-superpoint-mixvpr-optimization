"""Resumable CUDA-only GSV strengthening for the official MixVPR teacher."""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.autograd.graph import save_on_cpu

ROOT = Path("/home/quyet/k210_lab")
sys.path.insert(0, str(ROOT / "experiments" / "edge_vpr_students"))
import success_first_train_core as core


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=30000)
    parser.add_argument("--chunk-steps", type=int, default=250)
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--dev-max-places", type=int, default=1000)
    parser.add_argument("--physical-places", type=int, default=4)
    parser.add_argument("--accumulation", type=int, default=8)
    parser.add_argument("--views-per-place", type=int, default=4)
    parser.add_argument("--hard-bank", type=Path,
                        help="Immutable SQLite bank with place-level global hard negatives.")
    parser.add_argument("--checkpoint-every", type=int, default=1)
    parser.add_argument("--max-steps-this-invocation", type=int,
                        help="Run a bounded resumable chunk then exit cleanly.")
    parser.add_argument("--stage", default="teacher_gsv")
    parser.add_argument("--database", type=Path, default=core.GSV_INDEX)
    parser.add_argument("--dev-database", type=Path)
    parser.add_argument("--init", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/success_first/strong_teacher/gsv")
    args = parser.parse_args()
    device = core.require_cuda(); args.output.mkdir(parents=True, exist_ok=True)
    random.seed(core.SEED); np.random.seed(core.SEED); torch.manual_seed(core.SEED); torch.cuda.manual_seed_all(core.SEED)
    index = core.PlaceIndex(args.database)
    hard_neighbors = core.load_hard_neighbors(args.hard_bank) if args.hard_bank else None
    dev_database = args.dev_database or args.database
    dev_index = core.PlaceIndex(dev_database)
    model = core.build_teacher().to(device)
    if args.init:
        initial = torch.load(args.init, map_location="cpu", weights_only=False)
        model.load_state_dict(initial["model_state_dict"], strict=True)
    optimizer = torch.optim.AdamW([
        {"params": [p for p in model.backbone.parameters() if p.requires_grad], "lr": 1e-5},
        {"params": model.aggregator.parameters(), "lr": 1e-4},
    ], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda")
    state = {"optimizer_step": 0, "best_key": [-1.0, -1.0, -1.0], "best_metrics": None,
             "trace": [], "plan_chunk": None, "micro_cursor": 0}
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"]); optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"]); scaler.load_state_dict(checkpoint["scaler_state_dict"])
        state.update(checkpoint["cursor"]); core.restore_rng(checkpoint["rng_state"])

    if state["optimizer_step"] == 0 and state["best_metrics"] is None:
        baseline = core.dev_metrics(model, dev_index, 320, device,
                                    max_places=args.dev_max_places)
        state["best_metrics"] = baseline
        state["best_key"] = [baseline["R@1"], baseline["R@5"],
                             baseline["positive_hardest_negative_margin"]]
        core.atomic_torch(args.output / "best.pt", {"model_state_dict": model.state_dict(),
                          "step": 0, "dev_metrics": baseline})

    accumulation = args.accumulation
    while state["optimizer_step"] < args.steps:
        chunk_start = (state["optimizer_step"] // args.chunk_steps) * args.chunk_steps
        chunk_end = min(chunk_start + args.chunk_steps, args.steps)
        plan_path = args.output / f"plan_{chunk_start:06d}_{chunk_end:06d}.pt"
        expected_micro = (chunk_end - chunk_start) * accumulation
        if not plan_path.exists():
            stage_seed = int.from_bytes(args.stage.encode()[:4].ljust(4, b"_"), "little")
            plan = core.make_plan(index, expected_micro, core.SEED + stage_seed + chunk_start,
                                  views=args.views_per_place,
                                  places_per_step=args.physical_places,
                                  hard_neighbors=hard_neighbors)
            core.atomic_torch(plan_path, {"schema_version": 1, "stage": args.stage,
                                         "start_step": chunk_start, "end_step": chunk_end, "plan": plan})
        plan_payload = torch.load(plan_path, map_location="cpu", weights_only=False)
        plan = plan_payload["plan"]
        if len(plan) != expected_micro:
            raise RuntimeError("persisted teacher plan length mismatch")
        local_step = state["optimizer_step"] - chunk_start
        micro_start = local_step * accumulation + state.get("micro_cursor", 0)
        model.train(); optimizer.zero_grad(set_to_none=True); task_sum = 0.0
        for micro_index in range(micro_start, len(plan)):
            images, labels = core.batch_from_plan(plan[micro_index], 320, device)
            with save_on_cpu(pin_memory=False):
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    descriptors = model(images)
                    task = core.retrieval_loss(descriptors, labels)
                    loss = task / accumulation
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite teacher loss at micro {micro_index}")
            scaler.scale(loss).backward()
            task_sum += float(task.detach())
            state["micro_cursor"] = micro_index % accumulation + 1
            if state["micro_cursor"] == accumulation:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
                if not torch.isfinite(grad_norm):
                    raise RuntimeError("non-finite teacher gradients")
                scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True); scheduler.step()
                state["optimizer_step"] += 1; state["micro_cursor"] = 0
                state["trace"].append({"step": state["optimizer_step"],
                                       "task_loss": task_sum / accumulation,
                                       "lr_backbone": optimizer.param_groups[0]["lr"],
                                       "lr_aggregator": optimizer.param_groups[1]["lr"]})
                task_sum = 0.0
                if state["optimizer_step"] % args.eval_every == 0 or state["optimizer_step"] == args.steps:
                    metrics = core.dev_metrics(model, dev_index, 320, device,
                                               max_places=args.dev_max_places)
                    key = [metrics["R@1"], metrics["R@5"], metrics["positive_hardest_negative_margin"]]
                    if key > state["best_key"]:
                        state["best_key"] = key; state["best_metrics"] = metrics
                        core.atomic_torch(args.output / "best.pt", {"model_state_dict": model.state_dict(),
                                          "step": state["optimizer_step"], "dev_metrics": metrics})
                    model.train()
                checkpoint = {"schema_version": 2, "stage": args.stage,
                    "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(), "scaler_state_dict": scaler.state_dict(),
                    "cursor": state, "rng_state": core.rng_state(), "plan_path": str(plan_path),
                    "plan_sha256": core.sha256(plan_path), "config": {"steps": args.steps,
                    "physical_places": args.physical_places, "views_per_place": args.views_per_place,
                    "accumulation": accumulation,
                    "effective_places": args.physical_places * accumulation,
                    "effective_images": args.physical_places * args.views_per_place * accumulation,
                    "input": [320, 320], "database": str(args.database),
                    "dev_database": str(dev_database), "gt1_used": False}}
                # The teacher checkpoint includes model, Adam moments and AMP
                # state.  Writing it every step creates severe disk pressure;
                # a 50-step atomic cadence remains safely resumable.
                if state["optimizer_step"] % args.checkpoint_every == 0 or state["optimizer_step"] == args.steps:
                    core.atomic_torch(args.output / "latest.pt", checkpoint)
                if (args.max_steps_this_invocation is not None
                        and state["optimizer_step"] % args.max_steps_this_invocation == 0):
                    print(json.dumps({"status": "PARTIAL", "step": state["optimizer_step"],
                                      "checkpoint": str(args.output / "latest.pt")}))
                    return
        state["plan_chunk"] = None
    final = args.output / "final.pt"; core.atomic_torch(final, checkpoint)
    report = {"status": "COMPLETE", "checkpoint": str(final), "sha256": core.sha256(final),
              "steps": state["optimizer_step"], "best_metrics": state["best_metrics"],
              "official_base_sha256": core.sha256(core.OFFICIAL), "gt1_used": False}
    core.atomic_json(args.output / "report.json", report); print(json.dumps(report))


if __name__ == "__main__":
    main()
