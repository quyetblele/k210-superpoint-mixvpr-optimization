from __future__ import annotations
import json
import os
import fcntl
from pathlib import Path
import shutil
import subprocess
import sys
import time
import torch

ROOT = Path("/home/quyet/k210_lab")
RUNNER = ROOT / "mixvpr/run"
TEMPLATE = ROOT / "mixvpr/configs/v3_large_phase2_p4k2.json"
PHASE1_BEST = ROOT / "mixvpr/artifacts/v3_random_large_pruning_final/stages/00_large/best.pt"
AUTO = ROOT / "mixvpr/artifacts/v3_auto_controller"
CONFIG_DIR = AUTO / "configs"
STATE_PATH = AUTO / "controller_state.json"
LOG_PATH = AUTO / "controller.log"
BLOCK = 250
BUDGET = 1500
PLATEAU_EPS = 0.002
REGRESSION_EPS = 0.01
POOL_ORDER = [16, 12, 8]
LR_ORDER = [5e-5, 3e-5]


def log(message: str) -> None:
    AUTO.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{stamp}] {message}"
    print(line, flush=True)
    with LOG_PATH.open("a") as f:
        f.write(line + "\n")


def read_json(path: Path, default=None):
    if not path.exists():
        return default
    return json.loads(path.read_text())


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def checkpoint_metrics(path: Path) -> dict:
    record = torch.load(path, map_location="cpu", weights_only=False)
    return record["dev_metrics"]


def metric_key(metrics: dict) -> tuple:
    x = metrics["Overall"]
    return (x["R@1"], x["R@5"], x["R@10"], x["margin"])


def overall_r1(metrics: dict) -> float:
    return float(metrics["Overall"]["R@1"])


def recent_task_active(progress: dict, window: int = BLOCK) -> float:
    trace = progress.get("trace", [])
    if not trace:
        return 0.0
    rows = trace[-min(window, len(trace)):]
    return sum(float(x["task_active_fraction"]) for x in rows) / len(rows)


def evaluated_r1s(progress: dict) -> list[float]:
    out = []
    for row in progress.get("evaluations", []):
        if int(row.get("local_step", 0)) <= 0:
            continue
        out.append(overall_r1(row["metrics"]))
    return out


def make_config(name: str, source: Path, pool: int, lr: float,
                kd: float = 0.2) -> Path:
    cfg = json.loads(TEMPLATE.read_text())
    cfg["name"] = name
    cfg["initialization"] = {
        "mode": "checkpoint",
        "checkpoint": str(source.resolve()),
    }
    cfg["sampler"] = {
        "mode": "p4k2",
        "project_probability": 0.1,
        "neighbor_pool": int(pool),
    }
    cfg["training"]["parent_probe_updates"] = BUDGET
    cfg["training"]["parent_max_updates"] = BUDGET
    cfg["training"]["lr"] = float(lr)
    cfg["training"]["eta_min"] = 3e-6
    cfg["training"]["teacher_kd_weight"] = float(kd)
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    path = CONFIG_DIR / f"{name}.json"
    write_json(path, cfg)
    return path


def run_cli(cfg: Path, run_root: Path, *args: str) -> None:
    env = os.environ.copy()
    env["MIXVPR_CONFIG"] = str(cfg)
    env["MIXVPR_RUN_ROOT"] = str(run_root)
    command = [str(RUNNER), *args]
    log("RUN " + " ".join(command))
    result = subprocess.run(command, cwd=ROOT, env=env)
    if result.returncode != 0:
        raise RuntimeError(
            f"command failed rc={result.returncode}: {' '.join(command)}"
        )


def diagnostic(cfg: Path, run_root: Path) -> dict:
    report = run_root / "diagnostics/kd_gradient/report.json"
    if not report.exists():
        run_cli(cfg, run_root, "kd-diagnostic", "--batches", "8")
    return read_json(report, {})


def train_to(cfg: Path, run_root: Path, target: int) -> dict:
    run_cli(
        cfg, run_root, "train", "00_large",
        "--stop-after-local-step", str(target),
    )
    progress = read_json(
        run_root / "stages/00_large/progress.json", {}
    )
    if int(progress.get("local_step", -1)) != target:
        raise RuntimeError(f"expected local_step={target}, got {progress.get('local_step')}")
    return progress


def summarize(progress: dict, source_metrics: dict) -> dict:
    evals = evaluated_r1s(progress)
    current = evals[-1] if evals else overall_r1(source_metrics)
    best = overall_r1(progress.get("best_metrics") or source_metrics)
    active = recent_task_active(progress)
    recent3 = evals[-3:]
    plateau = (
        len(recent3) == 3
        and max(recent3) - min(recent3) < PLATEAU_EPS
    )
    regression = current < overall_r1(source_metrics) - REGRESSION_EPS
    return {
        "local_step": int(progress.get("local_step", 0)),
        "current_r1": current,
        "best_r1": best,
        "source_r1": overall_r1(source_metrics),
        "improvement": best - overall_r1(source_metrics),
        "task_active": active,
        "recent_r1": recent3,
        "plateau": plateau,
        "regression": regression,
    }


def next_knobs(summary: dict, pool: int, lr: float):
    if summary["regression"] and summary["task_active"] >= 0.4:
        if lr > LR_ORDER[-1]:
            return pool, LR_ORDER[-1], "regression_with_active_task_reduce_lr"
    if summary["plateau"] and summary["task_active"] < 0.35:
        idx = POOL_ORDER.index(pool)
        if idx + 1 < len(POOL_ORDER):
            return POOL_ORDER[idx + 1], lr, "plateau_low_task_harder_negatives"
    if summary["plateau"] and summary["task_active"] >= 0.35:
        if lr > LR_ORDER[-1]:
            return pool, LR_ORDER[-1], "plateau_with_active_task_reduce_lr"
    return None


def new_experiment(state: dict, pool: int, lr: float, reason: str) -> dict:
    index = int(state.get("experiment_counter", 1)) + 1
    name = f"exp{index:03d}_p4k2_pool{pool}_lr{lr:.0e}".replace("-", "m")
    source = Path(state["champion_checkpoint"])
    cfg = make_config(name, source, pool, lr)
    run_root = AUTO / "experiments" / name
    exp = {
        "name": name,
        "config": str(cfg),
        "run_root": str(run_root),
        "source_checkpoint": str(source),
        "pool": int(pool),
        "lr": float(lr),
        "kd": 0.2,
        "reason": reason,
        "last_processed_step": 0,
    }
    state["experiment_counter"] = index
    state["current"] = exp
    state.setdefault("history", []).append({
        "event": "branch",
        "experiment": name,
        "reason": reason,
        "source_checkpoint": str(source),
        "pool": pool,
        "lr": lr,
    })
    write_json(STATE_PATH, state)
    log(f"BRANCH {name}: {reason}; pool={pool}, lr={lr}")
    return exp


def initialize_state() -> dict:
    phase1_metrics = checkpoint_metrics(PHASE1_BEST)
    current_cfg = ROOT / "mixvpr/configs/v3_large_phase2_p4k2.json"
    current_run = ROOT / "mixvpr/artifacts/v3_large_phase2_p4k2"
    state = {
        "status": "DENSE_SEARCH",
        "experiment_counter": 1,
        "champion_checkpoint": str(PHASE1_BEST),
        "champion_metrics": phase1_metrics,
        "history": [],
        "current": {
            "name": "exp001_p4k2_pool16_lr5em05",
            "config": str(current_cfg),
            "run_root": str(current_run),
            "source_checkpoint": str(PHASE1_BEST),
            "pool": 16,
            "lr": 5e-5,
            "kd": 0.2,
            "reason": "adopt_existing_phase2",
            "last_processed_step": 0,
        },
    }
    write_json(STATE_PATH, state)
    return state


def promote_if_better(state: dict, run_root: Path, progress: dict) -> bool:
    candidate = progress.get("best_metrics")
    if not candidate:
        return False
    if metric_key(candidate) <= metric_key(state["champion_metrics"]):
        return False
    checkpoint = run_root / "stages/00_large/best.pt"
    state["champion_checkpoint"] = str(checkpoint)
    state["champion_metrics"] = candidate
    state.setdefault("history", []).append({
        "event": "promote",
        "checkpoint": str(checkpoint),
        "metrics": candidate,
    })
    write_json(STATE_PATH, state)
    log(
        "PROMOTE champion Overall R@1="
        f"{overall_r1(candidate):.6f}"
    )
    return True


def final_dense_record(state: dict) -> None:
    write_json(AUTO / "dense_champion.json", {
        "checkpoint": state["champion_checkpoint"],
        "metrics": state["champion_metrics"],
        "history": state.get("history", []),
    })
    log(
        "DENSE CHAMPION Overall R@1="
        f"{overall_r1(state['champion_metrics']):.6f}"
    )


def prepare_pruning(state: dict) -> tuple[Path, Path]:
    cfg = json.loads(TEMPLATE.read_text())
    cfg["name"] = "mixvpr_v3_auto_pruning"
    cfg["initialization"] = {
        "mode": "checkpoint",
        "checkpoint": state["champion_checkpoint"],
    }
    cfg["training"]["parent_probe_updates"] = BUDGET
    cfg["training"]["parent_max_updates"] = BUDGET
    cfg["training"]["lr"] = 5e-5
    cfg["training"]["teacher_kd_weight"] = 0.2
    cfg_path = CONFIG_DIR / "auto_pruning.json"
    write_json(cfg_path, cfg)
    run_root = AUTO / "pruning"
    stage0 = run_root / "stages/00_large"
    stage0.mkdir(parents=True, exist_ok=True)
    shutil.copy2(state["champion_checkpoint"], stage0 / "best.pt")
    write_json(stage0 / "training.json", {
        "status": "COMPLETE",
        "stage_id": "00_large",
        "seeded_from_dense_champion": state["champion_checkpoint"],
        "best_metrics": state["champion_metrics"],
    })
    return cfg_path, run_root


def run_pruning(state: dict) -> None:
    cfg_path, run_root = prepare_pruning(state)
    cfg = read_json(cfg_path)
    summary = []
    for stage in cfg["stages"][1:]:
        stage_id = stage["id"]
        training = run_root / "stages" / stage_id / "training.json"
        if not training.exists() or read_json(training, {}).get("status") != "COMPLETE":
            run_cli(
                cfg_path, run_root, "train", stage_id,
                "--stop-after-local-step",
                str(cfg["training"]["recovery_max_updates"]),
            )
        record = read_json(training, {})
        metrics = record.get("best_metrics")
        if metrics is None:
            best = torch.load(
                run_root / "stages" / stage_id / "best.pt",
                map_location="cpu", weights_only=False,
            )
            metrics = best["dev_metrics"]
        summary.append({
            "stage": stage_id,
            "metrics": metrics,
            "overall_r1": overall_r1(metrics),
        })
        write_json(run_root / "pruning_summary.json", summary)
        log(
            f"PRUNE {stage_id} complete Overall R@1="
            f"{overall_r1(metrics):.6f}"
        )

    run_cli(cfg_path, run_root, "feasibility", "all")
    state["status"] = "PRUNING_COMPLETE"
    state["pruning_config"] = str(cfg_path)
    state["pruning_root"] = str(run_root)
    write_json(STATE_PATH, state)
    log("PRUNING + STRUCTURAL K210 FEASIBILITY COMPLETE")


def dense_search(state: dict) -> dict:
    while state.get("status") == "DENSE_SEARCH":
        exp = state["current"]
        cfg = Path(exp["config"])
        run_root = Path(exp["run_root"])
        source = Path(exp["source_checkpoint"])
        source_metrics = checkpoint_metrics(source)
        progress_path = run_root / "stages/00_large/progress.json"
        progress = read_json(progress_path, {})
        local_step = int(progress.get("local_step", 0))

        if local_step <= int(exp.get("last_processed_step", 0)):
            target = min(local_step + BLOCK, BUDGET)
            if target <= local_step:
                target = BUDGET
            if local_step == 0:
                diag = diagnostic(cfg, run_root)
                log(f"DIAG {exp['name']} heuristics={diag.get('heuristics', {})}")
            progress = train_to(cfg, run_root, target)
            local_step = target

        summary = summarize(progress, source_metrics)
        promote_if_better(state, run_root, progress)
        exp["last_processed_step"] = local_step
        state["current"] = exp
        state.setdefault("history", []).append({
            "event": "eval",
            "experiment": exp["name"],
            **summary,
        })
        write_json(STATE_PATH, state)
        log(
            f"EVAL {exp['name']} step={local_step} "
            f"current={summary['current_r1']:.6f} "
            f"best={summary['best_r1']:.6f} "
            f"active={summary['task_active']:.3f} "
            f"plateau={summary['plateau']} "
            f"regression={summary['regression']}"
        )

        branch = next_knobs(summary, int(exp["pool"]), float(exp["lr"]))
        if branch is not None and state["experiment_counter"] < 5:
            pool, lr, reason = branch
            new_experiment(state, pool, lr, reason)
            continue

        if local_step < BUDGET:
            continue

        if state["experiment_counter"] < 5:
            if summary["task_active"] < 0.35 and int(exp["pool"]) > 8:
                idx = POOL_ORDER.index(int(exp["pool"]))
                pool = POOL_ORDER[min(idx + 1, len(POOL_ORDER) - 1)]
                new_experiment(state, pool, float(exp["lr"]),
                               "budget_end_low_task_harder_negatives")
                continue
            if summary["improvement"] < PLATEAU_EPS and float(exp["lr"]) > 3e-5:
                new_experiment(state, int(exp["pool"]), 3e-5,
                               "budget_end_no_gain_reduce_lr")
                continue

        state["status"] = "DENSE_COMPLETE"
        write_json(STATE_PATH, state)
        break

    return state


def main() -> None:
    AUTO.mkdir(parents=True, exist_ok=True)
    lock_path = AUTO / "controller.lock"
    lock_file = lock_path.open("w")
    try:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("controller already running")

    state = read_json(STATE_PATH)
    if state is None:
        state = initialize_state()
        log("INITIALIZED controller from phase-1 champion")

    if state.get("status") == "DENSE_SEARCH":
        state = dense_search(state)

    if state.get("status") == "DENSE_COMPLETE":
        final_dense_record(state)
        state["status"] = "PRUNING"
        write_json(STATE_PATH, state)

    if state.get("status") == "PRUNING":
        run_pruning(state)

    log(f"CONTROLLER STOP status={state.get('status')}")


if __name__ == "__main__":
    main()
