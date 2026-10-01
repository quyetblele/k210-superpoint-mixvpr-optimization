from __future__ import annotations
from pathlib import Path
import argparse
import json
import subprocess
import sys

ROOT = Path("/home/quyet/k210_lab")
MIX = ROOT / "mixvpr"
SRC = MIX / "src"
RUN_ROOT = MIX / "artifacts" / "v2_progressive"
CONFIG = MIX / "configs" / "v2_progressive.json"
K210_PY = Path("/home/quyet/miniconda3/envs/k210/bin/python")

sys.path.insert(0, str(SRC))

from mixvpr_k210.config import (
    load_config, stage_ids, spec_for_stage, stage_config
)
from mixvpr_k210.deployment import export_stage
from mixvpr_k210.trainer import (
    lineage_status, next_stage, run_stage
)

def cfg():
    return load_config(CONFIG)

def stage_dir(stage_id: str) -> Path:
    return RUN_ROOT / "stages" / stage_id

def deployment_dir(stage_id: str) -> Path:
    return stage_dir(stage_id) / "deployment"

def checkpoint_for(stage_id: str) -> Path | None:
    best = stage_dir(stage_id) / "best.pt"
    return best if best.is_file() else None

def cmd_status(_args):
    rows = lineage_status(cfg(), RUN_ROOT)
    print(json.dumps(rows, indent=2))
    print("NEXT_STAGE", next_stage(cfg(), RUN_ROOT))

def cmd_test(_args):
    subprocess.run(
        [
            str(ROOT / ".venv-mixvpr-cuda/bin/python"),
            "-m", "unittest", "-v",
            "mixvpr/tests/test_structure.py",
        ],
        cwd=ROOT,
        env={**__import__("os").environ,
             "PYTHONPATH": str(SRC)},
        check=True,
    )

def cmd_export(args):
    out = deployment_dir(args.stage) / "onnx"
    report = export_stage(
        cfg(), args.stage, out,
        checkpoint=checkpoint_for(args.stage),
    )
    print(json.dumps(report, indent=2))

def cmd_compile(args):
    onnx = deployment_dir(args.stage) / "onnx" / "model.onnx"
    if not onnx.is_file():
        raise SystemExit(
            f"missing ONNX for {args.stage}; run export first"
        )
    out = deployment_dir(args.stage) / "k210"
    subprocess.run(
        [
            str(K210_PY),
            str(MIX / "scripts/nncase_compile.py"),
            "--onnx", str(onnx),
            "--calibration",
            cfg()["data"]["calibration"],
            "--out", str(out),
        ],
        cwd=ROOT,
        check=True,
    )

def cmd_feasibility(args):
    cmd_export(args)
    cmd_compile(args)

def cmd_train(args):
    result = run_stage(cfg(), RUN_ROOT, args.stage)
    print(json.dumps(result, indent=2))

def cmd_train_next(_args):
    stage = next_stage(cfg(), RUN_ROOT)
    if stage is None:
        print("ALL_STAGES_COMPLETE")
        return
    print("TRAINING", stage, flush=True)
    result = run_stage(cfg(), RUN_ROOT, stage)
    print(json.dumps(result, indent=2))

def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("status").set_defaults(fn=cmd_status)
    sub.add_parser("test").set_defaults(fn=cmd_test)

    for name, fn in [
        ("export", cmd_export),
        ("compile", cmd_compile),
        ("feasibility", cmd_feasibility),
        ("train", cmd_train),
    ]:
        p = sub.add_parser(name)
        p.add_argument("stage", choices=stage_ids(cfg()))
        p.set_defaults(fn=fn)

    sub.add_parser("train-next").set_defaults(fn=cmd_train_next)
    args = parser.parse_args()
    args.fn(args)

if __name__ == "__main__":
    main()
