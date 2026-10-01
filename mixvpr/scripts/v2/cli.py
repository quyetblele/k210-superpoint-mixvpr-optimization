from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "mixvpr/src"))

from mixvpr_k210.config import (
    load_config,
    spec_for_stage,
    stage_ids,
    stage_config,
    validate_progression,
)
from mixvpr_k210.data import verify_rows, sha256
from mixvpr_k210.diagnostics import kd_diagnostic
from mixvpr_k210.feasibility import structural_probe, probe_all
from mixvpr_k210.model import (
    K210MixVPR,
    count_params,
    conv_linear_macs,
)
from mixvpr_k210.smoke import training_smoke
from mixvpr_k210.trainer import (
    run_stage,
    next_stage,
    lineage_status,
    master_plan,
)

CONFIG = Path(os.environ.get(
    "MIXVPR_CONFIG",
    ROOT / "mixvpr/configs/v2_progressive.json",
))
RUN_ROOT = Path(os.environ.get(
    "MIXVPR_RUN_ROOT",
    ROOT / "mixvpr/artifacts/v2_progressive",
))

def config():
    cfg = load_config(CONFIG)
    validate_progression(cfg)
    return cfg

def _scene_name(path: str) -> str:
    _prefix, sep, rest = path.partition("/7scenes/")
    if not sep or "/" not in rest:
        raise RuntimeError(f"cannot parse 7Scenes path: {path}")
    return rest.split("/", 1)[0]


def _training_integrity_guard(data, image_ids):
    rows = data["rows"]
    unique = sorted({int(i) for i in image_ids})
    project_ids = []
    scene_plan_counts = Counter()
    scene_roots = {}
    for image_id in unique:
        row = rows[image_id]
        if row["split"] != "train":
            raise RuntimeError(
                f"row {image_id} split={row['split']} expected=train"
            )
        if row["domain"] == "project":
            project_ids.append(image_id)
            continue
        if row["domain"] != "indoor":
            raise RuntimeError(f"unexpected domain: {row['domain']}")
        scene = _scene_name(row["path"])
        scene_plan_counts[scene] += 1
        prefix = row["path"].split("/7scenes/", 1)[0]
        scene_roots[scene] = Path(prefix) / "7scenes" / scene

    scene_totals = Counter()
    for row in rows:
        if row["domain"] == "indoor":
            scene_totals[_scene_name(row["path"])] += 1

    h = hashlib.sha256()
    scene_records = {}
    for scene in sorted(scene_plan_counts):
        marker = scene_roots[scene] / ".restore_verified.json"
        if not marker.is_file():
            raise RuntimeError(f"missing restore marker: {marker}")
        record = json.loads(marker.read_text())
        if record.get("status") != "PASS":
            raise RuntimeError(f"restore marker not PASS: {marker}")
        if int(record.get("metadata_rows", -1)) != scene_totals[scene]:
            raise RuntimeError(
                f"restore marker row count mismatch for {scene}"
            )
        marker_hash = sha256(marker)
        h.update(scene.encode())
        h.update(marker_hash.encode())
        scene_records[scene] = {
            "plan_unique_images": scene_plan_counts[scene],
            "metadata_rows_verified_at_restore": scene_totals[scene],
            "marker_sha256": marker_hash,
        }

    for image_id in project_ids:
        path = Path(rows[image_id]["path"])
        if not path.is_file():
            raise FileNotFoundError(path)
        stat = path.stat()
        h.update(str(image_id).encode())
        h.update(str(path).encode())
        h.update(str(stat.st_size).encode())
        h.update(str(stat.st_mtime_ns).encode())

    return h.hexdigest(), scene_records, project_ids


def ensure_data_integrity(cfg):
    plan, data = master_plan(cfg, RUN_ROOT)
    plan_path = RUN_ROOT / "shared/train_plan.npy"
    marker = RUN_ROOT / "shared/data_integrity.json"
    plan_hash = sha256(plan_path)
    metadata_hash = sha256(cfg["data"]["metadata"])
    teacher_manifest_hash = sha256(
        cfg["data"]["teacher_bank_manifest"]
    )
    image_ids = plan[:, :, :, 0].ravel()
    guard, scene_records, project_ids = _training_integrity_guard(
        data, image_ids
    )

    if marker.exists():
        cached = json.loads(marker.read_text())
        if (
            cached.get("plan_sha256") == plan_hash
            and cached.get("training_guard_sha256") == guard
            and cached.get("metadata_sha256") == metadata_hash
            and cached.get("teacher_manifest_sha256")
                == teacher_manifest_hash
            and cached.get("status") == "PASS"
        ):
            return cached

    project_report = verify_rows(
        data,
        project_ids,
        expected_split="train",
    )
    unique_ids = sorted({int(i) for i in image_ids})
    report = {
        "status": "PASS",
        "unique_images": len(unique_ids),
        "indoor_unique_images": len(unique_ids) - len(project_ids),
        "project_unique_images": len(project_ids),
        "verification_mode": {
            "indoor": "full-scene SHA256 restore markers",
            "project": "direct SHA256 of train-plan rows",
        },
        "scene_markers": scene_records,
        "project_verification": project_report,
        "plan_sha256": plan_hash,
        "training_guard_sha256": guard,
        "metadata_sha256": metadata_hash,
        "teacher_manifest_sha256": teacher_manifest_hash,
    }
    marker.write_text(
        json.dumps(report, indent=2) + "\n"
    )
    return report

def cmd_status(_args):
    cfg = config()
    data = json.loads(
        Path(cfg["data"]["metadata"]).read_text()
    )
    paths = Counter()
    for row in data["rows"]:
        key = (
            f'{row["domain"]}_'
            f'{"ok" if Path(row["path"]).is_file() else "missing"}'
        )
        paths[key] += 1

    result = {
        "pipeline": cfg["name"],
        "dataset_paths": dict(paths),
        "next_training_stage": next_stage(cfg, RUN_ROOT),
        "lineage": lineage_status(cfg, RUN_ROOT),
    }
    summary = RUN_ROOT / "feasibility/summary.json"
    if summary.exists():
        result["feasibility"] = json.loads(
            summary.read_text()
        )
    print(json.dumps(result, indent=2))

def cmd_check(_args):
    cfg = config()
    rows = []
    for stage_id in stage_ids(cfg):
        spec = spec_for_stage(cfg, stage_id)
        model = K210MixVPR(spec)
        with torch.inference_mode():
            y = model(torch.zeros(1, 3, 240, 240))
        if tuple(y.shape) != (1, 512):
            raise RuntimeError(
                f"{stage_id}: unexpected output {tuple(y.shape)}"
            )
        rows.append({
            "stage": stage_id,
            "action": stage_config(cfg, stage_id)["action"],
            "widths": list(spec.widths),
            "hidden": spec.mixer_hidden,
            "depth": spec.mixer_depth,
            "descriptor": spec.descriptor_dim,
            "params": count_params(model),
            "macs": conv_linear_macs(model),
        })

    compact = rows[-1]
    if compact["params"] != 731300:
        raise RuntimeError(
            "compact D4 parameter parity failed"
        )
    if compact["macs"] != 258553600:
        raise RuntimeError("compact D4 MAC parity failed")

    print(json.dumps({
        "status": "PASS",
        "stages": rows,
        "compact_matches_frozen_D4_geometry": True,
    }, indent=2))

def cmd_smoke(_args):
    cfg = config()
    report = training_smoke(
        cfg, RUN_ROOT / "smoke"
    )
    print(json.dumps(report, indent=2))


def cmd_kd_diagnostic(args):
    cfg = config()
    integrity = ensure_data_integrity(cfg)
    if integrity["status"] != "PASS":
        raise SystemExit("data integrity gate failed")
    report = kd_diagnostic(
        cfg, RUN_ROOT, batches=args.batches
    )
    print(json.dumps(report, indent=2))


def cmd_feasibility(args):
    cfg = config()
    root = RUN_ROOT / "feasibility"
    if args.stage == "all":
        result = probe_all(cfg, root)
    else:
        if args.stage not in stage_ids(cfg):
            raise SystemExit(
                f"unknown stage: {args.stage}"
            )
        result = structural_probe(
            cfg,
            args.stage,
            root / args.stage,
        )
    print(json.dumps(result, indent=2))

def cmd_train(args):
    cfg = config()
    if args.stage not in stage_ids(cfg):
        raise SystemExit(
            f"unknown stage: {args.stage}"
        )

    integrity = ensure_data_integrity(cfg)
    if integrity["status"] != "PASS":
        raise SystemExit("data integrity gate failed")

    expected = next_stage(cfg, RUN_ROOT)
    if expected is not None and args.stage != expected:
        raise SystemExit(
            f"lineage order violation: next stage is {expected}"
        )

    stop_after = args.stop_after_local_step
    first_stage = stage_ids(cfg)[0]
    if stop_after is None and args.stage == first_stage:
        stage_root = RUN_ROOT / "stages" / first_stage
        training_path = stage_root / "training.json"
        progress_path = stage_root / "progress.json"
        if not training_path.exists():
            probe = int(cfg["training"]["parent_probe_updates"])
            current = 0
            if progress_path.exists():
                current = int(json.loads(progress_path.read_text()).get("local_step", 0))
            if current < probe:
                if args.continue_after_probe:
                    raise SystemExit("parent capacity probe must complete before full training")
                stop_after = probe
            elif not args.continue_after_probe:
                raise SystemExit(
                    "parent probe complete; review DEV evidence, then rerun with "
                    "--continue-after-probe only if the capacity gate passes"
                )

    result = run_stage(
        cfg, RUN_ROOT, args.stage,
        stop_after_local_step=stop_after,
    )
    print(json.dumps(result, indent=2))

def cmd_next(_args):
    cfg = config()
    print(next_stage(cfg, RUN_ROOT))

def cmd_plan(_args):
    cfg = config()
    result = []
    for stage in cfg["stages"]:
        spec = spec_for_stage(cfg, stage["id"])
        result.append({
            **stage,
            "spec": {
                "widths": list(spec.widths),
                "hidden": spec.mixer_hidden,
                "depth": spec.mixer_depth,
                "descriptor": spec.descriptor_dim,
            },
        })
    print(json.dumps(result, indent=2))

def main():
    parser = argparse.ArgumentParser(
        description="Canonical MixVPR V2 workflow"
    )
    sub = parser.add_subparsers(
        dest="command",
        required=True,
    )

    for name, fn in (
        ("status", cmd_status),
        ("check", cmd_check),
        ("smoke", cmd_smoke),
        ("next", cmd_next),
        ("plan", cmd_plan),
    ):
        command = sub.add_parser(name)
        command.set_defaults(func=fn)

    command = sub.add_parser("kd-diagnostic")
    command.add_argument(
        "--batches", type=int, default=8,
        help="number of microbatches to inspect before training",
    )
    command.set_defaults(func=cmd_kd_diagnostic)

    command = sub.add_parser("feasibility")
    command.add_argument(
        "stage",
        help="stage id or 'all'",
    )
    command.set_defaults(func=cmd_feasibility)

    command = sub.add_parser("train")
    command.add_argument("stage")
    command.add_argument(
        "--stop-after-local-step",
        type=int,
        default=None,
        help="stop this stage after N local updates for a screening probe",
    )
    command.add_argument(
        "--continue-after-probe",
        action="store_true",
        help="continue stage00 beyond the mandatory capacity probe",
    )
    command.set_defaults(func=cmd_train)

    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
