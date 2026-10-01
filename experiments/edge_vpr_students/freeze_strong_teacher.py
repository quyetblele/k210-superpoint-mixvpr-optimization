"""Create an immutable provenance boundary for the DEV-selected strong teacher."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path("/home/quyet/k210_lab")
sys.path.insert(0, str(ROOT / "experiments" / "edge_vpr_students"))
import success_first_train_core as core


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True); parser.add_argument("--stages", nargs="+", type=Path, required=True)
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    target = args.output / "strong_mixvpr_teacher.pt"
    source_hash = core.sha256(args.source)
    if target.exists() and core.sha256(target) != source_hash:
        raise RuntimeError("immutable frozen teacher already exists with different hash")
    if not target.exists():
        temporary = target.with_suffix(".pt.tmp"); shutil.copyfile(args.source, temporary); temporary.replace(target)
    reports = [json.loads((stage / "report.json").read_text()) for stage in args.stages]
    manifest = {"status": "FROZEN", "architecture": "Official MixVPR ResNet50+MixVPR",
                "input": [320, 320], "preprocessing": "direct RGB bicubic stretch + ImageNet normalization",
                "descriptor_dim": 4096, "checkpoint": str(target), "checkpoint_sha256": source_hash,
                "selection": "DEV only; lexicographic R@1, R@5, positive-hardest-negative margin",
                "stage_reports": reports, "gt1_used": False}
    core.atomic_json(args.output / "manifest.json", manifest); print(json.dumps(manifest))


if __name__ == "__main__":
    main()
