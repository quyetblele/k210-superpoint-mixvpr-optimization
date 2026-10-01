"""Build compact immutable manifests without admitting incomplete datasets.

GT1 Floor6/Floor7 is an evaluation-only universe.  The target-domain source
in this pipeline is the separate stair_6_7 sequence; exact content hashes are
also checked against every frozen GT1 image as a second leakage guard.
"""
from __future__ import annotations

import hashlib
import json
import os
import zipfile
from pathlib import Path

import numpy as np
import pycolmap

ROOT = Path("/home/quyet/k210_lab")
EDGE = Path("/home/quyet/edge_ai_project")
OUT = ROOT / "reports" / "success_first"
GT0 = ROOT / "artifacts" / "edge_vpr_students" / "gt0"
SEVEN = ROOT / "datasets" / "7scenes_archives"
SEVEN_EXTRACTED = ROOT / "datasets" / "7scenes"
GSV = ROOT / "datasets" / "gsv_cities"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".ppm"}
EXPECTED_7SCENES = {
    "chess": 3079608937, "fire": 2301204154, "heads": 956332240,
    "office": 4707873861, "pumpkin": 2890874911,
    "redkitchen": 6141181406, "stairs": 1496412431,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def validate_7scenes() -> dict:
    scenes = {}
    all_ready = True
    for scene, expected in EXPECTED_7SCENES.items():
        path = SEVEN / f"{scene}.zip"
        size = path.stat().st_size if path.exists() else 0
        entry = {"archive": str(path), "expected_bytes": expected, "actual_bytes": size,
                 "size_match": size == expected, "zip_test": "NOT_RUN"}
        if size == expected:
            try:
                with zipfile.ZipFile(path) as archive:
                    bad = archive.testzip()
                    entry.update({"zip_test": "PASS" if bad is None else "FAIL",
                                  "archive_entries": len(archive.infolist()),
                                  "sha256": sha256(path)})
            except Exception as error:
                entry.update({"zip_test": "FAIL", "error": f"{type(error).__name__}: {error}"})
        scene_root = SEVEN_EXTRACTED / scene
        colors = sorted(scene_root.rglob("*.color.png")) if scene_root.exists() else []
        poses = sorted(scene_root.rglob("*.pose.txt")) if scene_root.exists() else []
        entry.update({"extracted_root": str(scene_root), "color_images": len(colors),
                      "pose_files": len(poses), "pose_image_count_match": len(colors) == len(poses)})
        ready = (entry["size_match"] and entry["zip_test"] == "PASS"
                 and len(colors) > 0 and len(colors) == len(poses))
        entry["accepted_into_manifest"] = ready
        all_ready &= ready
        scenes[scene] = entry
    return {"role": "indoor_specialization_only", "ready": all_ready, "scenes": scenes}


def gt1_paths() -> list[Path]:
    paths = []
    for floor in ("floor6", "floor7"):
        names = set()
        for role in ("reference", "query", "excluded"):
            names.update((GT0 / f"{floor}_{role}.txt").read_text().splitlines())
        paths.extend(EDGE / "assets" / floor / name for name in sorted(names))
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise RuntimeError(f"missing frozen GT1 source images: {missing[:3]}")
    return paths


def project_target_manifest() -> dict:
    excluded_paths = gt1_paths()
    excluded_hashes = {sha256(path) for path in excluded_paths}
    source = EDGE / "assets" / "stair_6_7"
    candidates = sorted(path for path in source.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)
    # The original stair_6_7_v1 map-package name was retired.  The immutable
    # rebuild contains the same 205 sampled frames and is the actual pose
    # source for this target-domain manifest.
    pose_candidates = [
        EDGE / "map_packages" / "stair_6_7_v1" / "colmap" / "refined_model_v2",
        EDGE / "outputs" / "stair_6_7_rebuild_prepared_v2" / "sfm_v2" / "refined_model_v2",
    ]
    pose_source = next((candidate for candidate in pose_candidates if candidate.is_dir()), None)
    if pose_source is None:
        raise RuntimeError(f"no stair_6_7 pose reconstruction found: {pose_candidates}")
    reconstruction = pycolmap.Reconstruction(pose_source)
    poses = {}
    for image in reconstruction.images.values():
        transform = image.cam_from_world
        transform = transform() if callable(transform) else transform
        rotation = np.asarray(transform.rotation.matrix())
        translation = np.asarray(transform.translation)
        poses[image.name] = {
            "center": (-rotation.T @ translation).tolist(),
            "view": (rotation.T @ np.asarray([0.0, 0.0, 1.0])).tolist(),
        }
    names = [path.relative_to(source).as_posix() for path in candidates]
    if set(names) != set(poses):
        raise RuntimeError(f"target images/poses differ: images={len(names)} poses={len(poses)}")
    # Sequence-level temporal ordering; reserve a contiguous middle 20% DEV
    # block and an eight-frame guard at both boundaries.  This is immutable,
    # avoids random near-neighbour leakage, and does not inspect GT1 metrics.
    ordered = sorted(names, key=lambda name: (name.split("/")[0], int(Path(name).stem.split("_")[-1])))
    dev_start = (len(ordered) - len(ordered) // 5) // 2
    dev_end = dev_start + len(ordered) // 5
    guard = 8
    split_by_name = {}
    for index, name in enumerate(ordered):
        if dev_start <= index < dev_end:
            split_by_name[name] = "dev"
        elif dev_start - guard <= index < dev_start or dev_end <= index < dev_end + guard:
            split_by_name[name] = "excluded_guard"
        else:
            split_by_name[name] = "train"
    rows = []
    overlaps = []
    for path in candidates:
        digest = sha256(path)
        if digest in excluded_hashes:
            overlaps.append(str(path))
            continue
        relative = path.relative_to(source).as_posix()
        split = split_by_name[relative]
        rows.append({"id": f"stair_6_7/{relative}", "path": str(path),
                     "sha256": digest, "split": split, **poses[relative]})
    if overlaps:
        raise RuntimeError(f"project/GT1 content leakage detected: {overlaps[:3]}")
    return {
        "role": "target_domain_finetuning", "source_root": str(source),
        "pose_supervision": str(pose_source), "pose_source_exists": pose_source.exists(),
        "gt1_exclusion": {"path_count": len(excluded_paths), "unique_sha256": len(excluded_hashes),
                          "content_overlap": 0},
        "split_policy": "contiguous middle 20% temporal DEV block with 8-frame guards",
        "counts": {"all": len(rows), "train": sum(r["split"] == "train" for r in rows),
                   "dev": sum(r["split"] == "dev" for r in rows),
                   "excluded_guard": sum(r["split"] == "excluded_guard" for r in rows)},
        "samples": rows,
    }


def gsv_manifest() -> dict:
    # Kaggle CLI 2.x stores OAuth credentials in credentials.json by default;
    # older CLI versions use access_token or kaggle.json.  Accept all three
    # formats, but keep the private-mode check fail-closed.
    credential_files = [Path.home() / ".kaggle" / "access_token",
                        Path.home() / ".kaggle" / "kaggle.json",
                        Path.home() / ".kaggle" / "credentials.json"]
    credentials = next((path for path in credential_files if path.is_file()), None)
    credential_ok = credentials is not None and (os.stat(credentials).st_mode & 0o077) == 0
    images = sorted(path for path in GSV.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES) if GSV.exists() else []
    metadata = sorted(path for path in GSV.rglob("*.csv")) if GSV.exists() else []
    ready = credential_ok and bool(images) and bool(metadata)
    return {
        "role": "large_scale_vpr_pretraining", "dataset": "GSV-Cities",
        "root": str(GSV), "credential_present_and_private": credential_ok,
        "credential_kind": credentials.name if credentials is not None else None,
        "image_count": len(images), "metadata_csv_count": len(metadata), "ready": ready,
        # Compact integrity evidence only: no giant file listing in the report.
        "metadata": [{"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}
                     for path in metadata],
    }


def main() -> None:
    manifest = {
        "schema_version": 1,
        "policy": {"gt1": "evaluation_only_absolute_exclusion",
                   "selection_split": "DEV derived only from training-domain data",
                   "stage_order": ["gsv_cities", "7scenes", "project_target"]},
        "gsv_cities": gsv_manifest(), "seven_scenes": validate_7scenes(),
        "project_target": project_target_manifest(),
    }
    manifest["ready_for_success_first_training"] = (
        manifest["gsv_cities"]["ready"] and manifest["seven_scenes"]["ready"]
        and manifest["project_target"]["counts"]["train"] > 0
        and manifest["project_target"]["counts"]["dev"] > 0
        and manifest["project_target"]["pose_source_exists"]
    )
    atomic_json(OUT / "dataset_manifest.json", manifest)
    print(json.dumps({
        "ready": manifest["ready_for_success_first_training"],
        "gsv": {k: manifest["gsv_cities"][k] for k in ("ready", "credential_present_and_private", "image_count", "metadata_csv_count")},
        "seven_scenes_ready": manifest["seven_scenes"]["ready"],
        "project_counts": manifest["project_target"]["counts"],
        "gt1_overlap": manifest["project_target"]["gt1_exclusion"]["content_overlap"],
    }))


if __name__ == "__main__":
    main()
