"""Build supervised 7-Scenes and project-target SQLite indexes, excluding GT1."""
from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from pathlib import Path

import numpy as np

ROOT = Path("/home/quyet/k210_lab")
EDGE = Path("/home/quyet/edge_ai_project")
SEVEN = ROOT / "datasets" / "7scenes"
OUT = ROOT / "datasets" / "success_first"
REPORT = ROOT / "reports" / "success_first"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def place_key(prefix: str, pose: np.ndarray) -> str:
    center = pose[:3, 3]
    view = pose[:3, :3] @ np.asarray([0.0, 0.0, 1.0])
    yaw = math.atan2(view[0], view[2]); pitch = math.asin(float(np.clip(view[1], -1, 1)))
    xyz = tuple(int(round(float(v) / 0.5)) for v in center)
    angles = (int(round(yaw / math.radians(45))), int(round(pitch / math.radians(45))))
    return prefix + ":" + ":".join(map(str, xyz + angles))


def write_index(path: Path, rows: list[tuple]) -> dict:
    temporary = path.with_suffix(".sqlite.tmp"); temporary.unlink(missing_ok=True)
    connection = sqlite3.connect(temporary)
    connection.executescript("""
      CREATE TABLE images (image_id TEXT PRIMARY KEY, place_id TEXT, city TEXT, path TEXT, split TEXT,
                           year INTEGER, month INTEGER, northdeg INTEGER, lat REAL, lon REAL);
      CREATE INDEX images_place ON images(place_id); CREATE INDEX images_split_place ON images(split,place_id);
    """)
    connection.executemany("INSERT INTO images VALUES (?,?,?,?,?,?,?,?,?,?)", rows)
    connection.execute("DELETE FROM images WHERE place_id IN (SELECT place_id FROM images WHERE split='train' GROUP BY place_id HAVING COUNT(*) < 2)")
    connection.commit()
    stats = {split: {"images": connection.execute("SELECT COUNT(*) FROM images WHERE split=?", (split,)).fetchone()[0],
                     "places": connection.execute("SELECT COUNT(DISTINCT place_id) FROM images WHERE split=?", (split,)).fetchone()[0]}
             for split in ("train", "dev")}
    connection.close(); temporary.replace(path)
    return {"path": str(path), "sha256": sha256(path), "stats": stats}


def seven_scenes_rows() -> list[tuple]:
    rows = []
    for scene_root in sorted(path for path in SEVEN.iterdir() if path.is_dir()):
        train_file = scene_root / "TrainSplit.txt"
        test_file = scene_root / "TestSplit.txt"
        if not train_file.is_file() or not test_file.is_file():
            raise RuntimeError(f"missing official split for {scene_root.name}")
        split_sequences = {}
        for split, file in (("train", train_file), ("dev", test_file)):
            for line in file.read_text().splitlines():
                number = int(line.strip().replace("sequence", ""))
                split_sequences[f"seq-{number:02d}"] = split
        for sequence, split in sorted(split_sequences.items()):
            for image in sorted((scene_root / sequence).glob("*.color.png")):
                pose_path = image.with_name(image.name.replace(".color.png", ".pose.txt"))
                if not pose_path.is_file():
                    raise RuntimeError(f"missing pose {pose_path}")
                pose = np.loadtxt(pose_path)
                if pose.shape != (4, 4) or not np.isfinite(pose).all():
                    raise RuntimeError(f"invalid pose {pose_path}")
                place = place_key(f"7scenes:{scene_root.name}", pose)
                image_id = f"7scenes:{scene_root.name}:{sequence}:{image.name}"
                rows.append((image_id, place, scene_root.name, str(image), split, 0, 0, 0, 0.0, 0.0))
    return rows


def project_rows() -> list[tuple]:
    manifest = json.loads((REPORT / "dataset_manifest.json").read_text())["project_target"]
    rows = []
    for sample in manifest["samples"]:
        if sample["split"] == "excluded_guard":
            continue
        pose = np.eye(4); pose[:3, 3] = np.asarray(sample["center"])
        view = np.asarray(sample["view"]); yaw = math.atan2(view[0], view[2]); pitch = math.asin(float(np.clip(view[1], -1, 1)))
        center = pose[:3, 3]; xyz = tuple(int(round(float(v) / 0.5)) for v in center)
        angles = (int(round(yaw / math.radians(45))), int(round(pitch / math.radians(45))))
        place = "project:stair:" + ":".join(map(str, xyz + angles))
        rows.append((sample["id"], place, "project_stair", sample["path"], sample["split"], 0, 0, 0, 0.0, 0.0))
    return rows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True); REPORT.mkdir(parents=True, exist_ok=True)
    seven = write_index(OUT / "7scenes.sqlite", seven_scenes_rows())
    project = write_index(OUT / "project_target.sqlite", project_rows())
    report = {"schema_version": 1, "place_definition": "0.5m XYZ plus 45deg yaw/pitch bins",
              "seven_scenes": seven, "project_target": project, "gt1_used": False}
    temporary = REPORT / "indoor_indexes.json.tmp"; temporary.write_text(json.dumps(report, indent=2) + "\n")
    temporary.replace(REPORT / "indoor_indexes.json"); print(json.dumps(report))


if __name__ == "__main__":
    main()
