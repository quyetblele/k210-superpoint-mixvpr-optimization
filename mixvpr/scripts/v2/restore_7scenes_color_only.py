from __future__ import annotations
import argparse
import hashlib
import json
import shutil
from pathlib import Path, PurePosixPath
from zipfile import ZipFile


METADATA = Path("/mnt/d/k210_official_ab/data.json")
DATASET_ROOT = Path("/mnt/d/k210_datasets/7scenes")
LEGACY_ROOT = Path("/home/quyet/k210_lab/datasets/7scenes")


def sha256_small_buffer(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb", buffering=0) as f:
        while True:
            block = f.read(262144)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def expected_for_scene(scene: str):
    data = json.loads(METADATA.read_text())
    rows = [
        row for row in data["rows"]
        if f"/7scenes/{scene}/" in row["path"]
    ]
    expected = {
        Path(row["path"]).relative_to(LEGACY_ROOT).as_posix():
        row["sha256"]
        for row in rows
    }
    by_sequence = {}
    for rel in expected:
        parts = PurePosixPath(rel).parts
        if len(parts) < 3 or parts[0] != scene:
            raise RuntimeError(f"unexpected metadata path: {rel}")
        by_sequence.setdefault(parts[1], set()).add(
            "/".join(parts[1:])
        )
    return rows, expected, by_sequence


def restore(scene: str, archive: Path, delete_archive: bool):
    scene_root = DATASET_ROOT / scene
    marker = scene_root / ".restore_verified.json"
    if marker.exists():
        record = json.loads(marker.read_text())
        if record.get("status") == "PASS":
            print("ALREADY_VERIFIED", scene, record)
            return record

    rows, _expected, by_sequence = expected_for_scene(scene)
    scene_root.mkdir(parents=True, exist_ok=True)
    print(
        "RESTORE_START", scene,
        "rows", len(rows),
        "sequences", sorted(by_sequence),
        flush=True,
    )

    with ZipFile(archive) as outer:
        outer_names = set(outer.namelist())
        for sequence in sorted(by_sequence):
            nested_name = f"{scene}/{sequence}.zip"
            if nested_name not in outer_names:
                raise RuntimeError(
                    f"missing nested archive: {nested_name}"
                )
            nested_path = scene_root / f"{sequence}.zip"
            print("OUTER_EXTRACT", sequence, flush=True)
            with outer.open(nested_name) as src, nested_path.open("wb") as dst:
                shutil.copyfileobj(src, dst, length=1 << 20)

            wanted = by_sequence[sequence]
            found = set()
            print(
                "NESTED_OPEN", sequence,
                nested_path.stat().st_size,
                flush=True,
            )

            with ZipFile(nested_path) as nested:
                for info in nested.infolist():
                    name = info.filename.replace(chr(92), "/")
                    if name not in wanted:
                        continue
                    parts = PurePosixPath(name).parts
                    if ".." in parts or parts[0] != sequence:
                        raise RuntimeError(
                            f"unsafe nested member: {name}"
                        )
                    dst = scene_root / name
                    dst.parent.mkdir(
                        parents=True, exist_ok=True
                    )
                    with nested.open(info) as src, dst.open("wb") as out:
                        shutil.copyfileobj(
                            src, out, length=1 << 20
                        )
                    found.add(name)
            missing = wanted - found
            if missing:
                raise RuntimeError(
                    f"{sequence}: missing {len(missing)} color images"
                )
            nested_path.unlink()
            print(
                "SEQ_PASS", sequence,
                "color", len(found),
                flush=True,
            )


    print("HASH_VERIFY_START", len(rows), flush=True)
    for index, row in enumerate(rows, 1):
        path = Path(row["path"])
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = sha256_small_buffer(path)
        if actual != row["sha256"]:
            raise RuntimeError(
                f"hash mismatch: {path}"
            )
        if index % 1000 == 0:
            print("HASH_OK", index, flush=True)

    record = {
        "scene": scene,
        "metadata_rows": len(rows),
        "status": "PASS",
        "color_only": True,
        "hash_buffer": 262144,
    }
    marker.write_text(json.dumps(record, indent=2) + "\n")

    if delete_archive:
        archive.unlink()
    print(
        "SCENE_PASS", scene,
        "rows", len(rows),
        "archive_deleted", not archive.exists(),
        flush=True,
    )
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("scene")
    parser.add_argument("archive", type=Path)
    parser.add_argument(
        "--delete-archive",
        action="store_true",
    )
    args = parser.parse_args()
    restore(
        args.scene,
        args.archive,
        args.delete_archive,
    )


if __name__ == "__main__":
    main()
