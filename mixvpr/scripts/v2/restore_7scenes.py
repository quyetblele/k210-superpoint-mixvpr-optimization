from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
from zipfile import ZipFile

LEGACY_ROOT = Path(
    "/home/quyet/k210_lab/datasets/7scenes"
)


def file_sha256(path: Path, block=262144):
    h = hashlib.sha256()
    with path.open("rb", buffering=0) as f:
        while True:
            chunk = f.read(block)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def scene_rows(metadata: Path, scene: str):
    data = json.loads(metadata.read_text())
    return [
        row for row in data["rows"]
        if f"/7scenes/{scene}/" in row["path"]
    ]


def expected_by_sequence(rows):
    result = {}
    for row in rows:
        rel = Path(row["path"]).relative_to(
            LEGACY_ROOT
        ).as_posix()
        parts = PurePosixPath(rel).parts
        if len(parts) < 3:
            raise RuntimeError(f"bad metadata path: {rel}")
        seq = parts[1]
        inner = "/".join(parts[1:])
        result.setdefault(seq, {})[inner] = row["sha256"]
    return result


def extract_sequence(
    outer: ZipFile,
    scene: str,
    seq: str,
    expected: dict,
    scene_root: Path,
):
    existing_ok = True
    for rel, digest in expected.items():
        path = scene_root / rel
        if not path.is_file() or file_sha256(path) != digest:
            existing_ok = False
            break
    if existing_ok:
        print("SEQ_VERIFIED_PRESENT", seq, len(expected), flush=True)
        return

    nested_name = f"{scene}/{seq}.zip"
    if nested_name not in outer.namelist():
        raise RuntimeError(f"missing outer member: {nested_name}")
    nested = scene_root / f"{seq}.zip"

    print("OUTER_EXTRACT", seq, flush=True)
    with outer.open(nested_name) as src, nested.open("wb") as dst:
        shutil.copyfileobj(src, dst, length=1 << 20)

    found = set()
    try:
        with ZipFile(nested) as inner:
            for info in inner.infolist():
                name = info.filename.replace(chr(92), "/")
                if name not in expected:
                    continue
                parts = PurePosixPath(name).parts
                if ".." in parts or not parts or parts[0] != seq:
                    raise RuntimeError(f"unsafe member: {name}")
                dst = scene_root / name
                dst.parent.mkdir(parents=True, exist_ok=True)
                with inner.open(info) as src, dst.open("wb") as out:
                    shutil.copyfileobj(src, out, length=1 << 20)
                found.add(name)
    finally:
        nested.unlink(missing_ok=True)

    missing = set(expected) - found
    if missing:
        raise RuntimeError(
            f"{seq}: missing {len(missing)} expected color files"
        )
    print("SEQ_PASS", seq, len(found), flush=True)


def verify_scene(rows, scene_root: Path):
    print("HASH_VERIFY_START", len(rows), flush=True)
    for index, row in enumerate(rows, 1):
        legacy = Path(row["path"])
        rel = legacy.relative_to(LEGACY_ROOT)
        actual = scene_root.parent / rel
        if not actual.is_file():
            raise RuntimeError(f"missing {actual}")
        digest = file_sha256(actual)
        if digest != row["sha256"]:
            raise RuntimeError(f"hash mismatch {actual}")
        if index % 1000 == 0:
            print("HASH_OK", index, flush=True)


def restore(args):
    archive = Path(args.archive)
    metadata = Path(args.metadata)
    root = Path(args.output_root)
    scene_root = root / args.scene
    scene_root.mkdir(parents=True, exist_ok=True)
    rows = scene_rows(metadata, args.scene)
    byseq = expected_by_sequence(rows)

    print(
        "RESTORE_START", args.scene,
        "rows", len(rows),
        "sequences", sorted(byseq),
        flush=True,
    )
    with ZipFile(archive) as outer:
        for seq in sorted(byseq):
            extract_sequence(
                outer, args.scene, seq,
                byseq[seq], scene_root,
            )

    verify_scene(rows, scene_root)

    marker = scene_root / ".restore_verified.json"
    marker.write_text(json.dumps({
        "scene": args.scene,
        "metadata_rows": len(rows),
        "status": "PASS",
        "color_only": True,
        "hash_buffer": 262144,
        "archive": str(archive),
    }, indent=2) + "\n")
    if args.delete_archive:
        archive.unlink()
    print(
        "SCENE_PASS", args.scene,
        "rows", len(rows),
        "archive_deleted", not archive.exists(),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("scene")
    parser.add_argument("--archive", required=True)
    parser.add_argument(
        "--metadata",
        default="/mnt/d/k210_official_ab/data.json",
    )
    parser.add_argument(
        "--output-root",
        default="/mnt/d/k210_datasets/7scenes",
    )
    parser.add_argument(
        "--delete-archive",
        action="store_true",
    )
    restore(parser.parse_args())


if __name__ == "__main__":
    main()
