"""Build immutable descriptors, rankings and global negatives from frozen teacher."""
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import faiss
import numpy as np
import torch

ROOT = Path("/home/quyet/k210_lab")
sys.path.insert(0, str(ROOT / "experiments" / "edge_vpr_students"))
import success_first_train_core as core


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--database", type=Path, default=core.GSV_INDEX)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/success_first/strong_teacher_bank")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--loader-workers", type=int, default=8,
                        help="CPU threads for parallel image decode/preprocess.")
    parser.add_argument("--checkpoint-every-images", type=int, default=512)
    parser.add_argument("--images-per-place", type=int, default=1,
                        help="Deterministic representative images per place for mining only.")
    parser.add_argument("--max-images-this-run", type=int,
                        help="Extract at most this many new images, then exit resumably.")
    args = parser.parse_args(); device = core.require_cuda(); args.output.mkdir(parents=True, exist_ok=True)
    faiss.omp_set_num_threads(2)
    checkpoint = torch.load(args.teacher, map_location="cpu", weights_only=False)
    model = core.build_teacher(); model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model = model.to(device).eval()
    connection = sqlite3.connect(args.database)
    rows = connection.execute("SELECT image_id,place_id,path FROM images WHERE split='train' ORDER BY place_id,path").fetchall()
    connection.close()
    if args.images_per_place <= 0 or args.loader_workers <= 0:
        raise ValueError("--images-per-place and --loader-workers must be positive")
    selected = []
    per_place = {}
    for row in rows:
        count = per_place.get(row[1], 0)
        if count < args.images_per_place:
            selected.append(row)
            per_place[row[1]] = count + 1
    rows = selected
    descriptor_path = args.output / "teacher_descriptors_f16.npy"
    progress_path = args.output / "descriptor_progress.json"
    expected_bytes = len(rows) * 4096 * np.dtype(np.float16).itemsize
    complete = 0
    if progress_path.is_file() and descriptor_path.is_file():
        progress = json.loads(progress_path.read_text())
        if (progress.get("rows") != len(rows) or progress.get("teacher_sha256") != core.sha256(args.teacher)
                or progress.get("images_per_place") != args.images_per_place):
            raise RuntimeError("existing bank progress belongs to a different teacher or dataset")
        if descriptor_path.stat().st_size < expected_bytes:
            raise RuntimeError("descriptor memmap is truncated")
        complete = int(progress.get("complete", 0))
        descriptors = np.lib.format.open_memmap(descriptor_path, mode="r+", dtype=np.float16, shape=(len(rows), 4096))
    else:
        descriptors = np.lib.format.open_memmap(descriptor_path, mode="w+", dtype=np.float16, shape=(len(rows), 4096))
        core.atomic_json(progress_path, {"status": "EXTRACTING", "rows": len(rows), "complete": 0,
                                         "teacher_sha256": core.sha256(args.teacher), "batch_size": args.batch_size,
                                         "images_per_place": args.images_per_place})
    stop = len(rows)
    if args.max_images_this_run is not None:
        if args.max_images_this_run <= 0:
            raise ValueError("--max-images-this-run must be positive")
        stop = min(stop, complete + args.max_images_this_run)
    with ThreadPoolExecutor(max_workers=args.loader_workers) as loader, torch.inference_mode():
        for start in range(complete, stop, args.batch_size):
            batch_rows = rows[start:start + args.batch_size]
            images = list(loader.map(lambda row: core.image_tensor(row[2], 320, None), batch_rows))
            batch = torch.stack(images).to(device, non_blocking=True)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                value = model(batch).float()
            value = torch.nn.functional.normalize(value, dim=1).cpu().numpy().astype(np.float16)
            descriptors[start:start + len(batch_rows)] = value
            end = start + len(batch_rows)
            if end % args.checkpoint_every_images < len(batch_rows) or end == stop:
                descriptors.flush()
                core.atomic_json(progress_path, {"status": "EXTRACTING", "rows": len(rows), "complete": end,
                                                 "teacher_sha256": core.sha256(args.teacher), "batch_size": args.batch_size,
                                                 "images_per_place": args.images_per_place})
    descriptors.flush()
    if stop != len(rows):
        core.atomic_json(progress_path, {"status": "EXTRACTING", "rows": len(rows), "complete": stop,
                                         "teacher_sha256": core.sha256(args.teacher), "batch_size": args.batch_size,
                                         "images_per_place": args.images_per_place})
        print(json.dumps({"status": "PARTIAL", "complete": stop, "rows": len(rows)}))
        return
    core.atomic_json(progress_path, {"status": "EXTRACTED", "rows": len(rows), "complete": len(rows),
                                     "teacher_sha256": core.sha256(args.teacher), "batch_size": args.batch_size,
                                     "images_per_place": args.images_per_place})
    places = sorted({place for _, place, _ in rows}); place_index = {place: i for i, place in enumerate(places)}
    centroids = np.zeros((len(places), 4096), dtype=np.float32); counts = np.zeros(len(places), dtype=np.int32)
    for offset, (_, place, _) in enumerate(rows):
        index = place_index[place]; centroids[index] += descriptors[offset].astype(np.float32); counts[index] += 1
    centroids /= counts[:, None]; faiss.normalize_L2(centroids)
    # Exact all-pairs search is quadratic in the number of places and becomes
    # prohibitive for GSV-Cities.  Keep the frozen 4096-D teacher descriptors,
    # but use a deterministic HNSW index solely to mine global candidates.
    search = faiss.IndexHNSWFlat(4096, 32, faiss.METRIC_INNER_PRODUCT)
    search.hnsw.efConstruction = 80
    search.hnsw.efSearch = 128
    search.add(centroids)
    _, neighbors = search.search(centroids, 33)
    bank_path = args.output / "bank.sqlite"; temporary = bank_path.with_suffix(".sqlite.tmp"); temporary.unlink(missing_ok=True)
    output = sqlite3.connect(temporary)
    output.executescript("CREATE TABLE descriptors(image_id TEXT PRIMARY KEY,place_id TEXT,path TEXT,offset INTEGER); CREATE TABLE hard_neighbors(place_id TEXT,rank INTEGER,negative_place_id TEXT,PRIMARY KEY(place_id,rank));")
    output.executemany("INSERT INTO descriptors VALUES (?,?,?,?)", [(image, place, path, i) for i, (image, place, path) in enumerate(rows)])
    hard_rows = []
    for i, place in enumerate(places):
        rank = 0
        for candidate in neighbors[i]:
            if int(candidate) == i: continue
            hard_rows.append((place, rank, places[int(candidate)])); rank += 1
            if rank == 32: break
    output.executemany("INSERT INTO hard_neighbors VALUES (?,?,?)", hard_rows); output.commit(); output.close(); temporary.replace(bank_path)
    report = {"status": "COMPLETE", "teacher_checkpoint": str(args.teacher),
              "teacher_checkpoint_sha256": core.sha256(args.teacher), "images": len(rows), "places": len(places),
              "database": str(args.database), "database_sha256": core.sha256(args.database),
              "descriptor_shape": [len(rows), 4096], "descriptor_dtype": "float16",
              "descriptors": str(descriptor_path), "descriptors_sha256": core.sha256(descriptor_path),
              "bank": str(bank_path), "bank_sha256": core.sha256(bank_path), "neighbors_per_place": 32,
              "images_per_place": args.images_per_place,
              "mining_index": {"type": "HNSWFlat", "metric": "inner_product", "M": 32,
                                 "efConstruction": 80, "efSearch": 128},
              "gt1_used": False}
    core.atomic_json(args.output / "manifest.json", report); print(json.dumps(report))


if __name__ == "__main__":
    main()
