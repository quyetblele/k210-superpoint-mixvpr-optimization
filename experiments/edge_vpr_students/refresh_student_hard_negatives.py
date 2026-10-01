"""Refresh global negatives with a student index while keeping teacher targets frozen."""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

import faiss
import numpy as np
import torch

ROOT = Path("/home/quyet/k210_lab")
sys.path.insert(0, str(ROOT / "experiments" / "edge_vpr_students"))
import success_first_train_core as core


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--identity", choices=("A_C144_H112", "B_C160_H96"), required=True)
    parser.add_argument("--student", type=Path, required=True)
    parser.add_argument("--teacher-bank", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); device = core.require_cuda(); args.output.mkdir(parents=True, exist_ok=True)
    faiss.omp_set_num_threads(2)
    source_manifest = json.loads((args.teacher_bank / "manifest.json").read_text())
    connection = sqlite3.connect(source_manifest["bank"])
    rows = connection.execute("SELECT image_id,place_id,path,offset FROM descriptors ORDER BY offset").fetchall()
    teacher_neighbors: dict[str, list[str]] = defaultdict(list)
    for place, _, negative in connection.execute(
        "SELECT place_id,rank,negative_place_id FROM hard_neighbors ORDER BY place_id,rank"
    ):
        teacher_neighbors[place].append(negative)
    connection.close()

    checkpoint = torch.load(args.student, map_location="cpu", weights_only=False)
    model = core.build_student(args.identity); model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model = model.to(device).eval()
    places = sorted({place for _, place, _, _ in rows}); place_to_index = {place: i for i, place in enumerate(places)}
    centroids = np.zeros((len(places), 512), np.float32); counts = np.zeros(len(places), np.int32)
    with torch.inference_mode():
        for start in range(0, len(rows), 16):
            batch_rows = rows[start:start + 16]
            images = torch.stack([core.image_tensor(path, 240, None) for _, _, path, _ in batch_rows]).to(device)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                values = model(images).float().cpu().numpy()
            for value, (_, place, _, _) in zip(values, batch_rows):
                index = place_to_index[place]; centroids[index] += value; counts[index] += 1
    centroids /= counts[:, None]; faiss.normalize_L2(centroids)
    search = faiss.IndexHNSWFlat(512, 32, faiss.METRIC_INNER_PRODUCT)
    search.hnsw.efConstruction = 80; search.hnsw.efSearch = 128; search.add(centroids)
    _, found = search.search(centroids, 65)

    output_path = args.output / "bank.sqlite"; temporary = output_path.with_suffix(".sqlite.tmp")
    temporary.unlink(missing_ok=True); output = sqlite3.connect(temporary)
    output.executescript("CREATE TABLE descriptors(image_id TEXT PRIMARY KEY,place_id TEXT,path TEXT,offset INTEGER); CREATE TABLE hard_neighbors(place_id TEXT,rank INTEGER,negative_place_id TEXT,PRIMARY KEY(place_id,rank));")
    output.executemany("INSERT INTO descriptors VALUES (?,?,?,?)", rows)
    hard_rows = []
    for i, place in enumerate(places):
        student = [places[int(j)] for j in found[i] if int(j) != i]
        merged = []
        for rank in range(max(len(student), len(teacher_neighbors[place]))):
            for source in (student, teacher_neighbors[place]):
                if rank < len(source) and source[rank] != place and source[rank] not in merged:
                    merged.append(source[rank])
                if len(merged) == 32: break
            if len(merged) == 32: break
        hard_rows.extend((place, rank, negative) for rank, negative in enumerate(merged))
    output.executemany("INSERT INTO hard_neighbors VALUES (?,?,?)", hard_rows)
    output.commit(); output.close(); temporary.replace(output_path)
    manifest = {**source_manifest, "bank": str(output_path), "bank_sha256": core.sha256(output_path),
                "student_refresh": {"identity": args.identity, "checkpoint": str(args.student),
                                    "checkpoint_sha256": core.sha256(args.student),
                                    "strategy": "deduplicated alternating student-HNSW and frozen-teacher neighbors"},
                "gt1_used": False}
    core.atomic_json(args.output / "manifest.json", manifest); print(json.dumps(manifest))


if __name__ == "__main__":
    main()
