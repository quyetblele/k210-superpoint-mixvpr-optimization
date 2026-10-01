from __future__ import annotations
from pathlib import Path
import hashlib
import json
import random

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import torch
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
def sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()

def load_bundle(cfg: dict) -> dict:
    meta = Path(cfg["data"]["metadata"])
    manifest_path = Path(cfg["data"]["teacher_bank_manifest"])
    bank_path = Path(cfg["data"]["teacher_bank"])
    data = json.loads(meta.read_text())
    bank = json.loads(manifest_path.read_text())
    if sha256(bank_path) != bank["descriptors_sha256"]:
        raise RuntimeError("teacher bank hash mismatch")
    data["teacher_bank_manifest"] = bank
    data["teacher_bank_path"] = str(bank_path)
    data["dev_root"] = str(meta.parent)
    return data

def image_tensor(path: str, size: int, augment_seed: int | None):
    with Image.open(path) as src:
        image = src.convert("RGB")
    image = TF.resize(
        image, [size, size],
        InterpolationMode.BICUBIC, antialias=True
    )
    if augment_seed is not None:
        rng = random.Random(augment_seed)
        image = ImageEnhance.Brightness(image).enhance(rng.uniform(0.7, 1.3))
        image = ImageEnhance.Contrast(image).enhance(rng.uniform(0.75, 1.25))
        image = ImageEnhance.Color(image).enhance(rng.uniform(0.75, 1.25))
        if rng.random() < 0.25:
            image = image.filter(ImageFilter.GaussianBlur(rng.uniform(0.1, 1.2)))
        if rng.random() < 0.15:
            image = image.convert("L").convert("RGB")
    return TF.normalize(TF.to_tensor(image), MEAN, STD)

def make_plan(
    data: dict,
    steps: int,
    seed: int,
    project_probability=0.1,
    sampler: dict | None = None,
):
    rows = data["rows"]
    places = {k: [int(i) for i in v] for k, v in data["train_places"].items()}
    neighbors = data["teacher_bank_manifest"]["neighbors"]
    indoor = sorted(
        k for k, v in places.items()
        if rows[v[0]]["domain"] == "indoor"
    )
    project = sorted(
        k for k, v in places.items()
        if rows[v[0]]["domain"] == "project"
    )
    sampler = sampler or {}
    mode = sampler.get("mode", "p2k4")
    project_probability = float(
        sampler.get("project_probability", project_probability)
    )
    rng = random.Random(seed)
    plan = np.empty((steps, 4, 8, 3), np.int64)
    for step in range(steps):
        for micro in range(4):
            use_project = project and rng.random() < project_probability
            pool = project if use_project else indoor
            anchor = rng.choice(pool)
            if mode == "p2k4":
                choices = [x for x in neighbors[anchor][:16] if x in places]
                negative = rng.choice(choices) if choices else rng.choice(indoor)
                selected = ((anchor, 4), (negative, 4))
            elif mode == "p4k2":
                neighbor_pool = int(sampler.get("neighbor_pool", 16))
                choices = []
                for place in neighbors[anchor][:neighbor_pool]:
                    if (
                        place in places
                        and place != anchor
                        and rows[places[place][0]]["domain"]
                            == rows[places[anchor][0]]["domain"]
                        and place not in choices
                    ):
                        choices.append(place)
                picked = rng.sample(choices, min(3, len(choices)))
                selected_places = [anchor, *picked]
                while len(selected_places) < 4:
                    place = rng.choice(pool)
                    if place not in selected_places:
                        selected_places.append(place)
                selected = tuple((place, 2) for place in selected_places)
            else:
                raise ValueError(f"unknown sampler mode: {mode}")

            cursor = 0
            for label, (place, per_place) in enumerate(selected):
                if mode == "p2k4":
                    chosen = rng.sample(
                        places[place], min(per_place, len(places[place]))
                    )
                    while len(chosen) < per_place:
                        chosen.append(rng.choice(places[place]))
                else:
                    if len(places[place]) < per_place:
                        raise RuntimeError(
                            f"place {place} has fewer than {per_place} images"
                        )
                    chosen = rng.sample(places[place], per_place)
                for image_id in chosen:
                    plan[step, micro, cursor] = [
                        image_id, label, rng.randrange(2**31)
                    ]
                    cursor += 1
    return plan

def training_batch(item, rows, size: int, device):
    images = torch.stack([
        image_tensor(rows[int(i)]["path"], size, int(seed))
        for i, _label, seed in item
    ]).to(device, non_blocking=True)
    labels = torch.tensor(
        item[:, 1].copy(), dtype=torch.long, device=device
    )
    return images, labels

@torch.inference_mode()
def encode_dev(model, data: dict, device, size=240, batch_size=4):
    ids = sorted({
        int(i)
        for scene in data["dev_protocol"]
        for i in np.load(
            Path(data["dev_root"]) / f"{scene}_dev.npz"
        )["ids"]
    })
    offset = {image_id: j for j, image_id in enumerate(ids)}
    values = []
    model.eval()
    rows = data["rows"]
    for start in range(0, len(ids), batch_size):
        batch_ids = ids[start:start + batch_size]
        x = torch.stack([
            image_tensor(rows[i]["path"], size, None)
            for i in batch_ids
        ]).to(device)
        values.append(model(x).float().cpu())
    return ids, offset, torch.cat(values).numpy()

def dev_metrics(values: np.ndarray, offset: dict, data: dict) -> dict:
    scenes = {}
    for name in data["dev_protocol"]:
        d = np.load(Path(data["dev_root"]) / f"{name}_dev.npz")
        v = torch.from_numpy(values[[offset[int(i)] for i in d["ids"]]])
        sim = v[d["query_indices"]] @ v.T
        allowed = torch.from_numpy(d["allowed"])
        positive = torch.from_numpy(d["positive"])
        negative = torch.from_numpy(d["negative"])
        sim.masked_fill_(~allowed, -1e4)
        order = sim.argsort(1, descending=True)
        result = {
            f"R@{k}": float(
                positive.gather(1, order[:, :k]).any(1).float().mean()
            )
            for k in (1, 5, 10)
        }
        pos = sim.masked_fill(~positive, -1e4).max(1).values
        neg = sim.masked_fill(~negative, -1e4).max(1).values
        result["margin"] = float((pos - neg).mean())
        result["queries"] = len(sim)
        scenes[name] = result
    keys = ("R@1", "R@5", "R@10", "margin")
    indoor = {
        k: float(np.mean([
            r[k] for s, r in scenes.items()
            if s != "project_stair"
        ]))
        for k in keys
    }
    project = {k: scenes["project_stair"][k] for k in keys}
    overall = {
        k: 0.9 * indoor[k] + 0.1 * project[k]
        for k in keys
    }
    return {
        "Indoor": indoor,
        "Project": project,
        "Overall": overall,
        "scenes": scenes,
    }

def verify_rows(
    data: dict,
    image_ids,
    expected_split: str | None = None,
) -> dict:
    """Verify exact bytes for a bounded set of metadata rows."""
    rows = data["rows"]
    unique = sorted({int(i) for i in image_ids})
    checked_bytes = 0
    domains = {}
    for image_id in unique:
        row = rows[image_id]
        path = Path(row["path"])
        if expected_split is not None and row["split"] != expected_split:
            raise RuntimeError(
                f"row {image_id} split={row['split']} "
                f"expected={expected_split}"
            )
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = sha256(path)
        if actual != row["sha256"]:
            raise RuntimeError(
                f"row {image_id} hash mismatch: "
                f"{actual} != {row['sha256']}"
            )
        checked_bytes += path.stat().st_size
        domains[row["domain"]] = domains.get(row["domain"], 0) + 1
    return {
        "status": "PASS",
        "unique_images": len(unique),
        "checked_bytes": checked_bytes,
        "domains": domains,
        "expected_split": expected_split,
    }
