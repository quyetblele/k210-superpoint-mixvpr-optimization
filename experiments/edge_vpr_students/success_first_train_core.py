"""CUDA-only, resumable core for success-first MixVPR training and screening.

This module deliberately contains no GT1 paths. Selection uses the whole-city
GSV DEV split from the immutable SQLite index.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import torch
import torch.nn.functional as F
from pytorch_metric_learning import losses, miners
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode

ROOT = Path("/home/quyet/k210_lab")
HUB = Path("/home/quyet/.cache/torch/hub/jarvisyjw_MixVPR_main")
OFFICIAL = Path("/home/quyet/.cache/torch/hub/checkpoints/model.ckpt")
GSV_INDEX = ROOT / "datasets" / "success_first" / "gsv_cities.sqlite"
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
SEED = 20260907


def require_cuda() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA_REQUIRED_NO_CPU_FALLBACK")
    # Image decoding is synchronous; small thread pools avoid competing with
    # downloads/checkpoint I/O on the user's 12-GiB machine.
    torch.set_num_threads(2)
    return torch.device("cuda")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_torch(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def rng_state() -> dict:
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(), "torch_cuda": torch.cuda.get_rng_state_all()}


def restore_rng(state: dict) -> None:
    random.setstate(state["python"]); np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"]); torch.cuda.set_rng_state_all(state["torch_cuda"])


def build_teacher():
    sys.path.insert(0, str(HUB))
    from vpr_model import VPRModel
    model = VPRModel(
        backbone_arch="resnet50", pretrained=False, layers_to_freeze=2, layers_to_crop=[4],
        agg_arch="MixVPR", agg_config={"in_channels": 1024, "in_h": 20, "in_w": 20,
        "out_channels": 1024, "mix_depth": 4, "mlp_ratio": 1, "out_rows": 4},
        faiss_gpu=False,
    )
    model.load_state_dict(torch.load(OFFICIAL, map_location="cpu", weights_only=False), strict=True)
    return model


def build_student(identity: str):
    sys.path.insert(0, str(ROOT / "experiments" / "edge_vpr_students"))
    if identity == "A_C144_H112":
        from s512_br1_sq240_m100_model import S512BR1SQ240M100
        return S512BR1SQ240M100()
    if identity == "B_C160_H96":
        from s512_br1_sq240_m100_c160h96_model import S512BR1SQ240M100C160H96
        return S512BR1SQ240M100C160H96()
    raise ValueError(identity)


class PlaceIndex:
    def __init__(self, database: Path = GSV_INDEX):
        if not database.is_file():
            raise RuntimeError(f"immutable dataset index missing: {database}")
        connection = sqlite3.connect(database)
        rows = connection.execute("SELECT place_id,path,split FROM images ORDER BY place_id,path").fetchall()
        connection.close()
        self.by_split = {"train": defaultdict(list), "dev": defaultdict(list)}
        for place, path, split in rows:
            self.by_split[split][place].append(path)
        for split in self.by_split:
            self.by_split[split] = {p: paths for p, paths in self.by_split[split].items() if len(paths) >= 4}


def image_tensor(path: str, size: int, augment_seed: int | None) -> torch.Tensor:
    with Image.open(path) as source:
        image = source.convert("RGB")
    image = TF.resize(image, [size, size], InterpolationMode.BICUBIC, antialias=True)
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


def make_plan(index: PlaceIndex, steps: int, seed: int, views: int = 4,
              places_per_step: int = 8, hard_neighbors: dict | None = None) -> list:
    rng = random.Random(seed)
    places = sorted(index.by_split["train"])
    plan = []
    for step in range(steps):
        selected = []
        if hard_neighbors:
            anchors = rng.sample(places, places_per_step // 2)
            selected.extend(anchors)
            occupied = set(anchors)
            for anchor in anchors:
                choices = [p for p in hard_neighbors.get(anchor, []) if p not in occupied]
                negative = rng.choice(choices) if choices else rng.choice([p for p in places if p not in occupied])
                selected.append(negative); occupied.add(negative)
        else:
            selected = rng.sample(places, places_per_step)
        samples = []
        for label, place in enumerate(selected):
            candidates = index.by_split["train"][place]
            paths = (rng.sample(candidates, views) if len(candidates) >= views
                     else rng.choices(candidates, k=views))
            samples.extend({"place": place, "label": label, "path": path,
                            "augmentation_seed": rng.randrange(2**31)} for path in paths)
        plan.append({"step": step, "samples": samples})
    return plan


def load_hard_neighbors(bank: Path) -> dict[str, list[str]]:
    """Load the immutable place-level mining table; no image data is copied."""
    if not bank.is_file():
        raise RuntimeError(f"hard-negative bank missing: {bank}")
    connection = sqlite3.connect(bank)
    rows = connection.execute(
        "SELECT place_id,negative_place_id FROM hard_neighbors ORDER BY place_id,rank"
    ).fetchall()
    connection.close()
    result: dict[str, list[str]] = defaultdict(list)
    for place, negative in rows:
        result[place].append(negative)
    if not result:
        raise RuntimeError("hard-negative bank has no rows")
    return dict(result)


def batch_from_plan(item: dict, size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    images = torch.stack([image_tensor(x["path"], size, x["augmentation_seed"]) for x in item["samples"]])
    labels = torch.tensor([x["label"] for x in item["samples"]], dtype=torch.long)
    return images.to(device, non_blocking=True), labels.to(device)


TASK_LOSS = losses.MultiSimilarityLoss(alpha=1.0, beta=50.0, base=0.0)
TASK_MINER = miners.MultiSimilarityMiner(epsilon=0.1)


def retrieval_loss(descriptors: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return TASK_LOSS(descriptors, labels, TASK_MINER(descriptors, labels))


def ranking_kd(student: torch.Tensor, teacher: torch.Tensor, labels: torch.Tensor,
               temperature: float = 0.07) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    s = student @ student.T; t = teacher @ teacher.T
    diagonal = torch.eye(len(labels), dtype=torch.bool, device=labels.device)
    positive = labels[:, None].eq(labels[None, :]) & ~diagonal
    negative = ~labels[:, None].eq(labels[None, :])
    t_masked = t.masked_fill(diagonal, -1e4)
    log_s = F.log_softmax(s.masked_fill(diagonal, -1e4) / temperature, dim=1)
    prob_t = F.softmax(t_masked / temperature, dim=1)
    pos_score = t.masked_fill(~positive, -1e4).max(dim=1).values
    neg_score = t.masked_fill(~negative, -1e4).max(dim=1).values
    confidence = (pos_score - neg_score).clamp(min=0.0, max=1.0).detach()
    row_kl = F.kl_div(log_s, prob_t, reduction="none").sum(dim=1)
    kd = (row_kl * (0.25 + confidence)).mean()
    student_margin = (s.masked_fill(~positive, -1e4).max(1).values
                      - s.masked_fill(~negative, -1e4).max(1).values).mean()
    return kd, confidence.mean(), student_margin


@torch.inference_mode()
def dev_metrics(model, index: PlaceIndex, size: int, device: torch.device,
                max_places: int = 2500, batch_size: int = 4) -> dict:
    model.eval()
    places = sorted(index.by_split["dev"])[:max_places]
    references, queries, labels = [], [], []
    for label, place in enumerate(places):
        paths = index.by_split["dev"][place]
        references.extend((path, label) for path in paths[:3])
        queries.append((paths[3], label)); labels.append(label)
    def encode(items):
        output = []
        for start in range(0, len(items), batch_size):
            batch = torch.stack([image_tensor(path, size, None) for path, _ in items[start:start + batch_size]]).to(device)
            output.append(model(batch).float().cpu())
        return torch.cat(output)
    ref = encode(references); query = encode(queries)
    ref_labels = torch.tensor([label for _, label in references]); query_labels = torch.tensor(labels)
    similarity = query @ ref.T
    order = similarity.argsort(dim=1, descending=True)
    result = {}
    for k in (1, 5, 10):
        result[f"R@{k}"] = float((ref_labels[order[:, :k]] == query_labels[:, None]).any(1).float().mean())
    positive = similarity.masked_fill(ref_labels[None, :] != query_labels[:, None], -1e4).max(1).values
    negative = similarity.masked_fill(ref_labels[None, :] == query_labels[:, None], -1e4).max(1).values
    result["positive_hardest_negative_margin"] = float((positive - negative).mean())
    result.update({"queries": len(queries), "references": len(references), "places": len(places)})
    return result
