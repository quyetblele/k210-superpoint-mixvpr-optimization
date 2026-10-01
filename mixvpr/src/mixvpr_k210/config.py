from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json

ROOT = Path("/home/quyet/k210_lab")

@dataclass(frozen=True)
class ModelSpec:
    input_hw: tuple[int, int]
    widths: tuple[int, int, int, int, int]
    tokens: int
    mixer_hidden: int
    mixer_depth: int
    projection: int
    descriptor_dim: int

    @property
    def rows(self) -> int:
        if self.descriptor_dim % self.projection:
            raise ValueError("descriptor_dim must be divisible by projection")
        return self.descriptor_dim // self.projection

def load_config(path: str | Path) -> dict:
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    return json.loads(path.read_text())

def stage_ids(cfg: dict) -> list[str]:
    return [stage["id"] for stage in cfg["stages"]]

def stage_config(cfg: dict, stage_id: str) -> dict:
    matches = [x for x in cfg["stages"] if x["id"] == stage_id]
    if len(matches) != 1:
        raise KeyError(f"unknown or duplicate stage: {stage_id}")
    return matches[0]

def spec_for_stage(cfg: dict, stage_id: str) -> ModelSpec:
    inv = cfg["invariants"]
    stage = stage_config(cfg, stage_id)
    early = tuple(int(x) for x in inv["early_widths"])
    if len(early) == 3:
        widths = early + (int(stage["w4"]), int(stage["w5"]))
    elif len(early) == 2:
        if "w3" not in stage:
            raise ValueError("variable-W3 configs require stage['w3']")
        widths = early + (
            int(stage["w3"]), int(stage["w4"]), int(stage["w5"])
        )
    else:
        raise ValueError("early_widths must contain 2 or 3 stages")
    return ModelSpec(
        input_hw=tuple(inv["input_hw"]),
        widths=widths,
        tokens=int(inv["tokens"]),
        mixer_hidden=int(stage["hidden"]),
        mixer_depth=int(inv["mixer_depth"]),
        projection=int(inv["projection"]),
        descriptor_dim=int(inv["descriptor_dim"]),
    )

def validate_progression(cfg: dict) -> None:
    ids = stage_ids(cfg)
    hypotheses = cfg.get("hypotheses", {})
    if set(hypotheses) != set(ids):
        raise ValueError("every stage must have exactly one research hypothesis")
    if any(not str(hypotheses[x]).strip() for x in ids):
        raise ValueError("stage hypotheses must be non-empty")
    specs = [spec_for_stage(cfg, x) for x in ids]
    for spec in specs:
        if spec.mixer_depth != 4:
            raise ValueError("V2 invariant broken: mixer_depth must stay D4")
        if spec.descriptor_dim != 512:
            raise ValueError("V2 invariant broken: descriptor must stay 512D")
    for prev, nxt in zip(specs, specs[1:]):
        changed = sum([
            prev.widths[2] != nxt.widths[2],
            prev.widths[3] != nxt.widths[3],
            prev.widths[4] != nxt.widths[4],
            prev.mixer_hidden != nxt.mixer_hidden,
        ])
        if changed != 1:
            raise ValueError("exactly one pruning axis must change per stage")
        if any(
            nxt.widths[i] > prev.widths[i]
            for i in (2, 3, 4)
        ):
            raise ValueError("channel widths must be monotonically non-increasing")
        if nxt.mixer_hidden > prev.mixer_hidden:
            raise ValueError("mixer hidden must be monotonically non-increasing")
