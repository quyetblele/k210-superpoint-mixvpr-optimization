from __future__ import annotations
import torch

from .model import K210MixVPR

# Old D4 backbone was one nn.Sequential. These are the Conv2d positions.
_LEGACY_CONVS = [
    (0, 0, "conv1"),
    (2, 0, "conv2"),
    (5, 1, "conv1"),
    (7, 1, "conv2"),
    (10, 2, "conv1"),
    (12, 2, "conv2"),
    (15, 3, "conv1"),
    (17, 3, "conv2"),
    (19, 4, "conv1"),
    (21, 4, "conv2"),
]

@torch.no_grad()
def load_legacy_d4_state(
    model: K210MixVPR,
    legacy_state: dict[str, torch.Tensor],
) -> None:
    """Load the frozen pre-V2 D4 weights into the readable V2 model.

    This is a migration/audit helper only. It proves the refactor changes
    module naming, not the math of the D4 architecture.
    """
    if model.spec.widths != (16, 24, 64, 112, 160):
        raise ValueError("legacy D4 migration requires D4 widths")
    if model.spec.mixer_hidden != 96 or model.spec.mixer_depth != 4:
        raise ValueError("legacy D4 migration requires H96/D4")
    if model.spec.descriptor_dim != 512:
        raise ValueError("legacy D4 migration requires 512D")

    new_state = model.state_dict()
    consumed = set()

    for legacy_index, stage_index, conv_name in _LEGACY_CONVS:
        for suffix in ("weight", "bias"):
            old_key = f"backbone.{legacy_index}.{suffix}"
            new_key = f"stages.{stage_index}.{conv_name}.{suffix}"
            new_state[new_key].copy_(legacy_state[old_key])
            consumed.add(old_key)

    direct_prefixes = (
        "mixers.",
        "channel_projection.",
        "row_projection.",
    )
    for old_key, value in legacy_state.items():
        if old_key.startswith(direct_prefixes):
            if old_key not in new_state:
                raise KeyError(f"new model missing {old_key}")
            new_state[old_key].copy_(value)
            consumed.add(old_key)

    ignored = [
        key for key in legacy_state
        if key not in consumed
    ]
    if ignored:
        raise RuntimeError(f"unmapped legacy tensors: {ignored}")

    model.load_state_dict(new_state, strict=True)

@torch.no_grad()
def widen_preserving_function(
    target: K210MixVPR,
    source: K210MixVPR,
) -> dict:
    """Widen a student while preserving the source function exactly.

    Inherited parameter slices are copied exactly. New feature channels or
    mixer neurons keep their seeded random incoming weights, while outgoing
    connections from new features into the inherited path are zeroed.
    """
    if target.spec.input_hw != source.spec.input_hw:
        raise ValueError("input geometry must stay fixed during widening")
    if target.spec.mixer_depth != source.spec.mixer_depth:
        raise ValueError("mixer depth must stay fixed during widening")
    if target.spec.descriptor_dim != source.spec.descriptor_dim:
        raise ValueError("descriptor interface must stay fixed during widening")

    src = source.state_dict()
    dst = target.state_dict()
    if set(src) != set(dst):
        raise RuntimeError("source/target parameter keys differ")

    report = {}
    for key, value in src.items():
        out = dst[key]
        if out.ndim != value.ndim or any(
            a < b for a, b in zip(out.shape, value.shape)
        ):
            raise ValueError(
                f"target tensor cannot contain source: {key} "
                f"{tuple(out.shape)} vs {tuple(value.shape)}"
            )
        slices = tuple(slice(0, n) for n in value.shape)
        if out.shape != value.shape:
            if out.ndim >= 2 and out.shape[1] > value.shape[1]:
                out[:value.shape[0], value.shape[1]:].zero_()
            out[slices].copy_(value)
        else:
            out.copy_(value)
        report[key] = {
            "source": list(value.shape),
            "target": list(out.shape),
        }

    target.load_state_dict(dst, strict=True)
    return report
