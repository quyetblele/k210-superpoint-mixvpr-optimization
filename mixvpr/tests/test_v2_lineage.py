from pathlib import Path
import sys

import torch

ROOT = Path("/home/quyet/k210_lab")
sys.path.insert(0, str(ROOT / "mixvpr/src"))
sys.path.insert(0, str(ROOT / "experiments/edge_vpr_students"))

from mixvpr_k210.config import load_config, spec_for_stage
from mixvpr_k210.legacy import (
    load_legacy_d4_state,
    widen_preserving_function,
)
from mixvpr_k210.model import K210MixVPR, count_params, conv_linear_macs
from mixvpr_k210.pruning import prune_one_axis
from s512_br1_sq240_m100_c160h96_model import S512BR1SQ240M100C160H96

CFG = load_config("mixvpr/configs/v2_progressive.json")

def test_compact_geometry_matches_frozen_d4():
    model = K210MixVPR(spec_for_stage(CFG, "06_prune_h"))
    assert count_params(model) == 731300
    assert conv_linear_macs(model) == 258553600
    x = torch.randn(2, 3, 240, 240)
    with torch.inference_mode():
        y = model(x)
    assert y.shape == (2, 512)
    torch.testing.assert_close(
        y.norm(dim=1),
        torch.ones(2),
        rtol=1e-5,
        atol=1e-5,
    )

def test_refactor_is_numerically_equal_to_legacy_d4():
    checkpoint = torch.load(
        "/home/quyet/training_recovery/final_gate2/candidate.pt",
        map_location="cpu",
        weights_only=False,
    )
    legacy = S512BR1SQ240M100C160H96().eval()
    legacy.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True,
    )
    modern = K210MixVPR(
        spec_for_stage(CFG, "06_prune_h")
    ).eval()
    load_legacy_d4_state(
        modern, checkpoint["model_state_dict"]
    )

    generator = torch.Generator().manual_seed(1234)
    x = torch.randn(
        2, 3, 240, 240, generator=generator
    )
    with torch.inference_mode():
        old_raw = legacy.raw(x)
        new_raw = modern.raw(x)
        old_y = legacy(x)
        new_y = modern(x)
    torch.testing.assert_close(
        new_raw, old_raw, rtol=0, atol=0
    )
    torch.testing.assert_close(
        new_y, old_y, rtol=0, atol=0
    )

def test_stage0_warmstart_preserves_frozen_d4_function():
    checkpoint = torch.load(
        CFG["protected_baseline"]["frozen_d4_checkpoint"],
        map_location="cpu",
        weights_only=False,
    )
    compact = K210MixVPR(
        spec_for_stage(CFG, "06_prune_h")
    ).eval()
    load_legacy_d4_state(
        compact, checkpoint["model_state_dict"]
    )
    torch.manual_seed(CFG["seed"])
    large = K210MixVPR(
        spec_for_stage(CFG, "00_large")
    ).eval()
    report = widen_preserving_function(large, compact)
    assert report

    generator = torch.Generator().manual_seed(777)
    x = torch.randn(2, 3, 240, 240, generator=generator)
    with torch.inference_mode():
        compact_raw = compact.raw(x)
        large_raw = large.raw(x)
        compact_y = compact(x)
        large_y = large(x)
    torch.testing.assert_close(
        large_raw, compact_raw, rtol=1e-5, atol=1e-6
    )
    torch.testing.assert_close(
        large_y, compact_y, rtol=1e-5, atol=1e-6
    )


def _make(stage_id, seed):
    torch.manual_seed(seed)
    return K210MixVPR(spec_for_stage(CFG, stage_id))

def test_w5_pruning_slices_exact_weights():
    parent = _make("00_large", 1)
    child = _make("01_prune_w5", 2)
    report = prune_one_axis(
        parent, child, "prune_w5"
    )
    keep = torch.tensor(report["kept_indices"])
    assert len(keep) == 176
    torch.testing.assert_close(
        child.stages[4].conv1.weight,
        parent.stages[4].conv1.weight[keep],
        rtol=0, atol=0,
    )
    torch.testing.assert_close(
        child.channel_projection.weight,
        parent.channel_projection.weight[:, keep],
        rtol=0, atol=0,
    )
    with torch.inference_mode():
        assert child(torch.randn(1, 3, 240, 240)).shape == (1, 512)

def test_w4_pruning_slices_exact_weights():
    parent = _make("01_prune_w5", 3)
    child = _make("02_prune_w4", 4)
    report = prune_one_axis(
        parent, child, "prune_w4"
    )
    keep = torch.tensor(report["kept_indices"])
    assert len(keep) == 120
    torch.testing.assert_close(
        child.stages[3].conv2.weight,
        parent.stages[3].conv2.weight[keep][:, keep],
        rtol=0, atol=0,
    )
    torch.testing.assert_close(
        child.stages[4].conv1.weight,
        parent.stages[4].conv1.weight[:, keep],
        rtol=0, atol=0,
    )

def test_hidden_pruning_slices_exact_weights():
    parent = _make("02_prune_w4", 5)
    child = _make("03_prune_h", 6)
    report = prune_one_axis(
        parent, child, "prune_hidden"
    )
    assert len(report["per_mixer"]) == 4
    for record, src, dst in zip(
        report["per_mixer"],
        parent.mixers,
        child.mixers,
    ):
        keep = torch.tensor(record["kept_indices"])
        assert len(keep) == 112
        torch.testing.assert_close(
            dst.fc1.weight,
            src.fc1.weight[keep],
            rtol=0, atol=0,
        )
        torch.testing.assert_close(
            dst.fc2.weight,
            src.fc2.weight[:, keep],
            rtol=0, atol=0,
        )

def test_every_stage_is_d4_and_512d():
    for stage in CFG["stages"]:
        spec = spec_for_stage(CFG, stage["id"])
        assert spec.mixer_depth == 4
        assert spec.descriptor_dim == 512
        model = K210MixVPR(spec)
        with torch.inference_mode():
            y = model(torch.zeros(1, 3, 240, 240))
        assert y.shape == (1, 512)
