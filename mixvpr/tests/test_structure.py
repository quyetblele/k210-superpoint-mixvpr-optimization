from __future__ import annotations
import unittest
import torch

from mixvpr_k210.config import (
    load_config, stage_ids, stage_config, spec_for_stage, validate_progression
)
from mixvpr_k210.model import K210MixVPR, count_params, conv_linear_macs
from mixvpr_k210.pruning import prune_one_axis

CFG = load_config("mixvpr/configs/v2_progressive.json")

class StructureTests(unittest.TestCase):
    def test_progression_contract(self):
        validate_progression(CFG)
        ids = stage_ids(CFG)
        self.assertEqual(ids[0], "00_large")
        self.assertEqual(ids[-1], "06_prune_h")
        self.assertEqual(CFG["training"]["parent_probe_updates"], 1000)
        self.assertLess(
            CFG["training"]["parent_probe_updates"],
            CFG["training"]["parent_max_updates"],
        )
        for sid in ids:
            spec = spec_for_stage(CFG, sid)
            self.assertEqual(spec.mixer_depth, 4)
            self.assertEqual(spec.descriptor_dim, 512)
            self.assertEqual(spec.input_hw, (240, 240))

    def test_final_stage_matches_frozen_d4_complexity(self):
        model = K210MixVPR(spec_for_stage(CFG, "06_prune_h"))
        self.assertEqual(count_params(model), 731300)
        self.assertEqual(conv_linear_macs(model), 258553600)

    def test_full_pruning_lineage_smoke(self):
        torch.manual_seed(20260920)
        ids = stage_ids(CFG)
        model = K210MixVPR(spec_for_stage(CFG, ids[0])).eval()
        x = torch.randn(2, 3, 240, 240)
        with torch.inference_mode():
            y = model(x)
        self.assertEqual(tuple(y.shape), (2, 512))
        torch.testing.assert_close(
            y.norm(dim=1), torch.ones(2), rtol=1e-5, atol=1e-5
        )

        for parent_id, child_id in zip(ids, ids[1:]):
            child = K210MixVPR(spec_for_stage(CFG, child_id)).eval()
            report = prune_one_axis(
                model, child, stage_config(CFG, child_id)["action"]
            )
            self.assertEqual(
                report["action"], stage_config(CFG, child_id)["action"]
            )
            with torch.inference_mode():
                y = child(x)
            self.assertEqual(tuple(y.shape), (2, 512))
            self.assertTrue(torch.isfinite(y).all())
            torch.testing.assert_close(
                y.norm(dim=1), torch.ones(2), rtol=1e-5, atol=1e-5
            )
            model = child

if __name__ == "__main__":
    unittest.main()
