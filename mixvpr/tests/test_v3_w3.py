from pathlib import Path
import sys
import unittest
import torch

ROOT = Path("/home/quyet/k210_lab")
sys.path.insert(0, str(ROOT / "mixvpr/src"))

from mixvpr_k210.config import load_config, spec_for_stage, validate_progression
from mixvpr_k210.legacy import load_legacy_d4_state, widen_preserving_function
from mixvpr_k210.model import K210MixVPR
from mixvpr_k210.pruning import prune_one_axis

CFG = load_config("mixvpr/configs/v3_progressive_w3.json")

class V3W3Tests(unittest.TestCase):
    def test_progression_and_endpoints(self):
        validate_progression(CFG)
        self.assertEqual(spec_for_stage(CFG, "00_large").widths, (16,24,80,160,224))
        self.assertEqual(spec_for_stage(CFG, "00_large").mixer_hidden, 160)
        self.assertEqual(spec_for_stage(CFG, "14_h_96").widths, (16,24,64,112,160))
        self.assertEqual(spec_for_stage(CFG, "14_h_96").mixer_hidden, 96)

    def test_w3_pruning_slices_consumer(self):
        torch.manual_seed(1)
        parent = K210MixVPR(spec_for_stage(CFG, "08_w4_112"))
        child = K210MixVPR(spec_for_stage(CFG, "09_w3_72"))
        report = prune_one_axis(parent, child, "prune_w3")
        keep = torch.tensor(report["kept_indices"])
        self.assertEqual(len(keep), 72)
        torch.testing.assert_close(
            child.stages[2].conv2.weight,
            parent.stages[2].conv2.weight[keep][:, keep],
            rtol=0, atol=0,
        )
        torch.testing.assert_close(
            child.stages[3].conv1.weight,
            parent.stages[3].conv1.weight[:, keep],
            rtol=0, atol=0,
        )

    def test_large_warmstart_preserves_d4(self):
        checkpoint = torch.load(
            CFG["protected_baseline"]["frozen_d4_checkpoint"],
            map_location="cpu", weights_only=False,
        )
        compact = K210MixVPR(spec_for_stage(CFG, "14_h_96")).eval()
        load_legacy_d4_state(compact, checkpoint["model_state_dict"])
        torch.manual_seed(CFG["seed"])
        large = K210MixVPR(spec_for_stage(CFG, "00_large")).eval()
        widen_preserving_function(large, compact)
        x = torch.randn(1, 3, 240, 240)
        with torch.inference_mode():
            a, b = compact(x), large(x)
        torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)

if __name__ == "__main__":
    unittest.main()
