from __future__ import annotations
import unittest
import numpy as np
import torch
import torch.nn.functional as F

from mixvpr_k210.config import (
    load_config, spec_for_stage, stage_config
)
from mixvpr_k210.data import (
    load_bundle, make_plan, training_batch
)
from mixvpr_k210.losses import retrieval_loss, ranking_kd
from mixvpr_k210.model import K210MixVPR
from mixvpr_k210.pruning import prune_one_axis

CFG = load_config("mixvpr/configs/v2_progressive.json")

class TrainingSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA required")
        torch.manual_seed(CFG["seed"])
        np.random.seed(CFG["seed"])
        cls.device = torch.device("cuda")
        cls.data = load_bundle(CFG)
        cls.plan = make_plan(cls.data, 1, CFG["seed"])
        cls.bank = np.load(
            cls.data["teacher_bank_path"], mmap_mode="r"
        )
        cls.bank_offset = {
            int(image_id): index
            for index, image_id in enumerate(
                cls.data["teacher_bank_manifest"]["ids"]
            )
        }

    def batch(self):
        item = self.plan[0, 0]
        x, labels = training_batch(
            item, self.data["rows"], 240, self.device
        )
        teacher = np.stack([
            self.bank[self.bank_offset[int(image_id)]].astype(np.float32)
            for image_id in item[:, 0]
        ])
        teacher = F.normalize(
            torch.from_numpy(teacher).to(self.device), dim=1
        )
        return x, labels, teacher

    def assert_finite_grads(self, model):
        grads = [
            p.grad for p in model.parameters()
            if p.grad is not None
        ]
        self.assertTrue(grads)
        self.assertTrue(all(torch.isfinite(g).all() for g in grads))

    def test_large_student_teacher_kd_update(self):
        x, labels, teacher = self.batch()
        model = K210MixVPR(
            spec_for_stage(CFG, "00_large")
        ).to(self.device)
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
        before = model.stages[0].conv1.weight.detach().clone()

        student = model(x).float()
        task = retrieval_loss(student, labels)
        kd = ranking_kd(student, teacher, labels)
        loss = task + kd
        self.assertTrue(torch.isfinite(loss))

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.assert_finite_grads(model)
        optimizer.step()

        delta = (
            model.stages[0].conv1.weight.detach() - before
        ).abs().max()
        self.assertGreater(float(delta), 0.0)

    def test_pruned_child_teacher_and_parent_kd_update(self):
        x, labels, teacher = self.batch()
        parent = K210MixVPR(
            spec_for_stage(CFG, "00_large")
        ).to(self.device).eval()
        parent.requires_grad_(False)

        child = K210MixVPR(
            spec_for_stage(CFG, "01_prune_w5")
        ).to(self.device)
        report = prune_one_axis(
            parent, child,
            stage_config(CFG, "01_prune_w5")["action"],
        )
        self.assertEqual(report["action"], "prune_w5")

        optimizer = torch.optim.SGD(child.parameters(), lr=1e-4)
        student = child(x).float()
        with torch.inference_mode():
            parent_desc = parent(x).float()

        task = retrieval_loss(student, labels)
        teacher_kd = ranking_kd(student, teacher, labels)
        parent_kd = ranking_kd(student, parent_desc, labels)
        loss = task + teacher_kd + 0.5 * parent_kd
        self.assertTrue(torch.isfinite(loss))

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.assert_finite_grads(child)
        optimizer.step()

class SamplerPlanTests(unittest.TestCase):
    def test_p4k2_has_four_places_two_images_each(self):
        cfg = load_config(
            "mixvpr/configs/v2_progressive_p4k2_ablation.json"
        )
        data = load_bundle(cfg)
        plan = make_plan(
            data, 8, cfg["seed"], sampler=cfg["sampler"]
        )
        for item in plan.reshape(-1, 8, 3):
            labels = item[:, 1].tolist()
            self.assertEqual(
                sorted(labels.count(label) for label in set(labels)),
                [2, 2, 2, 2],
            )
            self.assertEqual(len(set(item[:, 0].tolist())), 8)


if __name__ == "__main__":
    unittest.main()
