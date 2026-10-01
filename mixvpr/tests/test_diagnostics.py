from __future__ import annotations
import unittest

import torch

from mixvpr_k210.diagnostics import (
    _grad_geometry,
    _teacher_margin,
)


class DiagnosticUnitTests(unittest.TestCase):
    def test_gradient_cosine_same_direction(self):
        p = torch.nn.Parameter(
            torch.tensor([1.0, -2.0, 3.0])
        )
        task = p.sum()
        kd = 2.0 * p.sum()
        task_norm, kd_norm, cosine = _grad_geometry(
            task, kd, [p]
        )
        self.assertGreater(task_norm, 0.0)
        self.assertGreater(kd_norm, task_norm)
        self.assertAlmostEqual(cosine, 1.0, places=6)

    def test_gradient_cosine_conflict(self):
        p = torch.nn.Parameter(
            torch.tensor([1.0, -2.0, 3.0])
        )
        task = p.sum()
        kd = -p.sum()
        _task_norm, _kd_norm, cosine = _grad_geometry(
            task, kd, [p]
        )
        self.assertAlmostEqual(cosine, -1.0, places=6)

    def test_teacher_margin_has_positive_pairs(self):
        teacher = torch.tensor([
            [1.0, 0.0],
            [0.9, 0.1],
            [0.0, 1.0],
            [0.1, 0.9],
        ])
        teacher = torch.nn.functional.normalize(
            teacher, dim=1
        )
        labels = torch.tensor([0, 0, 1, 1])
        result = _teacher_margin(teacher, labels)
        self.assertGreater(result["mean"], 0.0)
        self.assertEqual(
            result["positive_fraction"], 1.0
        )


if __name__ == "__main__":
    unittest.main()
