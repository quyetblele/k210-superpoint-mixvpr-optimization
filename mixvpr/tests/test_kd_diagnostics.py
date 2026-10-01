from __future__ import annotations

import unittest

import torch

from mixvpr_k210.diagnostics import _grad_geometry


class KDDiagnosticTests(unittest.TestCase):
    def test_grad_geometry_same_direction(self):
        p = torch.nn.Parameter(torch.tensor([1.0, -2.0]))
        task = (p.square()).sum()
        kd = 3.0 * (p.square()).sum()
        task_norm, kd_norm, cosine = _grad_geometry(
            task, kd, [p]
        )
        self.assertGreater(task_norm, 0.0)
        self.assertGreater(kd_norm, task_norm)
        self.assertAlmostEqual(cosine, 1.0, places=6)

    def test_grad_geometry_opposite_direction(self):
        p = torch.nn.Parameter(torch.tensor([1.0, -2.0]))
        task = (p.square()).sum()
        kd = -2.0 * (p.square()).sum()
        task_norm, kd_norm, cosine = _grad_geometry(
            task, kd, [p]
        )
        self.assertGreater(task_norm, 0.0)
        self.assertGreater(kd_norm, 0.0)
        self.assertAlmostEqual(cosine, -1.0, places=6)


if __name__ == "__main__":
    unittest.main()
