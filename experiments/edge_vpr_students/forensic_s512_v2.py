"""Forensic A(8) vs B(4+reload+4), no real-training checkpoint mutation."""
import json,runpy,torch
torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
# The frozen equivalence test already uses the persisted plan and only _equiv_tmp.
runpy.run_path('/home/quyet/k210_lab/experiments/edge_vpr_students/test_s512_v2_equivalence.py',run_name='__main__')
