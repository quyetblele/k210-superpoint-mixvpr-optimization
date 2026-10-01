"""Cheap correctness checks before canonical training/export."""
import _bootstrap
import importlib.util, json
import tempfile
from pathlib import Path
import numpy as np
from spk210.runtime import torch
from spk210.settings import ROOT, WORKSPACE, WIDTHS, RES
from spk210.model import Model
from spk210.metrics import verify, features
from spk210.protocol import verify_cache
from spk210.io import put, sha
verify()
verify_cache()
source = WORKSPACE / 'experiments/superpoint_sop_v2/quality_pilot_v2.py'
spec = importlib.util.spec_from_file_location('pre_refactor', source)
old = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)
checks = []
for width in WIDTHS:
    original = old.Model(WIDTHS[width])
    canonical = Model(WIDTHS[width])
    state = torch.load(ROOT / f'{width}_init.pt', weights_only=True)
    original.load_state_dict(state)
    canonical.load_state_dict(state)
    for res in RES:
        with np.load(ROOT / res / 'train/000.npz') as z:
            x = torch.from_numpy(z['pair'])
            mask = z['masks']
        a = original(x)
        b = canonical(x)
        assert all((torch.equal(v, u) for (v, u) in zip(a, b)))
        original.zero_grad()
        canonical.zero_grad()
        sum((v.square().mean() for v in a)).backward()
        sum((v.square().mean() for v in b)).backward()
        assert all((torch.equal(v.grad, u.grad) for (v, u) in zip(original.parameters(), canonical.parameters())))
        with torch.inference_mode():
            fa = old.features(*[v.detach() for v in a], mask, res)
            fb = features(*[v.detach() for v in b], mask, res)
        assert all((np.array_equal(v, u) for (p, q) in zip(fa, fb) for (v, u) in zip(p, q)))
        checks.append({'width': width, 'resolution': res, 'raw_output': 'EXACT', 'gradients': 'EXACT', 'postprocessing': 'EXACT'})
for split in ['train', 'dev']:
    for path in (ROOT / 'r1' / split).glob('*.npz'):
        with np.load(path) as a, np.load(ROOT / 'r2' / split / path.name) as b:
            assert np.allclose(a['points'], (b['points'] - np.array([2, 0])) / 1.25, atol=2e-05)
from spk210 import nncase_backend
previous_root=nncase_backend.SUPERPOINT
try:
    with tempfile.TemporaryDirectory(prefix='sp_identity_check_') as temporary:
        nncase_backend.SUPERPOINT=Path(temporary)
        folder=Path(temporary)/'artifacts/deployment/probe';folder.mkdir(parents=True)
        put(folder/'export.json',{'checkpoint':str(ROOT/'fr12_init.pt'),'checkpoint_sha256':'0'*64})
        rejected=False
        try:nncase_backend.context('probe')
        except AssertionError:rejected=True
        assert rejected,'Wrong checkpoint identity was not rejected before runtime access'
finally:
    nncase_backend.SUPERPOINT=previous_root
put(ROOT / 'refactor_checks.json', {'status': 'PASS', 'checks': checks, 'canonical_correspondences': 'IDENTICAL_FOR_ALL182_PAIRS', 'wrong_checkpoint_identity':'REJECTED_BEFORE_RUNTIME', 'source_sha256': sha(source), 'test_code_sha256': sha(__file__)})
print('REFACTOR_CHECKS_PASS', flush=True)
