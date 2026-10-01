"""Evaluate actual compiled outputs with exactly the FP32 DEV metric implementation."""
import json
import numpy as np
from .runtime import torch
from .settings import ROOT
from .protocol import protocol
from .metrics import evaluate
from .io import sha, put
from .onnx_export import deployment_dir, load_model

def run(candidate):
    (model, res, checkpoint) = load_model(candidate)
    dest = deployment_dir(candidate)
    simulation = json.loads((dest / 'simulation.json').read_text())
    assert simulation['status'] == 'PASS'
    assert simulation['checkpoint_sha256'] == sha(checkpoint)
    assert simulation['kmodel_sha256'] == sha(dest / 'model.kmodel')
    for row in simulation['rows']:
        assert sha(ROOT / res / 'dev' / f"{row['index']:03d}.npz") == row['input_sha256']
        assert sha(dest / 'sim_outputs' / f"{row['index']:03d}.npz") == row['output_sha256']

    class CachedOutputs(torch.nn.Module):

        def __init__(self):
            super().__init__()
            self.index = 0

        def forward(self, x):
            with np.load(dest / 'sim_outputs' / f'{self.index:03d}.npz') as z:
                values = tuple((torch.from_numpy(z[k].copy()) for k in ['logits', 'desc']))
            self.index += 1
            return values
    fp32 = evaluate(model, res, protocol()['rows']['dev'])
    int8 = evaluate(CachedOutputs(), res, protocol()['rows']['dev'])
    ratios = {k: int8['macro'][k] / max(fp32['macro'][k], 1e-12) for k in ['correct_mnn_count', 'mnn_precision', 'repeatability', 'coverage']}
    cosine = []
    with torch.inference_mode():
        for i in range(len(protocol()['rows']['dev'])):
            with np.load(ROOT / res / 'dev' / f'{i:03d}.npz') as z:
                x = torch.from_numpy(z['pair'])
                mask = torch.from_numpy(z['cell']).bool()
            (_, a) = model(x)
            with np.load(dest / 'sim_outputs' / f'{i:03d}.npz') as z:
                b = torch.from_numpy(z['desc'])
            v = torch.nn.functional.cosine_similarity(a, b, dim=1)
            cosine.append(float(v[mask].mean()))
    put(dest / 'int8_quality.json', {'status': 'MEASURED', 'backend': 'actual nncase K210 simulator, no fake quantization', 'checkpoint_sha256': sha(checkpoint), 'kmodel_sha256': sha(dest / 'model.kmodel'), 'fp32': fp32, 'int8': int8, 'macro_retention': ratios, 'mean_valid_dense_descriptor_cosine': float(np.mean(cosine)), 'scope': 'synthetic homography DEV screening; not real-map pose or board qualification', 'release_gate': 'NOT_DEFINED; no post-hoc release threshold', 'board_latency_and_actual_peak_SRAM': 'NOT_MEASURED'})
