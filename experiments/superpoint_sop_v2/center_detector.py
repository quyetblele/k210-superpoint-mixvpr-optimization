"""Softmax-invariant detector reparameterization, then actual K210 PTQ test."""
import os
os.environ.update(OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2')
import sys
import json
import shutil
import subprocess
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).parent
PREVIOUS = ROOT.parent / 'superpoint_quality_first'
SOURCE = PREVIOUS / 'detector_candidate'
DEST = ROOT / 'centered_detector'
sys.path.insert(0, str(PREVIOUS))
import diagnose as d
TPY = '/home/quyet/k210_lab/.venv-mixvpr-cuda/bin/python'
KPY = '/home/quyet/miniconda3/envs/k210/bin/python'

def save(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)

@torch.inference_mode()
def prepare():
    DEST.mkdir(exist_ok=True)
    parent = json.loads((SOURCE / 'export.json').read_text())['checkpoint']
    assert d.digest(parent['path']) == parent['sha256']
    protocol = {
        'purpose': 'Remove input-dependent common-mode detector logits before INT8 quantization',
        'parent_checkpoint': parent,
        'transformation': 'For final 1x1 detector conv only: W_j <- W_j-mean_k(W_k); b_j <- b_j-mean_k(b_k), k over all65 outputs including dustbin',
        'proof': 'logits_centered_j(x)=logits_j(x)-mean_k(logits_k(x)); softmax is invariant to a common scalar per spatial cell',
        'unchanged': ['encoder', 'descriptor head', 'head dimensions', 'graph operator topology', 'postprocessing', 'TRAIN calibration sources'],
        'not_training': True,
        'parity_gate': 'all28 DEV pairs: max detector softmax abs error<=2e-5; descriptor tensors identical; aggregate precision/correct MNN differ<=1e-6',
        'retention_gate': 'actual kmodel correct MNN>=0.85*original FP32; precision and repeatability no worse than baseline INT8 by >0.02',
        'scope': 'fixed existing DEV; research numerical gate, not application acceptance',
        'code_sha256': d.digest(__file__),
    }
    if (DEST / 'transformation_protocol.json').exists():
        assert json.loads((DEST / 'transformation_protocol.json').read_text()) == protocol
    else:
        save(DEST / 'transformation_protocol.json', protocol)
    checkpoint = torch.load(parent['path'], map_location='cpu', weights_only=False)
    original = d.p.w0.V1FR12().eval()
    original.load_state_dict(checkpoint['model'])
    centered = d.p.w0.V1FR12().eval()
    centered.load_state_dict(checkpoint['model'])
    for param in [centered.convPb.weight, centered.convPb.bias]:
        value = param.double()
        param.copy_((value - value.mean(0, keepdim=True)).float())
    rows = []
    source_protocol = json.loads((d.BASE / 'protocol.json').read_text())
    for i in range(len(source_protocol['DEV'])):
        with np.load(d.BASE / 'dev' / ('%03d.npz' % i)) as z:
            pair = {k: z[k] for k in z.files}
        tensor = torch.from_numpy(pair['pair'])
        logits, desc = original(tensor)
        adjusted, adjusted_desc = centered(tensor)
        error = float((logits.softmax(1) - adjusted.softmax(1)).abs().max())
        assert error <= 2e-5, error
        assert torch.equal(desc, adjusted_desc)
        before = d.metrics(logits, desc, pair)
        after = d.metrics(adjusted, adjusted_desc, pair)
        rows.append({'index': i, 'probability_max_abs_error': error, 'original_logit_min': float(logits.min()),
                     'original_logit_max': float(logits.max()), 'centered_logit_min': float(adjusted.min()),
                     'centered_logit_max': float(adjusted.max()), 'before': before, 'after': after})
    for metric in ['mnn_precision', 'correct_mnn_count', 'repeatability_3px']:
        assert abs(np.mean([r['before'][metric] for r in rows]) - np.mean([r['after'][metric] for r in rows])) <= 1e-6, metric
    checkpoint['model'] = centered.state_dict()
    checkpoint['parent_checkpoint_sha256'] = parent['sha256']
    checkpoint['deployment_transform'] = protocol['transformation']
    torch.save(checkpoint, DEST / 'centered.pt')
    identity = dict(parent, path=str(DEST / 'centered.pt'), sha256=d.digest(DEST / 'centered.pt'))
    save(DEST / 'protocol.json', {'checkpoint_identities': {'DETECTOR_CENTERED': identity}, 'not_final_candidate': True,
                                'selection': 'no retraining/reselection; fixed parent plus validated softmax-invariant transform'})
    save(DEST / 'diagnosis.json', {'selected_existing_pilot': 'DETECTOR_CENTERED'})
    save(DEST / 'transformation_parity.json', {'status': 'PASS', 'pairs': len(rows), 'descriptor_tensors': 'IDENTICAL',
                                             'max_probability_abs_error': max(r['probability_max_abs_error'] for r in rows), 'rows': rows})
    for name in ['prepare_deployment.py', 'compile_simulate.py', 'evaluate_simulator.py']:
        shutil.copy2(SOURCE / name, DEST / name)
    print('TRANSFORM_PARITY_PASS', max(r['probability_max_abs_error'] for r in rows), flush=True)

def main():
    prepare()
    env = dict(os.environ, PYTHONPATH=str(PREVIOUS))
    for python, script, logname in [(TPY, 'prepare_deployment.py', 'export.log'),
                                    (KPY, 'compile_simulate.py', 'compile.log'),
                                    (TPY, 'evaluate_simulator.py', 'evaluation.log')]:
        with (DEST / logname).open('w') as log:
            subprocess.run([python, str(DEST / script)], env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    result = json.loads((DEST / 'int8_quality.json').read_text())
    before = json.loads((SOURCE / 'int8_quality.json').read_text())
    quant = result['int8']
    passed = quant['correct_mnn_count'] >= .85 * before['fp32']['correct_mnn_count'] and quant['mnn_precision'] >= before['int8']['mnn_precision'] - .02 and quant['repeatability_3px'] >= before['int8']['repeatability_3px'] - .02
    summary = {'status': 'COMPLETE', 'gate_pass': passed, 'parent_fp32': before['fp32'], 'parent_int8': before['int8'],
               'centered_int8': quant, 'retention': result['retention'],
               'scope': 'actual K210 kmodel; same architecture and mathematically equivalent FP32 detector; no board quality claim'}
    save(DEST / 'summary.json', summary)
    print(json.dumps(summary), flush=True)

if __name__ == '__main__':
    main()
