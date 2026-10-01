"""Fixed-checkpoint, equal-size TRAIN calibration experiment for real nncase PTQ."""
import json
import hashlib
import shutil
import subprocess
import os
from pathlib import Path
import numpy as np

ROOT = Path(__file__).parent
SOURCE = ROOT.parent / 'superpoint_quality_first' / 'detector_candidate'
BASE = Path('/home/quyet/training_recovery/sp_ab')
KPY = '/home/quyet/miniconda3/envs/k210/bin/python'
TPY = '/home/quyet/k210_lab/.venv-mixvpr-cuda/bin/python'

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def save(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)

def main():
    folder = ROOT / 'calibration_ab'
    folder.mkdir(exist_ok=True)
    export = json.loads((SOURCE / 'export.json').read_text())
    manifest = json.loads((SOURCE / 'calibration.json').read_text())['manifest']
    train_protocol = json.loads((BASE / 'protocol.json').read_text())
    assert len(manifest) == 28
    protocol = {
        'purpose': 'Test calibration distribution coverage before considering QAT',
        'checkpoint_sha256': export['checkpoint']['sha256'],
        'variants': {'original_repeated': 'same 28 TRAIN sources, original view duplicated: 56 samples',
                     'original_and_warped': 'same 28 TRAIN sources, original and cached geometric/photometric view: 56 samples'},
        'fixed': ['model', 'compiler', 'sample count', 'source images', 'NMS', 'threshold', 'top-k', 'evaluation'],
        'calibration_only_TRAIN': True, 'evaluation': 'existing DEV, no GT1',
        'research_gate': 'augmented calibration correct MNN >= 0.85 * FP32 correct MNN, with precision and repeatability no worse than control by >0.02',
        'gate_scope': 'diagnostic retention target, not application acceptance requirement',
        'code_sha256': sha(__file__),
    }
    if (folder / 'protocol.json').exists():
        assert json.loads((folder / 'protocol.json').read_text()) == protocol
    else:
        save(folder / 'protocol.json', protocol)
    results = {}
    for variant in protocol['variants']:
        dest = folder / variant
        dest.mkdir(exist_ok=True)
        if not (dest / 'int8_quality.json').exists():
            tensors = []
            records = []
            for item in manifest:
                i = item['index']
                path = BASE / 'train' / ('%03d.npz' % i)
                assert sha(path) == item['cache_sha256']
                assert train_protocol['TRAIN'][i]['sha256'] == item['source_sha256']
                assert item['source_sha256'] not in {r['sha256'] for r in train_protocol['DEV']}
                with np.load(path) as z:
                    pair = z['pair']
                    tensors.extend([pair[0].copy(), pair[0 if variant == 'original_repeated' else 1].copy()])
                records.append(item)
            np.save(dest / 'calibration_train.npy', np.stack(tensors).astype(np.float32))
            save(dest / 'calibration.json', {'split': 'TRAIN', 'count': len(tensors), 'manifest': records,
                                           'tensor_sha256': sha(dest / 'calibration_train.npy'), 'variant': variant})
            shutil.copy2(SOURCE / 'pilot_canonical.onnx', dest / 'pilot_canonical.onnx')
            shutil.copy2(SOURCE / 'export.json', dest / 'export.json')
            for name in ['compile_simulate.py', 'evaluate_simulator.py']:
                shutil.copy2(SOURCE / name, dest / name)
            with (dest / 'compile.log').open('w') as log:
                subprocess.run([KPY, str(dest / 'compile_simulate.py')], stdout=log, stderr=subprocess.STDOUT, check=True)
            env = dict(os.environ, PYTHONPATH=str(SOURCE.parent))
            with (dest / 'evaluation.log').open('w') as log:
                subprocess.run([TPY, str(dest / 'evaluate_simulator.py')], env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        result = json.loads((dest / 'int8_quality.json').read_text())
        deployment = json.loads((dest / 'deployment.json').read_text())
        assert deployment['checkpoint_sha256'] == protocol['checkpoint_sha256']
        results[variant] = {k: result[k] for k in ['fp32', 'int8', 'retention']}
        print('CALIBRATION_COMPLETE', variant, result['int8']['correct_mnn_count'], flush=True)
    control, change = results['original_repeated'], results['original_and_warped']
    a, b = control['int8'], change['int8']
    passed = (b['correct_mnn_count'] >= .85 * change['fp32']['correct_mnn_count']
              and b['mnn_precision'] >= a['mnn_precision'] - .02
              and b['repeatability_3px'] >= a['repeatability_3px'] - .02)
    summary = {'status': 'COMPLETE', 'gate_pass': passed, 'results': results,
               'next': 'validate downstream retention' if passed else 'calibration coverage alone insufficient; inspect numerical head behavior and training before QAT'}
    save(folder / 'summary.json', summary)
    print(json.dumps(summary), flush=True)

if __name__ == '__main__':
    main()
