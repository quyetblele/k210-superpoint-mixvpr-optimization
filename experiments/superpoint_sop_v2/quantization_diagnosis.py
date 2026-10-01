"""Attribute measured K210 PTQ degradation using fixed-head hybrid diagnostics."""
import os
os.environ.update(OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
import sys
import json
import hashlib
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).parent
PREVIOUS = ROOT.parent / 'superpoint_quality_first'
sys.path.insert(0, str(PREVIOUS))
import diagnose as d

def sha(path):
    return d.digest(path)

def save(name, value):
    temporary = ROOT / (name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(ROOT / name)

def features(logits, dense):
    return d.p.features(logits, dense, d.ops)

def quality(logits, dense, pair):
    points = features(logits, dense)
    result = d.p.homography_metrics(points[0], points[1], pair['homography'], d.p.W, d.p.H)
    result.update(correct_mnn_count=result['mnn_count'] * result['mnn_precision'],
                  mean_keypoints=float(np.mean([len(v['keypoints']) for v in points])))
    return result

@torch.inference_mode()
def main():
    source_protocol = json.loads((d.BASE / 'protocol.json').read_text())
    protocol = {
        'purpose': 'Identify detector versus descriptor contribution to actual K210 PTQ loss',
        'variants': ['FP32/FP32', 'FP32/INT8', 'INT8/FP32', 'INT8/INT8'],
        'notation': 'detector source / descriptor source; hybrids are diagnosis only',
        'models': ['original_SUP_pilot', 'teacher_detector_pilot'],
        'data': 'existing 28 DEV homography pairs; no independent-test claim',
        'postprocessing': 'unchanged NMS3, threshold0.005, border4, top160, original sampling/L2',
        'GT1_used': False, 'CPU_threads': 2, 'GPU': False,
        'code_sha256': sha(__file__), 'source_protocol_sha256': sha(d.BASE / 'protocol.json'),
    }
    save('quantization_protocol.json', protocol)
    aggregate = {}
    all_rows = []
    for label, directory in [('original_SUP_pilot', PREVIOUS), ('teacher_detector_pilot', PREVIOUS / 'detector_candidate')]:
        export = json.loads((directory / 'export.json').read_text())
        deploy = json.loads((directory / 'deployment.json').read_text())
        assert deploy['simulation'] == 'PASS'
        assert sha(export['checkpoint']['path']) == deploy['checkpoint_sha256']
        assert sha(directory / 'pilot_trained.kmodel') == deploy['kmodel_sha256']
        manifest = json.loads((directory / 'simulation_manifest.json').read_text())
        model = d.p.w0.V1FR12().eval()
        model.load_state_dict(torch.load(export['checkpoint']['path'], map_location='cpu', weights_only=False)['model'])
        rows = []
        for i, row in enumerate(source_protocol['DEV']):
            pairfile = d.BASE / 'dev' / ('%03d.npz' % i)
            quantfile = directory / 'sim_outputs' / ('%03d.npz' % i)
            assert sha(pairfile) == manifest[i]['input_cache_sha256']
            assert sha(quantfile) == manifest[i]['output_sha256']
            with np.load(pairfile) as z:
                pair = {k: z[k] for k in z.files}
            fp = model(torch.from_numpy(pair['pair']))
            with np.load(quantfile) as z:
                quant = (torch.from_numpy(z['logits']), torch.from_numpy(z['desc']))
            values = {'FP32': fp, 'INT8': quant}
            for variant in protocol['variants']:
                a, b = variant.split('/')
                metrics = quality(values[a][0], values[b][1], pair)
                rows.append({'model': label, 'index': i, 'scene': row['scene'], 'variant': variant, 'metrics': metrics})
            # Same geometric point locations isolate descriptor behavior from detector changes.
            points = torch.from_numpy(pair['points'])
            for mode, (logits, dense) in values.items():
                q = d.oracle(dense, points)
                q['dustbin_probability_mean'] = float(logits.softmax(1)[:, -1].mean())
                q['logit_abs_max'] = float(logits.abs().max())
                rows.append({'model': label, 'index': i, 'scene': row['scene'], 'variant': mode + '/ORACLE', 'metrics': q})
        summary = {}
        for name in sorted({r['variant'] for r in rows}):
            rs = [r for r in rows if r['variant'] == name]
            summary[name] = {k: float(np.mean([r['metrics'][k] for r in rs])) for k in rs[0]['metrics']}
        old = json.loads((directory / 'int8_quality.json').read_text())
        for mode, key in [('FP32', 'fp32'), ('INT8', 'int8')]:
            for metric in ['mnn_precision', 'correct_mnn_count', 'repeatability_3px']:
                assert abs(summary[mode + '/' + mode][metric] - old[key][metric]) < 1e-6
        aggregate[label] = {'checkpoint_sha256': deploy['checkpoint_sha256'], 'kmodel_sha256': deploy['kmodel_sha256'], 'variants': summary}
        all_rows.extend(rows)
        print(label, json.dumps(summary), flush=True)
        del model
    save('quantization_diagnosis_rows.json', all_rows)
    save('quantization_diagnosis.json', {'status': 'COMPLETE', 'old_metrics_reproduced': True, 'models': aggregate})

if __name__ == '__main__':
    main()
