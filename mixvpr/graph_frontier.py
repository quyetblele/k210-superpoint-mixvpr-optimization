"""Stage-1 resource probes only: no optimizer, training, retrieval eval or TEST."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import traceback
from collections import Counter

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'mixvpr/artifacts/stage1_frontier'
BASE = Path('/home/quyet/training_recovery/final_gate2')
CAL = BASE / 'calibration_train_f32.npy'
KPY = '/home/quyet/miniconda3/envs/k210/bin/python'
SEED = 20260912
# C=last backbone width, H=token MLP hidden, D=depth, P=channel projection.
# Descriptor dimension stays 512; rows=512/P. Early widths and 100 tokens fixed.
SPECS = {
    'baseline': (160, 96, 4, 128),
    'late192': (192, 96, 4, 128),
    'late224': (224, 96, 4, 128),
    'mixer128': (160, 128, 4, 128),
    'depth6': (160, 96, 6, 128),
    'projection256': (160, 96, 4, 256),
    'late192_mixer128': (192, 128, 4, 128),
}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def put(p, value):
    Path(p).write_text(json.dumps(value, indent=2) + '\n')


def compile_one(name):
    import nncase
    import _nncase
    from importlib.metadata import version
    d = OUT / name
    r = {k: 'NOT_RUN' for k in ('import', 'ptq', 'compile', 'gencode', 'sim')}
    stage = 'import'
    try:
        assert version('nncase') == '1.8.0.20220929'
        assert _nncase.__version__ == '1.8.0-55be52f'
        r['environment'] = {'nncase': version('nncase'), 'runtime': _nncase.__version__}
        options = nncase.CompileOptions()
        options.target = 'k210'
        options.quant_type = options.w_quant_type = 'uint8'
        options.dump_asm = True
        options.dump_dir = str(d / 'dump')
        c = nncase.Compiler(options)
        c.import_onnx((d / 'model.onnx').read_bytes(), nncase.ImportOptions())
        r[stage] = 'PASS'
        stage = 'ptq'
        data = np.load(CAL)
        q = nncase.PTQTensorOptions()
        q.samples_count = len(data)
        q.set_tensor_data(data.tobytes())
        c.use_ptq(q)
        r[stage] = 'PASS'
        stage = 'compile'
        c.compile()
        r[stage] = 'PASS'
        stage = 'gencode'
        blob = c.gencode_tobytes()
        assert len(blob) > 0
        (d / 'model.kmodel').write_bytes(blob)
        r.update(gencode='PASS', model_bytes=len(blob), kmodel_sha256=sha(d / 'model.kmodel'))
        info = (d / 'dump/kmodel_info.txt').read_text()
        r['compiler_memory'] = {k: int(v) for k, v in re.findall(
            r'(input|output|data|MODEL|TOTAL):?\s+[^\n]*?\((\d+) B\)', info)}
        assert 'TOTAL' in r['compiler_memory'], info
        mapping = {}
        for backend in ('stackvm', 'k210'):
            content = '\n'.join(p.read_text() for p in (d / 'dump' / backend).rglob('runtime_ops.txt'))
            mapping[backend] = dict(Counter(re.findall(r'\[([A-Za-z0-9_]+)\]', content)))
        r['mapping'] = mapping
        r['kpu_conv'] = mapping['k210'].get('KPUConv2D', 0)
        r['cpu_conv'] = mapping['stackvm'].get('Conv2D', 0)
        stage = 'sim'
        sim = nncase.Simulator()
        sim.load_model(blob)
        assert sim.get_input_tensor(0).dtype == np.dtype('float32')
        assert sim.get_output_tensor(0).dtype == np.dtype('float32')
        outputs = []
        for i in (0, 24, 56):
            sim.set_input_tensor(0, nncase.RuntimeTensor.from_numpy(np.ascontiguousarray(data[i:i+1])))
            sim.run()
            y = sim.get_output_tensor(0).to_numpy().copy()
            assert y.shape == (1, 512) and np.isfinite(y).all() and np.linalg.norm(y) > 1e-12
            outputs.append(y)
        np.save(d / 'sim_outputs.npy', np.concatenate(outputs))
        r.update(sim='PASS', sim_input_indices=[0, 24, 56], sim_output_shape=[3, 512],
                 simulator_scope='Execution/shape/finite/nonzero only, not retrieval quality or board performance')
        # nncase maps the 2*D mixer Linear layers and both projections to KPU too.
        r['expected_kpu_conv'] = 10 + 2 * SPECS[name][2] + 2
        r['graph_pass'] = r['kpu_conv'] == r['expected_kpu_conv'] and r['cpu_conv'] == 0
        r['reserve_screen_pass'] = r['graph_pass'] and r['compiler_memory']['TOTAL'] <= 3 * 1024**2
    except Exception as exc:
        r.update(failed_stage=stage, error=str(exc), traceback=traceback.format_exc(), graph_pass=False,
                 reserve_screen_pass=False)
        r[stage] = 'FAIL'
    put(d / 'compile.json', r)
    print(name, {k: r.get(k) for k in ('compile', 'gencode', 'sim', 'error')}, flush=True)


def prepare_and_run():
    import torch
    import onnx
    import onnxruntime as ort
    import torch.nn as nn
    sys.path.insert(0, str(ROOT / 'experiments/edge_vpr_students'))
    from sq240_resource_probe_export import SQ240Probe, PreL2, canonicalize, conv_linear_macs
    from s512_br1_sq240_m100_c160h96_model import S512BR1SQ240M100C160H96

    # Refuse overwriting completed or partial experiments.
    OUT.mkdir(parents=True, exist_ok=False)
    identity = json.loads((BASE / 'identity.json').read_text())
    assert sha(BASE / 'candidate.pt') == identity['checkpoint_sha256']
    calibration = json.loads((BASE / 'calibration.json').read_text())
    assert sha(CAL) == calibration['tensor_sha256'] and calibration['split'] == 'TRAIN'
    locked = {str(p): sha(p) for p in [BASE / n for n in (
        'candidate.pt', 'identity.json', 'student_canonical.onnx', 'student_c160h96_trained.kmodel',
        'calibration_train_f32.npy', 'deployment.json', 'gt1_fp32.json')]}
    for f in ('freeze_manifest.json', 'final_test/results.json'):
        p = ROOT / 'superpoint/artifacts/full_training' / f
        locked[str(p)] = sha(p)
    source_paths = [Path(__file__), ROOT / 'experiments/edge_vpr_students/sq240_resource_probe_export.py',
                    ROOT / 'experiments/edge_vpr_students/s512_br1_sq240_m100_c160h96_model.py']
    put(OUT / 'protocol.json', {
        'seed': SEED, 'specs_C_H_D_P': SPECS, 'input': [1, 3, 240, 240], 'tokens': 100,
        'early_widths': [16, 24, 64, 112], 'descriptor': 512, 'cpu_postprocess': 'L2 eps1e-12',
        'training': False, 'test_opened': False, 'new_weights': 'deterministic random initialization; no optimizer steps',
        'calibration_sha256': sha(CAL), 'calibration_count': 64,
        'parity': {'atol': 1e-4, 'rtol': 1e-3, 'samples': [0, 24, 56]},
        'compile_timeout_seconds': 600,
        'graph_gate': 'export parity + compile + gencode + simulator execution; 10 backbone +2*D mixer +2 projection KPU Conv, 0 CPU Conv; CPU normalization/data movement allowed',
        'shortlist_screen': 'compiler TOTAL <=3MiB, provisional PC resource screen inherited from project; not full-system peak or reserve proof',
        'scope': 'Six higher-capacity graphs, not proven stronger quality. No model promotion, training, KD, TEST or board measurement.',
        'locked_files': locked, 'sources': {str(p): sha(p) for p in source_paths},
    })
    data = np.load(CAL)
    assert data.shape == (64, 3, 240, 240)
    records = []
    for name, (channels, hidden, depth, projection) in SPECS.items():
        d = OUT / name
        d.mkdir()
        torch.manual_seed(SEED)
        if name == 'baseline':
            m = S512BR1SQ240M100C160H96()
            checkpoint = torch.load(BASE / 'candidate.pt', map_location='cpu', weights_only=False)
            m.load_state_dict(checkpoint['model_state_dict'], strict=True)
        else:
            m = SQ240Probe(10, depth, hidden, channels)
            if projection != 128:
                m.channel_projection = nn.Linear(channels, projection)
                m.row_projection = nn.Linear(100, 512 // projection)
        m.eval()
        shapes = {}
        handles = []
        for module_name, module in m.named_children():
            def hook(_module, _args, y, key=module_name):
                shapes[key] = list(y.shape)
            handles.append(module.register_forward_hook(hook))
        macs = conv_linear_macs(m, torch.from_numpy(data[:1]))
        for h in handles:
            h.remove()
        row = {'name': name, 'architecture': {'widths': [16, 24, 64, 112, channels], 'C': channels,
               'H': hidden, 'D': depth, 'P': projection, 'rows': 512 // projection, 'tokens': 100},
               'params': sum(p.numel() for p in m.parameters()), 'conv_linear_MACs': macs,
               'MAC_scope': 'batch1 Conv2d+Linear only; excludes normalization/pooling/nonlinear ops',
               'module_shapes': shapes, 'quality': 'HISTORICAL_ONLY' if name == 'baseline' else 'NOT_EVALUATED_UNTRAINED'}
        if name == 'baseline':
            shutil.copyfile(BASE / 'student_canonical.onnx', d / 'model.onnx')
        else:
            torch.onnx.export(PreL2(m), torch.from_numpy(data[:1]), str(d / 'raw.onnx'),
                              opset_version=13, dynamo=False, input_names=['normalized_rgb'], output_names=['raw_descriptor'])
            graph = onnx.shape_inference.infer_shapes(onnx.load(d / 'raw.onnx'), strict_mode=True)
            onnx.save(graph, d / 'raw.onnx')
            row['canonicalization'] = canonicalize(d / 'raw.onnx', d / 'model.onnx')
        onnx.checker.check_model(onnx.load(d / 'model.onnx'))
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 2
        opts.inter_op_num_threads = 1
        session = ort.InferenceSession(str(d / 'model.onnx'), opts, providers=['CPUExecutionProvider'])
        errors = []
        for i in (0, 24, 56):
            x = data[i:i+1]
            with torch.inference_mode():
                expected = m.raw(torch.from_numpy(x)).numpy()
            actual = session.run(None, {'normalized_rgb': x})[0]
            np.testing.assert_allclose(actual, expected, atol=1e-4, rtol=1e-3)
            errors.append(float(np.abs(actual - expected).max()))
        row.update(export='PASS', parity_max_abs=max(errors), onnx_sha256=sha(d / 'model.onnx'))
        put(d / 'export.json', row)
        with (d / 'compile.log').open('w') as log:
            try:
                process = subprocess.run([KPY, str(Path(__file__).resolve()), 'compile', name],
                                         stdout=log, stderr=subprocess.STDOUT, cwd=d, timeout=600)
                row['returncode'] = process.returncode
            except subprocess.TimeoutExpired:
                row.update(returncode=-1, error='Compiler timeout (600s)')
        if (d / 'compile.json').exists():
            row.update(json.loads((d / 'compile.json').read_text()))
        else:
            row.update(graph_pass=False, reserve_screen_pass=False, error=row.get('error', 'Process failed; inspect compile.log'))
        records.append(row)
        put(OUT / 'results.json', records)
        print(name, {k: row.get(k) for k in ('params', 'compile', 'gencode', 'sim', 'compiler_memory', 'error')}, flush=True)
    assert all(sha(p) == h for p, h in locked.items()), 'Protected artifact changed'
    put(OUT / 'verification.json', {'protected_hashes': 'PASS', 'count': len(locked), 'candidates': len(records),
                                  'training': False, 'test_opened': False})


if __name__ == '__main__':
    os.environ.update(OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
    if len(sys.argv) > 1 and sys.argv[1] == 'compile':
        compile_one(sys.argv[2])
    else:
        prepare_and_run()
