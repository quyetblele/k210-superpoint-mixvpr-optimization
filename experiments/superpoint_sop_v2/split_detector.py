"""Exact FP32 head partition, independently quantized 64-way and dustbin outputs."""
import os
os.environ.update(OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2')
import argparse
import json
import hashlib
import shutil
import subprocess
import resource
from pathlib import Path
import numpy as np

ROOT = Path(__file__).parent
OUT = ROOT / 'split_detector'
PREVIOUS = ROOT.parent / 'superpoint_quality_first'
SOURCE = PREVIOUS / 'detector_candidate'
BASE = Path('/home/quyet/training_recovery/sp_ab')
TPY = '/home/quyet/k210_lab/.venv-mixvpr-cuda/bin/python'
KPY = '/home/quyet/miniconda3/envs/k210/bin/python'

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def save(path, value):
    t = path.with_suffix(path.suffix + '.tmp')
    t.write_text(json.dumps(value, indent=2) + '\n')
    t.replace(path)

def prepare():
    import sys
    import torch
    from torch import nn
    import onnx
    import onnxruntime as ort
    sys.path.insert(0, str(PREVIOUS))
    import diagnose as d
    OUT.mkdir(exist_ok=True)
    parent = json.loads((SOURCE / 'export.json').read_text())['checkpoint']
    assert sha(parent['path']) == parent['sha256']
    protocol = {'purpose': 'Separate final detector point and dustbin quantization ranges',
                'change': '65-output 1x1 conv becomes 64-output and 1-output conv with exact weight slices; CPU concatenates logits before unchanged softmax/NMS',
                'parent': parent, 'not_training': True, 'parameter_count_unchanged': True,
                'parity_gate': 'original versus partitioned ONNX: raw heads allclose2e-4; softmax max abs<=2e-5',
                'quality_gate': 'correct MNN>=85% of parent FP32 and precision/repeatability not worse than baseline INT8 by >0.02',
                'calibration': 'same28 TRAIN originals as parent baseline', 'GT1_used': False,
                'code_sha256': sha(__file__), 'scope': 'numerical deployment probe, not final application acceptance'}
    save(OUT / 'protocol.json', protocol)
    state = torch.load(parent['path'], map_location='cpu', weights_only=False)['model']
    original = d.p.w0.V1FR12().eval()
    original.load_state_dict(state)

    class Split(nn.Module):
        def __init__(self):
            super().__init__()
            self.core = d.p.w0.V1FR12()
            self.core.load_state_dict(state)
            self.points = nn.Conv2d(128, 64, 1)
            self.dustbin = nn.Conv2d(128, 1, 1)
            with torch.no_grad():
                self.points.weight.copy_(self.core.convPb.weight[:64])
                self.points.bias.copy_(self.core.convPb.bias[:64])
                self.dustbin.weight.copy_(self.core.convPb.weight[64:])
                self.dustbin.bias.copy_(self.core.convPb.bias[64:])
            self.core.convPb = nn.Identity()

        def forward(self, image):
            hidden, descriptor = self.core.forward_raw(image)
            return self.points(hidden), self.dustbin(hidden), descriptor

    model = Split().eval()
    assert sum(p.numel() for p in model.parameters()) == sum(p.numel() for p in original.parameters())
    graphfile = OUT / 'split.onnx'
    torch.onnx.export(model, torch.zeros(1, 1, 192, 256), str(graphfile), opset_version=13, dynamo=False,
                      input_names=['image'], output_names=['point_logits', 'dustbin_logit', 'descriptor_map'], do_constant_folding=False)
    graph = onnx.shape_inference.infer_shapes(onnx.load(graphfile), strict_mode=True)
    onnx.checker.check_model(graph)
    assert {n.op_type for n in graph.graph.node} <= {'Conv', 'Relu', 'MaxPool'}
    onnx.save(graph, graphfile)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(graphfile), options, providers=['CPUExecutionProvider'])
    rows = []
    with torch.inference_mode():
        for i in range(28):
            with np.load(BASE / 'dev' / ('%03d.npz' % i)) as z:
                pair = z['pair']
            for view, image in enumerate(pair):
                x = image[None].copy()
                a, b = original(torch.from_numpy(x))
                result = session.run(None, {'image': x})
                logits = np.concatenate(result[:2], axis=1)
                assert np.allclose(a.numpy(), logits, atol=2e-4, rtol=2e-4)
                assert np.allclose(b.numpy(), result[2], atol=2e-4, rtol=2e-4)
                error = float((a.softmax(1) - torch.from_numpy(logits).softmax(1)).abs().max())
                assert error <= 2e-5
                rows.append({'index': i, 'view': view, 'probability_max_abs_error': error})
    for name in ['calibration_train.npy', 'calibration.json', 'evaluate_simulator.py']:
        shutil.copy2(SOURCE / name, OUT / name)
    save(OUT / 'export.json', {'status': 'PASS', 'checkpoint': parent, 'onnx_sha256': sha(graphfile),
         'input': [1, 1, 192, 256], 'outputs': [[1, 64, 24, 32], [1, 1, 24, 32], [1, 256, 24, 32]],
         'postprocessing': 'CPU concatenate point_logits and dustbin_logit, then unchanged65-way softmax; descriptor sampling/L2 unchanged',
         'parity': rows, 'quality_scope': 'same parent FP32 model, partitioned deployment graph'})
    print('SPLIT_PARITY_PASS', max(r['probability_max_abs_error'] for r in rows), flush=True)

def compile_simulate():
    import nncase
    import _nncase
    from importlib.metadata import version
    assert version('nncase') == '1.8.0.20220929' and _nncase.__version__ == '1.8.0-55be52f'
    export = json.loads((OUT / 'export.json').read_text())
    assert sha(OUT / 'split.onnx') == export['onnx_sha256']
    opts = nncase.CompileOptions()
    opts.target = 'k210'
    opts.quant_type = 'uint8'
    opts.w_quant_type = 'uint8'
    opts.dump_dir = str(OUT / 'compiler_dump')
    opts.dump_asm = True
    compiler = nncase.Compiler(opts)
    compiler.import_onnx((OUT / 'split.onnx').read_bytes(), nncase.ImportOptions())
    calibration = np.load(OUT / 'calibration_train.npy')
    assert sha(OUT / 'calibration_train.npy') == json.loads((OUT / 'calibration.json').read_text())['tensor_sha256']
    q = nncase.PTQTensorOptions()
    q.samples_count = len(calibration)
    q.set_tensor_data(calibration.tobytes())
    compiler.use_ptq(q)
    compiler.compile()
    blob = compiler.gencode_tobytes()
    assert blob
    (OUT / 'pilot_trained.kmodel').write_bytes(blob)
    report = {'backend': 'actual nncase K210 Simulator', 'checkpoint_sha256': export['checkpoint']['sha256'],
              'deployment_graph_sha256': export['onnx_sha256'], 'kmodel_sha256': sha(OUT / 'pilot_trained.kmodel'),
              'kmodel_bytes': len(blob), 'ptq': 'PASS', 'compile': 'PASS', 'gencode': 'PASS',
              'quantization': 'uint8 weights/activations, independent detector branches; FP32 I/O',
              'actual_SRAM_peak': 'NOT_MEASURED', 'board_latency': 'NOT_MEASURED'}
    save(OUT / 'deployment.json', report)
    sim = nncase.Simulator()
    sim.load_model(blob)
    assert sim.get_input_tensor(0).dtype == np.dtype('float32')
    for j in range(3):
        assert sim.get_output_tensor(j).dtype == np.dtype('float32')
    folder = OUT / 'sim_outputs'
    folder.mkdir(exist_ok=True)
    manifest = []
    for i in range(28):
        source = BASE / 'dev' / ('%03d.npz' % i)
        with np.load(source) as z:
            pair = z['pair']
        logits, descriptors = [], []
        for image in pair:
            sim.set_input_tensor(0, nncase.RuntimeTensor.from_numpy(np.ascontiguousarray(image[None])))
            sim.run()
            values = [sim.get_output_tensor(j).to_numpy().copy() for j in range(3)]
            for value, shape in zip(values, export['outputs']):
                assert list(value.shape) == shape and np.isfinite(value).all()
            logits.append(np.concatenate(values[:2], axis=1)[0])
            descriptors.append(values[2][0])
        target = folder / ('%03d.npz' % i)
        np.savez(target, logits=np.stack(logits), desc=np.stack(descriptors))
        manifest.append({'index': i, 'input_cache_sha256': sha(source), 'output_sha256': sha(target)})
    save(OUT / 'simulation_manifest.json', manifest)
    report.update(simulation='PASS', INT8_QUALITY='OUTPUTS_READY_FOR_EVALUATOR', rss_peak_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
    save(OUT / 'deployment.json', report)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--compile', action='store_true')
    args = parser.parse_args()
    if args.prepare:
        prepare()
        return
    if args.compile:
        compile_simulate()
        return
    OUT.mkdir(exist_ok=True)
    for python, flag, logname in [(TPY, '--prepare', 'prepare.log'), (KPY, '--compile', 'compile.log')]:
        with (OUT / logname).open('w') as log:
            subprocess.run([python, __file__, flag], stdout=log, stderr=subprocess.STDOUT, check=True)
    with (OUT / 'evaluation.log').open('w') as log:
        subprocess.run([TPY, str(OUT / 'evaluate_simulator.py')], env=dict(os.environ, PYTHONPATH=str(PREVIOUS)), stdout=log, stderr=subprocess.STDOUT, check=True)
    result = json.loads((OUT / 'int8_quality.json').read_text())
    old = json.loads((SOURCE / 'int8_quality.json').read_text())
    a, b = old['int8'], result['int8']
    passed = b['correct_mnn_count'] >= .85 * old['fp32']['correct_mnn_count'] and b['mnn_precision'] >= a['mnn_precision'] - .02 and b['repeatability_3px'] >= a['repeatability_3px'] - .02
    summary = {'status': 'COMPLETE', 'gate_pass': passed, 'parent_fp32': old['fp32'], 'parent_int8': a,
               'split_int8': b, 'retention': result['retention'], 'scope': 'fixed pilot checkpoint, actual K210 kmodel; not board/application readiness'}
    save(OUT / 'summary.json', summary)
    print(json.dumps(summary), flush=True)

if __name__ == '__main__':
    main()
