"""Rotate input and weights together to test a KPU-friendly equivalent layout."""
import os
os.environ.update(OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2')
import argparse
import json
import subprocess
from pathlib import Path
import numpy as np
import joint_feasibility as joint

ROOT = Path(__file__).parent / 'orientation_probe'
SOURCE = Path(__file__).parent / 'joint_feasibility'
TPY = '/home/quyet/k210_lab/.venv-mixvpr-cuda/bin/python'
KPY = '/home/quyet/miniconda3/envs/k210/bin/python'

def prepare():
    import onnx
    from onnx import numpy_helper
    import onnxruntime as ort
    ROOT.mkdir(exist_ok=True)
    # New channel j represents the pixel at j in the clockwise-rotated8x8 cell.
    permutation = np.r_[np.rot90(np.arange(64).reshape(8, 8), k=-1).ravel(), 64]
    inverse = np.argsort(permutation)
    calibration = np.load(SOURCE / 'r2_train.npy')
    rotated = np.ascontiguousarray(np.rot90(calibration, k=-1, axes=(2, 3)))
    assert np.array_equal(np.rot90(rotated, k=1, axes=(2, 3)), calibration)
    calfile = ROOT / 'r2_clockwise_train.npy'
    np.save(calfile, rotated)
    protocol = {'purpose': 'Remove observed early CPU fallback by a function-preserving orientation transform',
                'source_input_NCHW': [1, 1, 320, 184], 'target_input_NCHW': [1, 1, 184, 320],
                'change': 'CW90 input; CW90 all square convolution kernels; permute64 detector pixel channels identically; dustbin unchanged',
                'output_restore': 'CCW90 spatial outputs; inverse64-channel detector permutation; then original postprocessing',
                'geometry': 'x_rot=H-1-y, y_rot=x; restore x=y_rot, y=H-1-x_rot, before original letterbox inverse',
                'assumptions': 'square symmetric-padding stride1 convolutions, square pooling, both input dimensions divisible by overallstride8',
                'gate': 'all16 TRAIN input outputs match original graph after inverse transform within atol/rtol2e-4; all Conv nodes map to KPU in target compiler',
                'not_training': True, 'quality': 'NOT_EVALUATED_RANDOM_WEIGHTS', 'board_measurement': 'NOT_PERFORMED',
                'source_calibration_sha256': joint.sha(SOURCE / 'r2_train.npy'), 'target_calibration_sha256': joint.sha(calfile),
                'code_sha256': joint.sha(__file__)}
    joint.save(ROOT / 'protocol.json', protocol)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    for name in ['fr12_r2', 'wider_r2']:
        src = SOURCE / name
        dest = ROOT / (name + '_cw')
        dest.mkdir(exist_ok=True)
        graph = onnx.load(src / 'random.onnx')
        changed = []
        for i, tensor in enumerate(graph.graph.initializer):
            a = numpy_helper.to_array(tensor).copy()
            if a.ndim == 4:
                assert a.shape[2] == a.shape[3] and a.shape[2] in [1, 3]
                a = np.rot90(a, k=-1, axes=(2, 3))
            if tensor.name in ['convPb.weight', 'convPb.bias']:
                a = a[permutation]
                changed.append(tensor.name)
            graph.graph.initializer[i].CopyFrom(numpy_helper.from_array(np.ascontiguousarray(a), tensor.name))
        assert sorted(changed) == ['convPb.bias', 'convPb.weight']
        graph.graph.ClearField('value_info')
        shape = graph.graph.input[0].type.tensor_type.shape.dim
        shape[2].dim_value, shape[3].dim_value = 184, 320
        for output in graph.graph.output:
            dims = output.type.tensor_type.shape.dim
            dims[2].dim_value, dims[3].dim_value = 23, 40
        graph = onnx.shape_inference.infer_shapes(graph, strict_mode=True)
        onnx.checker.check_model(graph)
        path = dest / 'random.onnx'
        onnx.save(graph, path)
        original = ort.InferenceSession(str(src / 'random.onnx'), options, providers=['CPUExecutionProvider'])
        target = ort.InferenceSession(str(path), options, providers=['CPUExecutionProvider'])
        rows = []
        for i in range(len(calibration)):
            reference = original.run(None, {'image': calibration[i:i+1]})
            outputs = target.run(None, {'image': rotated[i:i+1]})
            restored = [np.rot90(v, k=1, axes=(2, 3)) for v in outputs]
            restored[0] = restored[0][:, inverse]
            errors = []
            for a, b in zip(reference, restored):
                error = float(np.max(np.abs(a-b)))
                assert np.allclose(a, b, atol=2e-4, rtol=2e-4), (name, i, error)
                errors.append(error)
            rows.append({'train_index': i, 'max_abs_each_head': errors})
        export = json.loads((src / 'export.json').read_text())
        export.update(input_WH=[320, 184], onnx_sha256=joint.sha(path), calibration=str(calfile), calibration_sha256=joint.sha(calfile),
                      orientation='CW90 input and kernels, detector output permutation', parent_onnx_sha256=joint.sha(src / 'random.onnx'),
                      inverse_detector_permutation=inverse.tolist(), parity=rows)
        for layer in export['layers']:
            layer['shape'][2], layer['shape'][3] = layer['shape'][3], layer['shape'][2]
        joint.save(dest / 'export.json', export)
        print('ORIENTATION_PARITY', name, max(max(r['max_abs_each_head']) for r in rows), flush=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--compile')
    args = parser.parse_args()
    if args.prepare:
        prepare()
        return
    if args.compile:
        joint.ROOT = ROOT
        joint.compile_one(args.compile)
        return
    ROOT.mkdir(exist_ok=True)
    with (ROOT / 'prepare.log').open('w') as log:
        subprocess.run([TPY, __file__, '--prepare'], stdout=log, stderr=subprocess.STDOUT, check=True)
    rows = []
    for name in ['fr12_r2_cw', 'wider_r2_cw']:
        dest = ROOT / name
        with (dest / 'compile.log').open('w') as log:
            subprocess.run([KPY, __file__, '--compile', name], stdout=log, stderr=subprocess.STDOUT, check=True)
        export = json.loads((dest / 'export.json').read_text())
        compile_report = json.loads((dest / 'compile.json').read_text())
        cpu = (dest / 'compiler_dump/stackvm/main/runtime_ops.txt').read_text()
        kpu_paths = list((dest / 'compiler_dump/k210').rglob('runtime_ops.txt'))
        kpu = '\n'.join(p.read_text() for p in kpu_paths)
        mapping = {'CPU_Conv2D_count': cpu.count('[Conv2D]'), 'KPU_Conv2D_count': kpu.count('[KPUConv2D]')}
        rows.append({'id': name, 'export': export, 'compile': compile_report, 'mapping': mapping,
                     'gate_pass': compile_report.get('gencode') == 'PASS' and mapping['CPU_Conv2D_count'] == 0 and mapping['KPU_Conv2D_count'] == 12})
        print('ORIENTATION_COMPILE', name, mapping, flush=True)
    joint.save(ROOT / 'summary.json', {'status': 'COMPLETE', 'candidates': rows,
                                      'scope': 'random graph FP32 equivalence and compiler mapping, not trained quality or board resource proof'})

if __name__ == '__main__':
    main()
