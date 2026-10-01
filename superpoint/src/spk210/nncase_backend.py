"""Pinned nncase v1 compile and faithful simulator; imports no PyTorch."""
import json, time
from pathlib import Path
from importlib.metadata import version
import numpy as np
from .settings import ROOT, SUPERPOINT
from .io import sha, put, guard

def context(candidate):
    dest = SUPERPOINT / 'artifacts/deployment' / candidate
    e = json.loads((dest / 'export.json').read_text())
    assert sha(Path(e['checkpoint'])) == e['checkpoint_sha256']
    assert sha(dest / 'canonical.onnx') == e['canonical_onnx_sha256']
    return (dest, e)

def compile_model(candidate):
    import nncase, _nncase, onnx
    guard()
    (dest, e) = context(candidate)
    parity = json.loads((dest / 'onnx_parity.json').read_text())
    cal = json.loads((dest / 'calibration.json').read_text())
    assert parity['status'] == 'PASS' and parity['onnx_sha256'] == e['canonical_onnx_sha256']
    assert cal['split'] == 'TRAIN_ONLY' and cal['checkpoint_sha256'] == e['checkpoint_sha256']
    assert sha(dest / 'calibration_train.npy') == cal['tensor_sha256']
    assert version('nncase') == '1.8.0.20220929' and _nncase.__version__ == '1.8.0-55be52f'
    options = nncase.CompileOptions()
    options.target = 'k210'
    options.quant_type = 'uint8'
    options.w_quant_type = 'uint8'
    options.dump_dir = str(dest / 'compiler_dump')
    options.dump_asm = True
    compiler = nncase.Compiler(options)
    compiler.import_onnx((dest / 'canonical.onnx').read_bytes(), nncase.ImportOptions())
    data = np.load(dest / 'calibration_train.npy')
    q = nncase.PTQTensorOptions()
    q.samples_count = len(data)
    q.set_tensor_data(data.tobytes())
    compiler.use_ptq(q)
    compiler.compile()
    blob = compiler.gencode_tobytes()
    assert blob
    (dest / 'model.kmodel').write_bytes(blob)
    cpu = (dest / 'compiler_dump/stackvm/main/runtime_ops.txt').read_text()
    kpu = '\n'.join((f.read_text() for f in (dest / 'compiler_dump/k210').rglob('runtime_ops.txt')))
    put(dest / 'compile.json', {'PTQ': 'PASS', 'compile': 'PASS', 'gencode': 'PASS', 'nncase': version('nncase'), 'runtime_version': _nncase.__version__, 'quantization': 'uint8 weights/activations,8-bit PTQ; FP32 external wrappers', 'checkpoint_sha256': e['checkpoint_sha256'], 'onnx_sha256': e['canonical_onnx_sha256'], 'calibration_sha256': cal['tensor_sha256'], 'kmodel_sha256': sha(dest / 'model.kmodel'), 'kmodel_bytes': len(blob), 'CPU_Conv2D_count': cpu.count('[Conv2D]'), 'KPU_Conv2D_count': kpu.count('[KPUConv2D]'), 'board_latency': 'NOT_MEASURED', 'actual_peak_SRAM': 'NOT_MEASURED', 'note': 'Compiler memory planning totals are not actual board peak SRAM'})

def simulate(candidate):
    import nncase
    guard()
    (dest, e) = context(candidate)
    c = json.loads((dest / 'compile.json').read_text())
    assert sha(dest / 'model.kmodel') == c['kmodel_sha256']
    report = {'backend': 'actual nncase K210 Simulator', 'kmodel_sha256': c['kmodel_sha256'], 'checkpoint_sha256': e['checkpoint_sha256'], 'INT8_QUALITY': 'BLOCKED'}
    try:
        sim = nncase.Simulator()
        sim.load_model((dest / 'model.kmodel').read_bytes())
        assert sim.get_input_tensor(0).dtype == np.dtype('float32'), 'Unknown input quantization mapping'
        assert all((sim.get_output_tensor(i).dtype == np.dtype('float32') for i in range(2))), 'Unknown output quantization mapping'
        res = e['logical_resolution']
        folder = dest / 'sim_outputs'
        folder.mkdir(exist_ok=True)
        records = []
        p = json.loads((ROOT / 'protocol.json').read_text())
        perm = np.r_[np.rot90(np.arange(64).reshape(8, 8), -1).ravel(), 64]
        for i in range(len(p['rows']['dev'])):
            source = ROOT / res / 'dev' / ('%03d.npz' % i)
            with np.load(source) as z:
                pair = z['pair'].copy()
            values = [[], []]
            for image in pair:
                x = image[None]
                x = np.ascontiguousarray(np.rot90(x, -1, (2, 3)) if res == 'r2' else x)
                sim.set_input_tensor(0, nncase.RuntimeTensor.from_numpy(x))
                sim.run()
                out = [sim.get_output_tensor(j).to_numpy().copy() for j in range(2)]
                for (j, value) in enumerate(out):
                    assert list(value.shape) == e['outputs'][j] and np.isfinite(value).all()
                if res == 'r2':
                    out = [np.rot90(v, 1, (2, 3)) for v in out]
                    out[0] = out[0][:, np.argsort(perm)]
                for (j, v) in enumerate(out):
                    values[j].append(v[0])
            output = folder / ('%03d.npz' % i)
            np.savez_compressed(output, logits=np.stack(values[0]), desc=np.stack(values[1]))
            records.append({'index': i, 'input_sha256': sha(source), 'output_sha256': sha(output)})
        report.update(status='PASS', INT8_QUALITY='OUTPUTS_READY_FOR_EVALUATION', rows=records)
    except Exception as ex:
        report.update(status='BLOCKED', reason=str(ex))
        put(dest / 'simulation.json', report)
        raise
    put(dest / 'simulation.json', report)
