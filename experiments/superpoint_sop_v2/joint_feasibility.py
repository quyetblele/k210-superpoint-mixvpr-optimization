"""Bounded 2x2 input/width compile probe. Random weights: no quality claims."""
import os
os.environ.update(OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2')
import argparse
import json
import hashlib
import subprocess
import traceback
import resource
from pathlib import Path
import numpy as np

ROOT = Path(__file__).parent / 'joint_feasibility'
TPY = '/home/quyet/k210_lab/.venv-mixvpr-cuda/bin/python'
KPY = '/home/quyet/miniconda3/envs/k210/bin/python'
WIDTHS = {'fr12': [12, 24, 32, 64], 'wider': [24, 32, 64, 96]}
RESOLUTIONS = {'r1': [144, 256], 'r2': [184, 320]}  # W,H; camera-oriented shortlist
SEED = 20260917

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def save(path, value):
    t = path.with_suffix(path.suffix + '.tmp')
    t.write_text(json.dumps(value, indent=2) + '\n')
    t.replace(path)

def prepare():
    import torch
    from torch import nn
    import onnx
    import cv2
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    ROOT.mkdir(exist_ok=True)
    rows = json.loads(Path('/mnt/d/k210_official_ab/data.json').read_text())['rows']
    train = [r for r in rows if r['domain'] == 'project' and r['split'] == 'train']
    selected = [train[i] for i in np.linspace(0, len(train) - 1, 16, dtype=int)]
    excluded = {r['sha256'] for r in rows if r['split'] != 'train'}
    assert not {r['sha256'] for r in selected} & excluded
    protocol = {'purpose': 'Joint input/width hardware feasibility, random-weight graphs only',
                'widths': WIDTHS, 'resolutions_WH': RESOLUTIONS, 'head_hidden': 128,
                'outputs': [65, 256], 'stride': 8, 'seed': SEED,
                'calibration': '16 project TRAIN originals, gray /255, aspect-preserving letterbox',
                'quality': 'NOT_EVALUATED; neither random PTQ nor compile PASS proves quality',
                'board_peak_and_latency': 'NOT_MEASURED', 'code_sha256': sha(__file__)}
    save(ROOT / 'protocol.json', protocol)

    class Model(nn.Module):
        def __init__(self, widths):
            super().__init__()
            ci = 1
            for stage, co in enumerate(widths, 1):
                setattr(self, 'conv%da' % stage, nn.Conv2d(ci, co, 3, padding=1))
                setattr(self, 'conv%db' % stage, nn.Conv2d(co, co, 3, padding=1))
                ci = co
            self.convPa = nn.Conv2d(ci, 128, 3, padding=1)
            self.convPb = nn.Conv2d(128, 65, 1)
            self.convDa = nn.Conv2d(ci, 128, 3, padding=1)
            self.convDb = nn.Conv2d(128, 256, 1)
            self.relu = nn.ReLU()
            self.pool = nn.MaxPool2d(2)

        def forward(self, x):
            for stage in range(1, 5):
                x = self.relu(getattr(self, 'conv%da' % stage)(x))
                x = self.relu(getattr(self, 'conv%db' % stage)(x))
                if stage < 4:
                    x = self.pool(x)
            return self.convPb(self.relu(self.convPa(x))), self.convDb(self.relu(self.convDa(x)))

    for resolution, (w, h) in RESOLUTIONS.items():
        cal = []
        manifest = []
        for row in selected:
            assert sha(row['path']) == row['sha256']
            image = cv2.imread(row['path'], cv2.IMREAD_GRAYSCALE)
            ih, iw = image.shape
            scale = min(w / iw, h / ih)
            nw, nh = max(1, round(iw * scale)), max(1, round(ih * scale))
            left, top = (w - nw) // 2, (h - nh) // 2
            canvas = np.zeros((h, w), np.float32)
            canvas[top:top + nh, left:left + nw] = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA) / 255.
            cal.append(canvas[None])
            manifest.append({'path': row['path'], 'sha256': row['sha256'], 'resize_WH': [nw, nh], 'left_top': [left, top]})
        calibration = ROOT / (resolution + '_train.npy')
        np.save(calibration, np.stack(cal))
        save(ROOT / (resolution + '_calibration.json'), {'manifest': manifest, 'sha256': sha(calibration)})
        for name, widths in WIDTHS.items():
            dest = ROOT / (name + '_' + resolution)
            dest.mkdir(exist_ok=True)
            torch.manual_seed(SEED)
            model = Model(widths).eval()
            image = torch.zeros(1, 1, h, w)
            layers = []
            handles = []
            for key, layer in model.named_modules():
                if isinstance(layer, nn.Conv2d):
                    def hook(m, inputs, output, key=key):
                        layers.append({'name': key, 'shape': list(output.shape), 'uint8_logical_bytes': output.numel(),
                                       'MAC': output.numel() * m.in_channels * m.kernel_size[0] * m.kernel_size[1]})
                    handles.append(layer.register_forward_hook(hook))
            with torch.inference_mode():
                output = model(image)
            for handle in handles:
                handle.remove()
            assert [list(v.shape) for v in output] == [[1, 65, h // 8, w // 8], [1, 256, h // 8, w // 8]]
            onnxfile = dest / 'random.onnx'
            torch.onnx.export(model, image, str(onnxfile), opset_version=13, dynamo=False,
                              input_names=['image'], output_names=['detector_logits', 'descriptor_map'], do_constant_folding=False)
            graph = onnx.shape_inference.infer_shapes(onnx.load(onnxfile), strict_mode=True)
            onnx.checker.check_model(graph)
            assert {n.op_type for n in graph.graph.node} <= {'Conv', 'Relu', 'MaxPool'}
            onnx.save(graph, onnxfile)
            save(dest / 'export.json', {'status': 'PASS', 'widths': widths, 'input_WH': [w, h],
                 'parameters': sum(t.numel() for t in model.parameters()), 'conv_MAC': sum(x['MAC'] for x in layers),
                 'layers': layers, 'logical_head_fp32_bytes': sum(t.numel() * 4 for t in output),
                 'onnx_sha256': sha(onnxfile), 'calibration': str(calibration), 'calibration_sha256': sha(calibration),
                 'quality': 'NOT_EVALUATED_RANDOM_WEIGHTS'})

def compile_one(name):
    import nncase
    import _nncase
    from importlib.metadata import version
    dest = ROOT / name
    export = json.loads((dest / 'export.json').read_text())
    report = {'nncase': version('nncase'), '_nncase': _nncase.__version__, 'quality': 'NOT_EVALUATED_RANDOM_WEIGHTS',
              'actual_SRAM_peak': 'NOT_MEASURED', 'board_latency': 'NOT_MEASURED'}
    stage = 'identity'
    try:
        assert version('nncase') == '1.8.0.20220929' and _nncase.__version__ == '1.8.0-55be52f'
        assert sha(dest / 'random.onnx') == export['onnx_sha256']
        assert sha(export['calibration']) == export['calibration_sha256']
        opts = nncase.CompileOptions()
        opts.target = 'k210'
        opts.quant_type = 'uint8'
        opts.w_quant_type = 'uint8'
        opts.dump_dir = str(dest / 'compiler_dump')
        opts.dump_asm = True
        compiler = nncase.Compiler(opts)
        stage = 'import'
        compiler.import_onnx((dest / 'random.onnx').read_bytes(), nncase.ImportOptions())
        report[stage] = 'PASS'
        stage = 'ptq'
        data = np.load(export['calibration'])
        optsq = nncase.PTQTensorOptions()
        optsq.samples_count = len(data)
        optsq.set_tensor_data(data.tobytes())
        compiler.use_ptq(optsq)
        report[stage] = 'PASS'
        stage = 'compile'
        compiler.compile()
        report[stage] = 'PASS'
        stage = 'gencode'
        blob = compiler.gencode_tobytes()
        assert blob
        (dest / 'random.kmodel').write_bytes(blob)
        report.update(gencode='PASS', kmodel_bytes=len(blob), kmodel_sha256=sha(dest / 'random.kmodel'))
    except Exception as error:
        report[stage] = 'FAIL'
        report.update(reason=str(error), traceback=traceback.format_exc())
    report['rss_peak_MiB'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    save(dest / 'compile.json', report)
    print(name, json.dumps(report), flush=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--compile')
    args = parser.parse_args()
    if args.prepare:
        prepare()
    elif args.compile:
        compile_one(args.compile)
    else:
        ROOT.mkdir(exist_ok=True)
        with (ROOT / 'prepare.log').open('w') as log:
            subprocess.run([TPY, __file__, '--prepare'], stdout=log, stderr=subprocess.STDOUT, check=True)
        rows = []
        for resolution in RESOLUTIONS:
            for width in WIDTHS:
                name = width + '_' + resolution
                dest = ROOT / name
                with (dest / 'compile.log').open('w') as log:
                    process = subprocess.run([KPY, __file__, '--compile', name], stdout=log, stderr=subprocess.STDOUT, timeout=180)
                report = json.loads((dest / 'compile.json').read_text()) if (dest / 'compile.json').exists() else {'status': 'PROCESS_FAILED', 'returncode': process.returncode}
                export = json.loads((dest / 'export.json').read_text())
                rows.append({'id': name, 'export': export, 'compile': report})
                print('FEASIBILITY', name, report.get('gencode', report.get('reason', report)), flush=True)
        save(ROOT / 'summary.json', {'status': 'COMPLETE', 'candidates': rows, 'quality': 'NOT_EVALUATED',
                                     'decision': 'compile shortlist only; no final architecture selection'})

if __name__ == '__main__':
    main()
