#!/usr/bin/env python3
"""Isolated random-architecture compiler gate. No training or graph rewrites."""
import argparse
import hashlib
import json
import platform
import subprocess
import sys
import traceback
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/superpoint_arch_gate'
LOG = OUT / 'logs'
MODEL_SEED, CALIBRATION_SEED, CALIBRATION_COUNT = 20260905, 20260906, 32
KPY = '/home/quyet/miniconda3/envs/k210/bin/python'
CANDIDATES = ('v0_s2', 'v1_fr12')


def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2) + '\n')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def environment():
    from importlib.metadata import version, PackageNotFoundError
    result = {'executable': sys.executable, 'python': platform.python_version()}
    for name in ('torch', 'onnx', 'numpy', 'nncase'):
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            result[name] = 'NOT INSTALLED'
    try:
        import _nncase
        result['_nncase'] = _nncase.__version__
    except ImportError:
        result['_nncase'] = 'NOT INSTALLED'
    return result


def prepare():
    import importlib.util
    env = environment()
    save(OUT / 'compiler_environment.json', env)
    assert env['nncase'] == '1.8.0.20220929', env
    assert env['_nncase'] == '1.8.0-55be52f', env
    # Execute original Gate2 function unchanged, redirecting only its outputs.
    spec = importlib.util.spec_from_file_location('gate2', ROOT / 'scripts/compile_tiny_k210.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.KMODEL_PATH = OUT / 'gate2_reproduction/tiny.kmodel'
    module.DUMP_DIR = OUT / 'gate2_reproduction/dumps'
    module.SUMMARY_PATH = OUT / 'gate2_reproduction/summary.md'
    assert module.main() == 0, 'Gate2 reproduction failed'
    print('Gate2 reproduction PASS; original script SHA256', digest(ROOT / 'scripts/compile_tiny_k210.py'))


def export(candidate):
    import torch
    from torch import nn
    import onnx
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleDict()
            first = 16 if candidate == 'v0_s2' else 12
            def conv(name, ci, co, kernel=3, stride=1, relu=True):
                self.layers[name] = nn.Conv2d(ci, co, kernel, stride, kernel // 2, bias=True)
                if relu:
                    self.layers[name + '_relu'] = nn.ReLU(inplace=False)
            conv('conv1a', 1, first, stride=2 if candidate == 'v0_s2' else 1)
            conv('conv1b', first, first)
            if candidate == 'v1_fr12':
                self.layers['pool1'] = nn.MaxPool2d(2, 2)
            conv('conv2a', first, 24)
            conv('conv2b', 24, 24)
            self.layers['pool2'] = nn.MaxPool2d(2, 2)
            conv('conv3a', 24, 32)
            conv('conv3b', 32, 32)
            self.layers['pool3'] = nn.MaxPool2d(2, 2)
            conv('conv4a', 32, 64)
            conv('conv4b', 64, 64)
            self.encoder_names = list(self.layers.keys())
            conv('convPa', 64, 128)
            conv('detector_logits', 128, 65, kernel=1, relu=False)
            conv('convDa', 64, 128)
            conv('descriptor_map', 128, 256, kernel=1, relu=False)

        def forward(self, image):
            x = image
            for name in self.encoder_names:
                x = self.layers[name](x)
            det = self.layers['detector_logits'](self.layers['convPa_relu'](self.layers['convPa'](x)))
            desc = self.layers['descriptor_map'](self.layers['convDa_relu'](self.layers['convDa'](x)))
            return det, desc

    torch.manual_seed(MODEL_SEED)
    model = Model().eval()
    torch.manual_seed(MODEL_SEED)
    replica = Model().eval()
    assert all(torch.equal(t, replica.state_dict()[k]) for k, t in model.state_dict().items())
    stem = 'superpoint_k210_' + candidate + '_random'
    statepath = ROOT / 'models' / (stem + '.pt')
    torch.save(model.state_dict(), statepath)
    restored = torch.load(statepath, map_location='cpu', weights_only=True)
    assert all(torch.equal(t, restored[k]) for k, t in model.state_dict().items())
    statehash = hashlib.sha256()
    for k, t in model.state_dict().items():
        statehash.update(k.encode())
        statehash.update(t.numpy().tobytes())
    rows = []
    def record(name, tensor):
        assert torch.isfinite(tensor).all(), name
        rows.append({'name': name, 'shape': list(tensor.shape), 'elements': tensor.numel(),
                     'fp32_bytes': tensor.numel()*4, 'projected_int8_bytes': tensor.numel()})
    image = torch.linspace(0, 1, 240*320).reshape(1, 1, 240, 320)
    record('image', image)
    handles = [m.register_forward_hook(lambda m, i, o, n=n: record(n, o)) for n, m in model.layers.items()]
    with torch.inference_mode():
        outputs = model(image)
    for h in handles:
        h.remove()
    assert [list(t.shape) for t in outputs] == [[1,65,30,40], [1,256,30,40]]
    assert next(r['shape'] for r in rows if r['name'] == 'conv4b_relu') == [1,64,30,40]
    path = ROOT / 'models' / (stem + '.onnx')
    with torch.inference_mode():
        torch.onnx.export(model, image, str(path), opset_version=13, input_names=['image'],
                          output_names=['detector_logits','descriptor_map'], dynamo=False,
                          do_constant_folding=False)
    graph = onnx.load(str(path))
    onnx.checker.check_model(graph)
    inferred = onnx.shape_inference.infer_shapes(graph, strict_mode=True)
    onnx.checker.check_model(inferred)
    inferredpath = ROOT / 'models' / (stem + '_inferred.onnx')
    onnx.save(inferred, str(inferredpath))
    assert inferred.graph.value_info
    assert all(n.domain in ('', 'ai.onnx') for n in inferred.graph.node)
    hist = dict(Counter(n.op_type for n in inferred.graph.node))
    assert set(hist) <= {'Conv','Relu','MaxPool'}, hist
    specs = []
    for v in list(inferred.graph.input) + list(inferred.graph.value_info) + list(inferred.graph.output):
        dims = v.type.tensor_type.shape.dim
        assert all(d.HasField('dim_value') and d.dim_value > 0 and not d.dim_param for d in dims), v
        specs.append({'name':v.name,'shape':[d.dim_value for d in dims], 'dtype':v.type.tensor_type.elem_type})
    save(OUT / (candidate + '_export.json'), {'environment':environment(), 'pytorch':'PASS',
         'checker':'PASS', 'parameters':sum(p.numel() for p in model.parameters()),
         'state_reproducibility':'PASS', 'state_tensor_sha256':statehash.hexdigest(),
         'state_file_sha256':digest(statepath), 'architecture':str(model), 'activations':rows,
         'node_count':len(inferred.graph.node), 'operators':hist, 'value_info_count':len(inferred.graph.value_info),
         'onnx_specs':specs, 'onnx_sha256':digest(path), 'inferred_sha256':digest(inferredpath)})
    print(candidate, 'PyTorch/export/checker/inference PASS', hist)


def compile_model(candidate):
    import nncase
    import numpy as np
    import onnx
    result = {'checker':'NOT RUN','import':'NOT RUN','ptq':'NOT RUN','compile':'NOT RUN','gencode':'NOT RUN'}
    resultpath = OUT / (candidate + '_compile.json')
    stage = 'checker'
    try:
        path = ROOT / 'models' / ('superpoint_k210_' + candidate + '_random_inferred.onnx')
        graph = onnx.load(str(path))
        onnx.checker.check_model(graph)
        assert graph.graph.value_info
        result[stage] = 'PASS'
        stage = 'import'
        options = nncase.CompileOptions()
        options.target = 'k210'
        options.quant_type = 'uint8'
        options.w_quant_type = 'uint8'
        dump = OUT / (candidate + '_nncase')
        dump.mkdir(parents=True, exist_ok=True)
        options.dump_dir = str(dump)
        options.dump_ir = True
        options.dump_asm = True
        options.dump_quant_error = True
        compiler = nncase.Compiler(options)
        compiler.import_onnx(path.read_bytes(), nncase.ImportOptions())
        result[stage] = 'PASS'
        save(resultpath,result)
        print('Import PASS', flush=True)
        stage = 'ptq'
        data = np.random.default_rng(CALIBRATION_SEED).uniform(0,1,size=(CALIBRATION_COUNT,1,240,320)).astype(np.float32)
        result['calibration_sha256'] = hashlib.sha256(data.tobytes(order='C')).hexdigest()
        ptq = nncase.PTQTensorOptions()
        ptq.samples_count = CALIBRATION_COUNT
        ptq.set_tensor_data(data.tobytes(order='C'))
        compiler.use_ptq(ptq)
        result[stage] = 'PASS'
        save(resultpath,result)
        print('PTQ configuration PASS (calibration lowering executes in compile)', flush=True)
        stage = 'compile'
        compiler.compile()
        result[stage] = 'PASS'
        save(resultpath,result)
        stage = 'gencode'
        blob = compiler.gencode_tobytes()
        assert blob, 'Empty kmodel'
        artifact = ROOT / 'artifacts' / ('superpoint_k210_' + candidate + '_random.kmodel')
        artifact.write_bytes(blob)
        result.update(gencode='PASS', size=len(blob), sha256=hashlib.sha256(blob).hexdigest())
        print('Compile/gencode PASS', len(blob), result['sha256'], flush=True)
    except Exception:
        result[stage] = 'FAIL'
        result['error'] = traceback.format_exc()
        traceback.print_exc()
    save(resultpath,result)
    return 0 if result['gencode'] == 'PASS' else 1


def run_logged(args, name):
    with (LOG / name).open('w') as log:
        return subprocess.run(args, cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT).returncode


def report():
    lines = ['# SuperPoint architecture compiler feasibility gate', '',
             'Compiler-only evaluation with random weights. No training, fine-tuning, distillation, pruning, QAT, quality or hardware benchmark.',
             'PASS does not prove board execution, latency, camera integration, localization quality or full-system RAM fit.', '',
             '## Reproducibility and toolchain', '',
             'MODEL_SEED=20260905; CALIBRATION_SEED=20260906; CALIBRATION_COUNT=32.',
             'Calibration: synthetic FP32 [32,1,240,320] in [0,1], identical for both candidates.',
             'All convolutions bias=True. Static ONNX opset 13; no graph rewrites.',
             'Gate2 original API reproduced with output destinations isolated; see logs/preparation.log.',
             'nncase target=k210; activation/weight quantization uint8, exactly as Gate2. Projected INT8 logical bytes also equal uint8 logical bytes.',
             'PTQ PASS means use_ptq accepted the configuration; actual calibration/lowering is performed by compile().', '',
             'Compiler environment:', '```json', (OUT/'compiler_environment.json').read_text().strip(), '```']
    results = []
    for c in CANDIDATES:
        epath, cpath = OUT/(c+'_export.json'), OUT/(c+'_compile.json')
        e = json.loads(epath.read_text()) if epath.exists() else {}
        r = json.loads(cpath.read_text()) if cpath.exists() else {'import':'NOT RUN','ptq':'NOT RUN','compile':'NOT RUN','gencode':'NOT RUN'}
        results.append((e,r))
    lines += ['', '## Comparison', '', '| Metric | V0-S2 | V1-FR12 |','|---|---|---|']
    def row(n,a,b): lines.append('| {} | {} | {} |'.format(n,a,b))
    row('Early downsampling','conv1a stride 2','two full-resolution convs, then pool')
    for key,label in [('parameters','Parameters'),('checker','ONNX checker'),('node_count','ONNX node count'),('operators','Operator histogram')]:
        row(label, *[str(e.get(key,'NOT RUN')) for e,r in results])
    for key,label in [('fp32_bytes','Largest logical FP32 activation bytes'),('projected_int8_bytes','Largest projected INT8 activation bytes')]:
        row(label,*[max((a[key] for a in e.get('activations',[])),default=0) for e,r in results])
    for key in ('import','ptq','compile','gencode','size'):
        row(key,*[r.get(key,'NOT RUN') for e,r in results])
    row('Compiler memory information',*[('See verbatim kmodel_info below' if (OUT/(c+'_nncase/kmodel_info.txt')).exists() else 'NOT PROVIDED') for c in CANDIDATES])
    row('Warnings/errors', *[('See exact log excerpts below; '+r.get('error','no caught exception').replace('\n',' ')) for e,r in results])
    for c,(e,r) in zip(CANDIDATES,results):
        lines += ['', '## '+c, '', 'Export environment: `'+json.dumps(e.get('environment',{}))+'`',
                  '', '### Exact architecture', '', '```', e.get('architecture','NOT RUN'), '```',
                  '', '### Tensor logical sizes in execution order', '',
                  'These are individual tensor sizes, NOT peak KPU RAM. No sum is used as a compiler peak estimate.', '',
                  '| Name | Shape | Elements | FP32 bytes | Projected INT8 bytes |', '|---|---|---:|---:|---:|']
        for a in e.get('activations',[]):
            lines.append('| {name} | {shape} | {elements} | {fp32_bytes} | {projected_int8_bytes} |'.format(**a))
        lines += ['', 'Largest tensors, descending logical bytes:', '']
        for a in sorted(e.get('activations',[]), key=lambda a:a['fp32_bytes'],reverse=True)[:8]:
            lines.append('- {} {}: FP32 {} bytes; projected INT8 {} bytes'.format(a['name'],a['shape'],a['fp32_bytes'],a['projected_int8_bytes']))
        lines += ['', '### ONNX metadata and reproducible state', '',
                  '- All input/intermediate/output dimensions static; custom domains absent; ops restricted to Conv/Relu/MaxPool.',
                  '- Serialized value_info entries: '+str(e.get('value_info_count','NOT RUN')),
                  '- State reconstruction/save-load equality: '+e.get('state_reproducibility','NOT RUN'),
                  '- Canonical state tensor SHA256: '+e.get('state_tensor_sha256','NOT RUN'),
                  '- State file SHA256: '+e.get('state_file_sha256','NOT RUN'),
                  '- Input: image FP32 [1,1,240,320]. Shared encoder: [1,64,30,40].',
                  '- Outputs: detector_logits FP32 [1,65,30,40]; descriptor_map FP32 [1,256,30,40].',
                  '- Full ONNX tensor specs and ONNX hashes: '+c+'_export.json',
                  '', '### Compiler result', '', '```json',json.dumps(r,indent=2),'```']
        info = OUT/(c+'_nncase/kmodel_info.txt')
        lines += ['', '### Compiler memory / model information', '',
                  'Verbatim compiler output (section allocations are not full-system or measured board RAM):' if info.exists() else 'NOT PROVIDED',
                  '```',info.read_text() if info.exists() else 'NOT PROVIDED','```']
        lines += ['', '### Compiler execution mapping', '']
        for ops in sorted((OUT/(c+'_nncase')).rglob('runtime_ops.txt')):
            lines += [str(ops.relative_to(OUT)), '```', ops.read_text(), '```']
        lines += ['Compiler mapping is evidence of planned KPU/CPU operations, not a hardware measurement.',
                  'Compiled external input/output remain FP32. The two outputs occupy 1,540,800 logical bytes together.',
                  'A dedicated peak KPU SRAM measurement is NOT PROVIDED by the quoted summary.']
        raw = (LOG/(c+'_compile.log')).read_text() if (LOG/(c+'_compile.log')).exists() else ''
        warnings = [l for l in raw.splitlines() if any(w in l.lower() for w in ('warning','cannot find','error','exception','failed'))]
        lines += ['', '### Warnings/errors', '', '```', '\n'.join(warnings) or 'None detected in compiler log.', '```',
                  'Full raw stdout/stderr/traceback: logs/'+c+'_compile.log',
                  'Exporter warnings and stdout/stderr: logs/'+c+'_export.log']
    passed = [bool(e) and e.get('checker') == 'PASS' and all(r.get(k)=='PASS' for k in ('checker','import','ptq','compile','gencode')) for e,r in results]
    decision = ('Recommend V1-FR12 for the next quality gate; retain V0-S2 as the memory-first fallback. Quality advantage remains untested.' if all(passed) else
                'Recommend '+('V0-S2' if passed[0] else 'V1-FR12')+' for the next gate.' if any(passed) else 'Both failed. STOP; no V2 or architecture rescue.')
    lines += ['', '## Recommendation', '', decision, '', '## Created/modified files', '',
              'Created scripts/superpoint_arch_gate.py, both random state_dict .pt files, four ONNX files, successful kmodels, and this report directory (JSON, logs, isolated Gate2 reproduction and compiler dumps). Exact generated inventory: files_created.txt.',
              'Existing Gate2/source scripts and dependency versions were not modified. No commit.', '',
              'Gate2 reproduction validates successful stages, not byte-identical compiler output: reproduced tiny.kmodel SHA256 '+digest(OUT/'gate2_reproduction/tiny.kmodel')+'. The historical Gate2 hash differs; no deterministic nncase binary guarantee is claimed. State tensors and calibration inputs are separately hashed.', '',
              'SUPERPOINT ARCH GATE: V0 {} / V1 {}'.format(*['PASS' if p else 'FAIL' for p in passed])]
    (OUT/'comparison.md').write_text('\n'.join(lines)+'\n')
    created = [ROOT/'scripts/superpoint_arch_gate.py'] + list(OUT.rglob('*'))
    for c in CANDIDATES:
        created.extend((ROOT/'models').glob('superpoint_k210_'+c+'_random*'))
        created.extend((ROOT/'artifacts').glob('superpoint_k210_'+c+'_random*'))
    inventory = OUT/'files_created.txt'
    created.append(inventory)
    inventory.write_text('\n'.join(sorted(set(str(p.relative_to(ROOT)) for p in created if p.is_file() or p == inventory)))+'\n')
    print(lines[-1])


def main():
    p=argparse.ArgumentParser()
    p.add_argument('action', choices=['run','prepare','export','compile','report'])
    p.add_argument('--candidate',choices=CANDIDATES)
    args=p.parse_args()
    LOG.mkdir(parents=True,exist_ok=True)
    if args.action=='prepare': prepare()
    elif args.action=='export': export(args.candidate)
    elif args.action=='compile': return compile_model(args.candidate)
    elif args.action=='report': report()
    else:
        script=str(Path(__file__).resolve())
        if run_logged([KPY,'-u',script,'prepare'],'preparation.log'):
            (OUT/'comparison.md').write_text('PREPARATION BLOCKED\n\nSee logs/preparation.log. Candidates NOT RUN. No dependencies changed.\n')
            print('PREPARATION BLOCKED'); return 1
        print('Preparation PASS',flush=True)
        for c in CANDIDATES:
            print('Exporting '+c,flush=True)
            if run_logged([sys.executable,'-u',script,'export','--candidate',c],c+'_export.log') == 0:
                print('Compiling '+c,flush=True)
                rc=run_logged([KPY,'-u',script,'compile','--candidate',c],c+'_compile.log')
                print(c+' compiler exit '+str(rc),flush=True)
        report()
    return 0


if __name__=='__main__':
    sys.exit(main())
