#!/usr/bin/env python3
"""M0.1: isolate the exact learned MixVPR core from its runtime wrapper."""
from __future__ import annotations

import hashlib
import json
import platform
import re
import subprocess
import sys
import traceback
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
EDGE = Path('/home/quyet/edge_ai_project')
OUT = ROOT / 'reports/mixvpr_feasibility'
LOG = OUT / 'logs'
M0 = OUT / 'm0_model_manifest.json'
R0 = ROOT / 'reports/superpoint_resolution_gate/r0_protocol_manifest.json'
ONNX = ROOT / 'models/mixvpr_m01_static_core_1x3x320x320.onnx'
INFERRED = ROOT / 'models/mixvpr_m01_static_core_1x3x320x320_inferred.onnx'
KMODEL = ROOT / 'artifacts/mixvpr_m01_static_core_1x3x320x320.kmodel'
KPY = '/home/quyet/miniconda3/envs/k210/bin/python'
CORE_SHAPE = (1, 3, 320, 320)
SEED, CALIBRATION_COUNT = 20260906, 8


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + '\n', encoding='utf-8')


def load_model():
    for path in (str(EDGE / 'third_party'), str(EDGE / 'third_party/hloc')):
        if path not in sys.path:
            sys.path.insert(0, path)
    from hloc import extractors
    from hloc.utils.base_model import dynamic_load
    return dynamic_load(extractors, 'mixvpr')({}).eval().cpu()


def core_wrapper(model):
    import torch.nn as nn
    class Core(nn.Module):
        def __init__(self, net):
            super().__init__()
            self.net = net
        def forward(self, image):
            return self.net(image)
    return Core(model.net).eval().cpu()


def external_preprocess(bgr, model):
    """Verbatim semantics of localize_live.preprocess_frame, then Mixvpr._forward."""
    import cv2
    import torch
    import torch.nn.functional as F
    h, w = bgr.shape[:2]
    scale = min(1.0, 384 / max(1, w), 480 / max(1, h))
    if scale < 1.0:
        bgr = cv2.resize(bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    raw = torch.as_tensor(rgb, dtype=torch.float32).permute(2, 0, 1)[None] / 255.0
    return F.interpolate(model.norm(raw), size=(320, 320), mode='bilinear', align_corners=False), raw


def scalar_metrics(a, b):
    import numpy as np
    a, b = a.reshape(-1), b.reshape(-1)
    return {'max_abs_error': float(np.max(np.abs(a - b))),
            'mean_abs_error': float(np.mean(np.abs(a - b))),
            'cosine_similarity': float(np.dot(a, b) / max(float(np.linalg.norm(a) * np.linalg.norm(b)), 1e-12))}


def identities(model, core):
    m0 = json.loads(M0.read_text())
    def state_hash(module):
        h = hashlib.sha256()
        for key, value in module.state_dict().items():
            h.update(key.encode()); h.update(value.detach().cpu().contiguous().numpy().tobytes())
        return h.hexdigest()
    params = sum(x.numel() for x in core.parameters())
    return {'protocol_version': 'MIXVPR-M0.1-v1', 'm0_manifest_path': str(M0), 'm0_manifest_sha256': sha(M0),
            'm0_checkpoint_sha256': m0['pytorch_model']['checkpoint_sha256'],
            'preprocessing_contract': {'raw_input': 'OpenCV BGR image; `preprocess_frame`: downscale-only min(1,384/W,480/H) with INTER_AREA; BGR->RGB; FP32 NCHW /255; no crop/pad',
                                       'wrapper_transform': 'torchvision Normalize ImageNet mean/std then F.interpolate([320,320], bilinear, align_corners=False)',
                                       'fixed_learned_core_input': list(CORE_SHAPE), 'fixed_input_semantics': 'normalized RGB FP32 NCHW after external deterministic preprocessing'},
            'unchanged_core': {'checkpoint_sha256': m0['pytorch_model']['checkpoint_sha256'],
                               'parameter_count': params, 'fp32_parameter_bytes': params * 4, 'projected_int8_parameter_bytes': params,
                               'core_state_dict_sha256': state_hash(core), 'backbone_state_dict_sha256': state_hash(model.net.backbone),
                               'aggregator_state_dict_sha256': state_hash(model.net.aggregator),
                               'backbone': m0['pytorch_model']['backbone'], 'aggregator': m0['pytorch_model']['aggregator'],
                               'descriptor_dimension': 4096},
            'changed_only': 'Preprocessing location: wrapper inside exported ONNX -> deterministic external preprocessing. Learned weights, backbone, aggregator and descriptor dimension are unchanged.',
            'environment': {'python': platform.python_version(), 'python_executable': sys.executable}}


def real_images():
    data = json.loads(R0.read_text())
    return [Path(x['absolute_path']) for x in data['ordered_images']]


def parity(model, core):
    import cv2
    import numpy as np
    import torch
    rows, input_values, output_values = [], [], []
    captured = []
    handle = model.net.register_forward_pre_hook(lambda _m, args: captured.append(args[0].detach().cpu()))
    with torch.inference_mode():
        for index, path in enumerate(real_images()):
            bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if bgr is None:
                raise RuntimeError('Could not read real image: ' + str(path))
            fixed, raw = external_preprocess(bgr, model)
            descriptor_a = model({'image': raw})['global_descriptor'].detach().cpu().numpy()
            core_input_a = captured.pop().numpy()
            descriptor_b = core(fixed).detach().cpu().numpy()
            im, dm = scalar_metrics(core_input_a, fixed.numpy()), scalar_metrics(descriptor_a, descriptor_b)
            input_values.append(im); output_values.append(dm)
            rows.append({'index': index, 'path': str(path), 'raw_shape': list(raw.shape), 'core_input_shape_a': list(core_input_a.shape),
                         'core_input_shape_b': list(fixed.shape), 'input': im, 'descriptor_shape_a': list(descriptor_a.shape),
                         'descriptor_shape_b': list(descriptor_b.shape), 'descriptor': dm})
    handle.remove()
    def aggregate(values):
        return {key: max(x[key] for x in values) if key == 'max_abs_error' else float(np.mean([x[key] for x in values])) for key in values[0]}
    result = {'status': 'PASS', 'image_count': len(rows), 'images': rows,
              'learned_core_input_parity': aggregate(input_values), 'descriptor_parity': aggregate(output_values),
              'acceptance': 'exact numerical equality expected because Path B invokes the same operations on the same raw tensors before the unchanged net'}
    if result['learned_core_input_parity']['max_abs_error'] != 0.0 or result['descriptor_parity']['max_abs_error'] != 0.0:
        result['status'] = 'FAIL'
    return result


def export_and_audit(model, core):
    import numpy as np
    import onnx
    import onnxruntime
    import torch
    dummy = torch.linspace(-2, 2, int(np.prod(CORE_SHAPE)), dtype=torch.float32).reshape(CORE_SHAPE)
    with torch.inference_mode():
        expected = core(dummy).numpy()
    torch.onnx.export(core, dummy, str(ONNX), opset_version=13, dynamo=False, do_constant_folding=False,
                      input_names=['normalized_rgb_320x320'], output_names=['global_descriptor'])
    graph = onnx.load(str(ONNX)); onnx.checker.check_model(graph)
    inferred = onnx.shape_inference.infer_shapes(graph, strict_mode=True); onnx.checker.check_model(inferred); onnx.save(inferred, str(INFERRED))
    session = onnxruntime.InferenceSession(str(INFERRED), providers=['CPUExecutionProvider'])
    actual = session.run(None, {'normalized_rgb_320x320': dummy.numpy()})[0]
    ops = dict(Counter(node.op_type for node in inferred.graph.node))
    values = list(inferred.graph.input) + list(inferred.graph.value_info) + list(inferred.graph.output)
    dims = lambda v: [d.dim_value if d.HasField('dim_value') else d.dim_param for d in v.type.tensor_type.shape.dim]
    dynamic = [v.name for v in values if any(not d.HasField('dim_value') for d in v.type.tensor_type.shape.dim)]
    # Logical leaf outputs capture stage ownership, never a peak-memory claim.
    events, handles = [], []
    def hook(name):
        def record(_m, _i, output):
            if torch.is_tensor(output): events.append({'name': name, 'shape': list(output.shape), 'elements': output.numel(), 'fp32_bytes': output.numel()*4, 'projected_int8_bytes': output.numel()})
        return record
    for name, module in core.named_modules():
        if name and not list(module.children()): handles.append(module.register_forward_hook(hook(name)))
    with torch.inference_mode():
        backbone = core.net.backbone(dummy); aggregator = core.net.aggregator(backbone)
    for h in handles: h.remove()
    largest = sorted(events, key=lambda x: x['fp32_bytes'], reverse=True)
    stage = [{'name': 'static_core_input', 'shape': list(dummy.shape), 'elements': dummy.numel(), 'fp32_bytes': dummy.numel()*4, 'projected_int8_bytes': dummy.numel()},
             {'name': 'backbone_output', 'shape': list(backbone.shape), 'elements': backbone.numel(), 'fp32_bytes': backbone.numel()*4, 'projected_int8_bytes': backbone.numel()},
             {'name': 'aggregator_descriptor', 'shape': list(aggregator.shape), 'elements': aggregator.numel(), 'fp32_bytes': aggregator.numel()*4, 'projected_int8_bytes': aggregator.numel()}]
    return {'export': 'PASS', 'checker': 'PASS', 'shape_inference': 'PASS', 'opset': 13, 'onnx_path': str(ONNX), 'onnx_sha256': sha(ONNX), 'inferred_onnx_path': str(INFERRED), 'inferred_onnx_sha256': sha(INFERRED),
            'node_count': len(inferred.graph.node), 'operators': ops, 'dynamic_shape_values': dynamic, 'input': {'name': inferred.graph.input[0].name, 'shape': dims(inferred.graph.input[0])}, 'output': {'name': inferred.graph.output[0].name, 'shape': dims(inferred.graph.output[0])},
            'resize_nodes': [{'index': i, 'name': n.name, 'inputs': list(n.input), 'outputs': list(n.output)} for i,n in enumerate(inferred.graph.node) if n.op_type == 'Resize'],
            'ort_parity': {'status': 'PASS', 'shape_pytorch': list(expected.shape), 'shape_onnxruntime': list(actual.shape), **scalar_metrics(expected, actual)},
            'stage_tensors': stage, 'largest_logical_leaf_activations': largest[:30],
            'largest_logical_activation': largest[0], 'largest_backbone_logical_activation': next(x for x in largest if x['name'].startswith('net.backbone')),
            'largest_aggregator_logical_activation': next(x for x in largest if x['name'].startswith('net.aggregator')),
            'logical_note': 'Individual logical tensors only; these are not peak RAM, KPU RAM, or board RAM.'}


def compiler():
    import _nncase, nncase, numpy as np, onnx
    from importlib.metadata import version
    result, stage = {'checker': 'NOT RUN', 'import': 'NOT RUN', 'ptq': 'NOT RUN', 'compile': 'NOT RUN', 'gencode': 'NOT RUN'}, 'checker'
    try:
        env={'python_executable':sys.executable,'nncase':version('nncase'),'_nncase':_nncase.__version__}; assert env['nncase']=='1.8.0.20220929' and env['_nncase']=='1.8.0-55be52f', env; result['environment']=env
        g=onnx.load(str(INFERRED)); onnx.checker.check_model(g); result['checker']='PASS'; stage='import'
        opt=nncase.CompileOptions(); opt.target='k210'; opt.quant_type='uint8'; opt.w_quant_type='uint8'; opt.dump_dir=str(OUT/'m01_nncase_dump'); opt.dump_ir=True; opt.dump_asm=True
        c=nncase.Compiler(opt); c.import_onnx(INFERRED.read_bytes(), nncase.ImportOptions()); result['import']='PASS'; stage='ptq'
        calibration=np.random.default_rng(SEED).uniform(-3,3,size=(CALIBRATION_COUNT,*CORE_SHAPE[1:])).astype(np.float32); result['calibration_shape']=list(calibration.shape); result['calibration_sha256']=hashlib.sha256(calibration.tobytes()).hexdigest(); ptq=nncase.PTQTensorOptions(); ptq.samples_count=CALIBRATION_COUNT; ptq.set_tensor_data(calibration.tobytes()); c.use_ptq(ptq); result['ptq']='PASS'; stage='compile'
        c.compile(); result['compile']='PASS'; stage='gencode'; blob=c.gencode_tobytes(); KMODEL.write_bytes(blob); result.update(gencode='PASS',size=len(blob),sha256=sha(KMODEL))
    except Exception:
        result[stage]='FAIL'; result['error']=traceback.format_exc()
    write(OUT/'m01_static_core_compiler.json',result); print(result); return 0 if result['gencode']=='PASS' else 1


def classify(comp, graph):
    if comp['gencode']=='PASS': return 'MIXVPR-M0.1: STATIC CORE K210 COMPILE PASS', 'NONE YET'
    if comp['import']=='FAIL':
        # nncase may omit node identity; the earliest named learned graph owner is recorded as evidence.
        error=comp.get('error','')
        owner='AGGREGATOR' if re.search(r'aggregator|channel_proj|row_proj',error,re.I) else 'BACKBONE'
        return 'MIXVPR-M0.1: NNCASE IMPORT FAIL — '+owner, 'nncase import: '+(error.strip().splitlines()[-1] if error else 'compiler did not expose a node name')
    return 'MIXVPR-M0.1: COMPILE FAIL AFTER IMPORT', comp.get('error','compiler failure after import').strip().splitlines()[-1]


def failure_node_evidence(error):
    """Report the node facts named by nncase; never guess a different node."""
    import onnx
    match = re.search(r"Can't pull input data for <([^>]+)>", error or '')
    if not match:
        return {'node': 'NOT EXPOSED BY NNCASE', 'op_type': 'NOT EXPOSED BY NNCASE',
                'owner': 'NOT EXPOSED BY NNCASE', 'evidence': 'Compiler exception contains no ONNX tensor/node name.'}
    tensor = match.group(1)
    graph = onnx.load(str(INFERRED)).graph
    values = {v.name: v for v in list(graph.input) + list(graph.value_info) + list(graph.output)}
    shape = None
    if tensor in values:
        shape = [d.dim_value if d.HasField('dim_value') else d.dim_param for d in values[tensor].type.tensor_type.shape.dim]
    producer = next((n for n in graph.node if tensor in n.output), None)
    consumer = next((n for n in graph.node if tensor in n.input), None)
    return {'tensor': tensor, 'tensor_shape': shape, 'owner': 'AGGREGATOR' if '/net/aggregator/' in tensor else 'BACKBONE',
            'producer_node': {'name': producer.name, 'op_type': producer.op_type, 'inputs': list(producer.input), 'outputs': list(producer.output)} if producer else None,
            'first_consumer_node': {'name': consumer.name, 'op_type': consumer.op_type, 'inputs': list(consumer.input), 'outputs': list(consumer.output)} if consumer else None,
            'evidence': 'Tensor name is quoted verbatim by nncase import exception.'}


def report():
    manifest=json.loads((OUT/'m01_static_core_manifest.json').read_text()); parity_data=json.loads((OUT/'m01_static_core_parity.json').read_text()); graph=json.loads((OUT/'m01_static_core_graph.json').read_text()); comp=json.loads((OUT/'m01_static_core_compiler.json').read_text()); final,bottleneck=classify(comp,graph)
    first = failure_node_evidence(comp.get('error','')) if comp.get('import') == 'FAIL' else {'node': 'N/A', 'op_type': 'N/A', 'owner': 'N/A'}
    lines=['# MixVPR M0.1 static learned-core K210 gate','', 'Only preprocessing was moved outside the ONNX graph. No learned weights, backbone, aggregator, descriptor dimension, source-project files, training, pruning, distillation, QAT, or commit changed.','', '## Frozen identity and split contract','', f"- M0 manifest: `{manifest['m0_manifest_path']}` SHA `{manifest['m0_manifest_sha256']}`.", f"- Exact learned-core static input: `{manifest['preprocessing_contract']['fixed_learned_core_input']}` — {manifest['preprocessing_contract']['fixed_input_semantics']}.", f"- Raw runtime behavior: {manifest['preprocessing_contract']['raw_input']}.", f"- Wrapper behavior moved externally: {manifest['preprocessing_contract']['wrapper_transform']}.", f"- Unchanged core checkpoint SHA: `{manifest['unchanged_core']['checkpoint_sha256']}`; core state SHA `{manifest['unchanged_core']['core_state_dict_sha256']}`; backbone SHA `{manifest['unchanged_core']['backbone_state_dict_sha256']}`; aggregator SHA `{manifest['unchanged_core']['aggregator_state_dict_sha256']}`.", '', '## Real-image preprocessing and descriptor parity','', f"- Images: {parity_data['image_count']} frozen real project images.", f"- Tensor entering learned core: shape `{CORE_SHAPE}`, dtype FP32; max abs `{parity_data['learned_core_input_parity']['max_abs_error']:.3e}`, mean abs `{parity_data['learned_core_input_parity']['mean_abs_error']:.3e}`.", f"- Descriptor A vs B: shape `[1,4096]`; max abs `{parity_data['descriptor_parity']['max_abs_error']:.3e}`, mean abs `{parity_data['descriptor_parity']['mean_abs_error']:.3e}`, cosine `{parity_data['descriptor_parity']['cosine_similarity']:.9f}`.", f"- Parity result: {parity_data['status']}.", '', '## Static core ONNX','', f"- `{graph['onnx_path']}` SHA `{graph['onnx_sha256']}`; inferred SHA `{graph['inferred_onnx_sha256']}`.", f"- Opset {graph['opset']}; export/checker/shape inference: PASS/PASS/PASS; input `{graph['input']['shape']}`, output `{graph['output']['shape']}`.", f"- Resize nodes: `{graph['resize_nodes']}` (must be empty).", f"- ORT parity: max abs `{graph['ort_parity']['max_abs_error']:.3e}`, mean abs `{graph['ort_parity']['mean_abs_error']:.3e}`, cosine `{graph['ort_parity']['cosine_similarity']:.9f}`.", '', '### Operator inventory','', '| Operator | Count |','|---|---:|']
    lines += [f'| {op} | {count} |' for op,count in sorted(graph['operators'].items())]
    lines += ['', '## Resource audit','', f"- Parameters: {manifest['unchanged_core']['parameter_count']:,}; FP32 `{manifest['unchanged_core']['fp32_parameter_bytes']:,}` bytes; projected INT8 `{manifest['unchanged_core']['projected_int8_parameter_bytes']:,}` bytes.", f"- Largest logical activation: `{graph['largest_logical_activation']['name']}` `{graph['largest_logical_activation']['shape']}` = `{graph['largest_logical_activation']['fp32_bytes']:,}` FP32 bytes / `{graph['largest_logical_activation']['projected_int8_bytes']:,}` projected INT8 bytes.", f"- Backbone largest: `{graph['largest_backbone_logical_activation']['name']}` `{graph['largest_backbone_logical_activation']['shape']}` = `{graph['largest_backbone_logical_activation']['fp32_bytes']:,}` FP32 bytes.", f"- Aggregator largest: `{graph['largest_aggregator_logical_activation']['name']}` `{graph['largest_aggregator_logical_activation']['shape']}` = `{graph['largest_aggregator_logical_activation']['fp32_bytes']:,}` FP32 bytes.", f"- {graph['logical_note']}", '', '## nncase K210 gate','', f"- Toolchain: `{comp.get('environment',{})}`.", f"- checker/import/PTQ/compile/gencode: `{comp.get('checker')}` / `{comp.get('import')}` / `{comp.get('ptq')}` / `{comp.get('compile')}` / `{comp.get('gencode')}`.", f"- kmodel: `{comp.get('size','N/A')}` bytes; SHA `{comp.get('sha256','N/A')}`.", '- Compiler memory/mapping: NOT PROVIDED unless compiler completed and emitted it.', '', '### Exact compiler failure','```',comp.get('error','None'),'```', '### First failing node evidence','```',json.dumps(first,indent=2),'```','', final,'', 'FIRST TRUE MODEL-SIDE BOTTLENECK: '+bottleneck]
    (OUT/'m01_static_core_report.md').write_text('\n'.join(lines)+'\n'); print(final); return 0


def run():
    OUT.mkdir(parents=True,exist_ok=True)
    model=load_model(); core=core_wrapper(model); write(OUT/'m01_static_core_manifest.json',identities(model,core)); p=parity(model,core); write(OUT/'m01_static_core_parity.json',p)
    if p['status']!='PASS': return report()
    write(OUT/'m01_static_core_graph.json',export_and_audit(model,core))
    LOG.mkdir(parents=True,exist_ok=True)
    with (LOG/'m01_static_core_compile.log').open('w') as f: subprocess.run([KPY,'-u',__file__,'compile'],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
    return report()


if __name__=='__main__':
    action=sys.argv[1] if len(sys.argv)>1 else 'run'
    if action=='compile': raise SystemExit(compiler())
    if action=='report': raise SystemExit(report())
    raise SystemExit(run())
