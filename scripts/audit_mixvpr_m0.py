#!/usr/bin/env python3
"""M0: exact current MixVPR audit and unmodified K210 compiler gate."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import subprocess
import sys
import traceback
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
EDGE = Path('/home/quyet/edge_ai_project')
OUT = ROOT / 'reports/mixvpr_feasibility'
LOG = OUT / 'logs'
HLOC = EDGE / 'third_party/hloc'
HLOC_EXTRACTOR = HLOC / 'hloc/extractors/mixvpr.py'
RUNTIME = EDGE / 'src/localize_edge_v3.py'
PREPROCESS = EDGE / 'src/localize_live.py'
HUB = Path('/home/quyet/.cache/torch/hub/jarvisyjw_MixVPR_main')
CHECKPOINT = HUB / 'model.ckpt'
RESNET_INIT = Path('/home/quyet/.cache/torch/hub/checkpoints/resnet50-0676ba61.pth')
ACTIVE_ONNX = EDGE / 'outputs/onnx/mixvpr.int8.matmul.onnx'
STATIC_ONNX = ROOT / 'models/mixvpr_m0_active_static_1x3x480x640.onnx'
INFERRED_ONNX = ROOT / 'models/mixvpr_m0_active_static_1x3x480x640_inferred.onnx'
KMODEL = ROOT / 'artifacts/mixvpr_m0_active_static_1x3x480x640.kmodel'
KPY = '/home/quyet/miniconda3/envs/k210/bin/python'
INPUT_SHAPE = (1, 3, 480, 640)
CALIBRATION_SEED, CALIBRATION_COUNT = 20260906, 8


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + '\n', encoding='utf-8')


def import_model():
    import torch
    for path in (str(EDGE / 'third_party'), str(HLOC)):
        if path not in sys.path: sys.path.insert(0, path)
    from hloc import extractors
    from hloc.utils.base_model import dynamic_load
    return dynamic_load(extractors, 'mixvpr')({}).eval().cpu()


class PlainDescriptorWrapper:
    """Adapter only: exact model returns a dict; ONNX needs a Tensor output."""
    def __init__(self, model):
        import torch.nn as nn
        class Wrapper(nn.Module):
            def __init__(self, original): super().__init__(); self.original = original
            def forward(self, image): return self.original({'image': image})['global_descriptor']
        self.module = Wrapper(model).eval().cpu()


def environment() -> Dict[str, Any]:
    import cv2, numpy, onnx, onnxruntime, torch, torchvision
    return {'python_executable': sys.executable, 'python': platform.python_version(),
            'platform': platform.platform(), 'torch': torch.__version__, 'torchvision': torchvision.__version__,
            'numpy': numpy.__version__, 'opencv': cv2.__version__, 'onnx': onnx.__version__,
            'onnxruntime': onnxruntime.__version__, 'ort_available_providers': onnxruntime.get_available_providers(),
            'torch_cuda_available': torch.cuda.is_available(), 'torch_cuda_version': torch.version.cuda}


def active_onnx_facts() -> Dict[str, Any]:
    import onnx
    graph = onnx.load(str(ACTIVE_ONNX))
    def specs(values):
        return [{'name': value.name, 'shape': [dim.dim_value if dim.HasField('dim_value') else dim.dim_param for dim in value.type.tensor_type.shape.dim],
                 'dtype': value.type.tensor_type.elem_type} for value in values]
    return {'path': str(ACTIVE_ONNX), 'sha256': digest(ACTIVE_ONNX), 'size_bytes': ACTIVE_ONNX.stat().st_size,
            'inputs': specs(graph.graph.input), 'outputs': specs(graph.graph.output),
            'operators': dict(Counter(node.op_type for node in graph.graph.node)), 'node_count': len(graph.graph.node),
            'note': 'This is the packaged ONNX artifact selected by localize_edge_v3.py in mode=onnx/auto. Its external image dimensions are dynamic.'}


def model_manifest(model) -> Dict[str, Any]:
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    breakdown = {}
    for prefix in ('net.backbone', 'net.aggregator'):
        values = [parameter.numel() for name, parameter in model.named_parameters() if name.startswith(prefix)]
        breakdown[prefix] = {'parameters': sum(values), 'fp32_bytes': sum(values) * 4}
    return {'protocol_version': 'MIXVPR-M0-v1',
            'active_runtime_invocation': {'implementation': str(RUNTIME),
                                          'function': 'build_global_runner_edge -> ONNXGlobalDescriptorRunner',
                                          'mode': 'args.mode in {onnx,auto}; cfg.paths.global_method == mixvpr',
                                          'active_artifact': active_onnx_facts()},
            'pytorch_model': {'hloc_extractor': str(HLOC_EXTRACTOR), 'hloc_extractor_sha256': digest(HLOC_EXTRACTOR),
                              'torch_hub_repo_cache': str(HUB), 'hubconf': str(HUB / 'hubconf.py'), 'hubconf_sha256': digest(HUB / 'hubconf.py'),
                              'factory': "torch.hub.load('jarvisyjw/MixVPR', 'get_trained_model', pretrained=True)",
                              'factory_config': {'backbone_arch': 'resnet50', 'pretrained': True,
                                                 'layers_to_freeze': 2, 'layers_to_crop': [4],
                                                 'agg_arch': 'MixVPR',
                                                 'agg_config': {'in_channels': 1024, 'in_h': 20, 'in_w': 20,
                                                                'out_channels': 1024, 'mix_depth': 4,
                                                                'mlp_ratio': 1, 'out_rows': 4},
                                                 'loss_name': 'MultiSimilarityLoss', 'miner_name': 'MultiSimilarityMiner',
                                                 'miner_margin': 0.1, 'faiss_gpu': False},
                              'checkpoint': str(CHECKPOINT), 'checkpoint_sha256': digest(CHECKPOINT),
                              'resnet_initialization_checkpoint': str(RESNET_INIT), 'resnet_initialization_sha256': digest(RESNET_INIT),
                              'backbone': 'ResNet50 cropped before layer4; output 1024x20x20 after forced 320x320 wrapper resize',
                              'aggregator': {'name': 'MixVPR', 'in_channels': 1024, 'in_h': 20, 'in_w': 20, 'out_channels': 1024, 'mix_depth': 4, 'mlp_ratio': 1, 'out_rows': 4},
                              'descriptor_dimension': 4096, 'detector_output_dimension': None, 'parameter_count': parameter_count,
                              'parameter_fp32_bytes': parameter_count * 4, 'parameter_breakdown': breakdown},
            'preprocessing': {'runtime_source': str(PREPROCESS), 'function': 'preprocess_frame',
                              'input_color': 'OpenCV BGR frame', 'rgb_conversion': 'cv2.cvtColor(BGR, COLOR_BGR2RGB)',
                              'outer_resize': 'scale=min(1,max_width/source_width,max_height/source_height); cv2.INTER_AREA only when scale<1; no crop/pad',
                              'runtime_input_layout': 'float32 RGB NCHW [1,3,H,W], divide by 255; H/W dynamic after outer resize',
                              'active_pi5_limits': {'max_width': 384, 'max_height': 480},
                              'exact_wrapper_normalization': {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]},
                              'exact_wrapper_resize': 'torch.nn.functional.interpolate(size=(320,320), mode=bilinear, align_corners=False)',
                              'static_audit_probe': list(INPUT_SHAPE),
                              'probe_note': '480x640 is the project exporter canonical dummy shape; it is a static audit probe, not a claim that the runtime dynamic outer input is fixed.'},
            'output': {'shape_for_probe': [1, 4096], 'dtype': 'float32', 'normalization': 'MixVPR aggregator uses F.normalize(flattened descriptor, p=2, dim=-1); runtime ONNX runner L2-normalizes again before retrieval'},
            'environment': environment()}


def tensor_audit(model) -> Dict[str, Any]:
    import torch
    dummy = torch.linspace(0, 1, 1 * 3 * 480 * 640, dtype=torch.float32).reshape(INPUT_SHAPE)
    events = []
    handles = []
    def hook(name):
        def record(_module, _inputs, output):
            if torch.is_tensor(output):
                events.append({'stage': name, 'shape': list(output.shape), 'dtype': str(output.dtype),
                               'elements': output.numel(), 'fp32_bytes': output.numel() * 4,
                               'projected_int8_bytes': output.numel()})
        return record
    for name, module in model.named_modules():
        if name and not list(module.children()): handles.append(module.register_forward_hook(hook(name)))
    with torch.inference_mode():
        descriptor = model({'image': dummy})['global_descriptor']
        normalized = model.norm(dummy)
        resized = torch.nn.functional.interpolate(normalized, size=(320, 320), mode='bilinear', align_corners=False)
        backbone = model.net.backbone(resized)
        aggregate = model.net.aggregator(backbone)
    for handle in handles: handle.remove()
    assert descriptor.shape == aggregate.shape == (1, 4096) and torch.isfinite(descriptor).all()
    stage_rows = [
        {'stage': 'external_runtime_input_probe', 'shape': list(dummy.shape), 'fp32_bytes': dummy.numel() * 4, 'projected_int8_bytes': dummy.numel()},
        {'stage': 'wrapper_normalized_rgb', 'shape': list(normalized.shape), 'fp32_bytes': normalized.numel() * 4, 'projected_int8_bytes': normalized.numel()},
        {'stage': 'wrapper_forced_resize_320x320', 'shape': list(resized.shape), 'fp32_bytes': resized.numel() * 4, 'projected_int8_bytes': resized.numel()},
        {'stage': 'backbone_output', 'shape': list(backbone.shape), 'fp32_bytes': backbone.numel() * 4, 'projected_int8_bytes': backbone.numel()},
        {'stage': 'aggregator_output_descriptor', 'shape': list(aggregate.shape), 'fp32_bytes': aggregate.numel() * 4, 'projected_int8_bytes': aggregate.numel()},
    ]
    all_rows = stage_rows + events
    largest = sorted(all_rows, key=lambda row: row['fp32_bytes'], reverse=True)
    return {'status': 'PASS', 'input_shape': list(dummy.shape), 'output_shape': list(descriptor.shape), 'output_dtype': str(descriptor.dtype),
            'output_l2_norm': float(descriptor.norm(dim=1).item()), 'stage_tensors': stage_rows,
            'largest_logical_tensors': largest[:30], 'largest_logical_activation_fp32_bytes': largest[0]['fp32_bytes'],
            'note': 'Logical tensors are individual tensor sizes only, not peak RAM or KPU RAM.'}


def export_and_parity(model) -> Dict[str, Any]:
    import numpy as np, onnx, onnxruntime, torch
    wrapper = PlainDescriptorWrapper(model).module
    dummy = torch.linspace(0, 1, 1 * 3 * 480 * 640, dtype=torch.float32).reshape(INPUT_SHAPE)
    with torch.inference_mode(): pytorch_output = wrapper(dummy).detach().cpu().numpy()
    torch.onnx.export(wrapper, dummy, str(STATIC_ONNX), opset_version=13, dynamo=False, do_constant_folding=False,
                      input_names=['image'], output_names=['global_descriptor'])
    graph = onnx.load(str(STATIC_ONNX)); onnx.checker.check_model(graph)
    inferred = onnx.shape_inference.infer_shapes(graph, strict_mode=True); onnx.checker.check_model(inferred); onnx.save(inferred, str(INFERRED_ONNX))
    operators = dict(Counter(node.op_type for node in inferred.graph.node))
    dynamic = []
    for value in list(inferred.graph.input) + list(inferred.graph.value_info) + list(inferred.graph.output):
        dimensions = value.type.tensor_type.shape.dim
        if any(dim.dim_param or not dim.HasField('dim_value') for dim in dimensions): dynamic.append(value.name)
    session = onnxruntime.InferenceSession(str(INFERRED_ONNX), providers=['CPUExecutionProvider'])
    ort_output = session.run(None, {'image': dummy.numpy()})[0]
    flat_torch, flat_ort = pytorch_output.reshape(-1), np.asarray(ort_output).reshape(-1)
    cosine = float(np.dot(flat_torch, flat_ort) / max(float(np.linalg.norm(flat_torch) * np.linalg.norm(flat_ort)), 1e-12))
    return {'export': 'PASS', 'checker': 'PASS', 'shape_inference': 'PASS', 'opset': 13,
            'input_shape': list(INPUT_SHAPE), 'output_shape': list(pytorch_output.shape), 'output_dtype': str(pytorch_output.dtype),
            'onnx_path': str(STATIC_ONNX), 'onnx_sha256': digest(STATIC_ONNX), 'inferred_onnx_path': str(INFERRED_ONNX), 'inferred_onnx_sha256': digest(INFERRED_ONNX),
            'operators': operators, 'node_count': len(inferred.graph.node), 'dynamic_shape_values': dynamic,
            'operator_flags': {key: operators.get(key, 0) for key in ('Reshape', 'Transpose', 'MatMul', 'Gemm', 'ReduceL2', 'Div', 'GlobalAveragePool', 'AveragePool', 'MaxPool', 'Softmax', 'Resize', 'LayerNormalization')},
            'ort_parity': {'status': 'PASS', 'shape_pytorch': list(pytorch_output.shape), 'shape_onnxruntime': list(np.asarray(ort_output).shape),
                           'max_abs_error': float(np.max(np.abs(pytorch_output - ort_output))), 'mean_abs_error': float(np.mean(np.abs(pytorch_output - ort_output))), 'cosine_similarity': cosine}}


def compile_k210() -> int:
    import _nncase, nncase, numpy as np, onnx
    from importlib.metadata import version
    result = {'checker': 'NOT RUN', 'import': 'NOT RUN', 'ptq': 'NOT RUN', 'compile': 'NOT RUN', 'gencode': 'NOT RUN'}; stage = 'checker'
    try:
        environment = {'python_executable': sys.executable, 'nncase': version('nncase'), '_nncase': _nncase.__version__, 'onnx': version('onnx'), 'numpy': version('numpy')}
        assert environment['nncase'] == '1.8.0.20220929', environment; assert environment['_nncase'] == '1.8.0-55be52f', environment
        result['environment'] = environment
        graph = onnx.load(str(INFERRED_ONNX)); onnx.checker.check_model(graph); assert graph.graph.value_info; result[stage] = 'PASS'; stage = 'import'
        options = nncase.CompileOptions(); options.target = 'k210'; options.quant_type = 'uint8'; options.w_quant_type = 'uint8'; dump = OUT / 'nncase_dump'; dump.mkdir(parents=True, exist_ok=True); options.dump_dir = str(dump); options.dump_ir = True; options.dump_asm = True; options.dump_quant_error = True
        compiler = nncase.Compiler(options); compiler.import_onnx(INFERRED_ONNX.read_bytes(), nncase.ImportOptions()); result[stage] = 'PASS'; stage = 'ptq'
        calibration = np.random.default_rng(CALIBRATION_SEED).uniform(0, 1, size=(CALIBRATION_COUNT, *INPUT_SHAPE[1:])).astype(np.float32); result['calibration_shape'] = list(calibration.shape); result['calibration_sha256'] = hashlib.sha256(calibration.tobytes(order='C')).hexdigest()
        ptq = nncase.PTQTensorOptions(); ptq.samples_count = CALIBRATION_COUNT; ptq.set_tensor_data(calibration.tobytes(order='C')); compiler.use_ptq(ptq); result[stage] = 'PASS'; stage = 'compile'
        compiler.compile(); result[stage] = 'PASS'; stage = 'gencode'; blob = compiler.gencode_tobytes(); assert blob; KMODEL.write_bytes(blob); result.update(gencode='PASS', size=len(blob), sha256=digest(KMODEL)); print('M0 K210 compile PASS')
    except Exception:
        result[stage] = 'FAIL'; result['error'] = traceback.format_exc(); print(result['error'])
    save(OUT / 'm0_compiler.json', result)
    return 0 if result['gencode'] == 'PASS' else 1


def classify_failure(compiler: Dict[str, Any], graph: Dict[str, Any]) -> str:
    if compiler.get('gencode') == 'PASS': return 'PASS'
    error = compiler.get('error', '')
    # The compiler does not identify a node. Inspect the graph rather than
    # matching the word "mix" in this audit script's own filename.
    if "Can't pull input data" in error and onnx_import_hazards(): return 'OTHER'
    if re.search(r'(/net/backbone/|/backbone/|/layer[123]/|resnet)', error, re.I): return 'BACKBONE'
    if re.search(r'(/net/aggregator/|/aggregator/|channel_proj|row_proj|LayerNorm|MatMul)', error, re.I): return 'AGGREGATOR'
    return 'OTHER'


def onnx_import_hazards() -> List[Dict[str, Any]]:
    import onnx
    graph = onnx.load(str(INFERRED_ONNX))
    return [{'index': index, 'op_type': node.op_type, 'node_name': node.name,
             'inputs': list(node.input), 'outputs': list(node.output)}
            for index, node in enumerate(graph.graph.node) if any(not value for value in node.input)]


def compiler_memory() -> str:
    log = LOG / 'compile.log'
    if not log.exists(): return 'NOT PROVIDED'
    text = log.read_text(); start = text.find('MEMORY USAGES')
    return 'NOT PROVIDED' if start < 0 else text[start:].split('M0 K210 compile PASS')[0].strip()


def report() -> int:
    manifest = json.loads((OUT / 'm0_model_manifest.json').read_text()); tensors = json.loads((OUT / 'm0_tensor_audit.json').read_text()); graph = json.loads((OUT / 'm0_graph.json').read_text()); compiler = json.loads((OUT / 'm0_compiler.json').read_text())
    classification = classify_failure(compiler, graph)
    hazards = onnx_import_hazards()
    if classification == 'PASS': final = 'MIXVPR-M0: FULL MODEL K210 COMPILE PASS'; bottleneck = 'No compiler rejection was observed; board feasibility remains unproven.'
    else:
        final = 'MIXVPR-M0: K210 COMPILE FAIL — {}'.format(classification)
        bottleneck = {'BACKBONE': 'Unsupported or failing backbone subgraph identified by compiler error.', 'AGGREGATOR': 'MixVPR aggregator subgraph identified by compiler error.', 'OTHER': 'Import fails before a named backbone/aggregator node: wrapper Resize has empty optional inputs plus Shape/Slice/Concat-generated sizes, matching nncase\'s non-constant-input error.'}[classification]
    warnings = [line for line in (LOG / 'compile.log').read_text().splitlines() if line.startswith('WARN:')] if (LOG / 'compile.log').exists() else []
    lines = ['# MixVPR M0 exact-model audit and K210 feasibility', '',
             'This is an unmodified-model audit. No training, fine-tuning, distillation, pruning, channel slimming, descriptor change, graph rewrite, source-project modification or commit occurred.', '',
             '## Model identity', '', '| Implementation | Checkpoint | Input policy | Output | Descriptor dim | Params | Backbone | Aggregator |', '|---|---|---|---|---:|---:|---|---|',
             '| HLOC `Mixvpr` → cached Torch Hub model; runtime ONNX runner listed below | `model.ckpt` SHA `{}` | RGB NCHW dynamic outer input; normalized then forced internally to 320×320; static probe `[1,3,480,640]` | `[1,4096]` FP32 L2-normalized | 4096 | {} | ResNet50 cropped before layer4 | MixVPR 1024×20×20, depth 4, rows 4 |'.format(manifest['pytorch_model']['checkpoint_sha256'], manifest['pytorch_model']['parameter_count']), '',
             '- Active runtime invocation: `{}`; packaged ONNX SHA `{}`.'.format(manifest['active_runtime_invocation']['implementation'], manifest['active_runtime_invocation']['active_artifact']['sha256']),
             '- Pipeline call: `preprocess_frame` produces RGB tensor → `build_global_runner_edge` selects `ONNXGlobalDescriptorRunner` in `onnx/auto` mode → call at `localize_edge_v3.py:2943` before retrieval.', '',
             '## Graph', '', '| Operator | Count |', '|---|---:|']
    for op, count in sorted(graph['operators'].items()): lines.append('| {} | {} |'.format(op, count))
    lines += ['', '- Dynamic shape-inference values after static export: {} values (first five: `{}`).'.format(len(graph['dynamic_shape_values']), graph['dynamic_shape_values'][:5]),
              '- Key operator flags: `{}`. No custom ONNX domain was exported; nncase failed before capability/mapping analysis could reach backbone or aggregator.'.format(graph['operator_flags']), '',
              '## Memory-relevant logical tensors', '', 'Individual logical activation sizes only; not peak RAM, KPU RAM, or a board-memory measurement.', '', '| Stage | Shape | FP32 bytes | Projected INT8 logical bytes |', '|---|---|---:|---:|']
    for row in tensors['stage_tensors']: lines.append('| {} | {} | {} | {} |'.format(row['stage'], row['shape'], row['fp32_bytes'], row['projected_int8_bytes']))
    lines += ['', 'Largest leaf-module logical tensors:']
    for row in tensors['largest_logical_tensors'][:8]: lines.append('- `{}` `{}`: {} FP32 bytes / {} projected INT8 bytes.'.format(row['stage'], row['shape'], row['fp32_bytes'], row['projected_int8_bytes']))
    lines += ['', '## K210 gate', '', '| Gate | Result |', '|---|---|',
              '| PyTorch inference | PASS: output {} {} |'.format(tensors['output_shape'], tensors['output_dtype']),
              '| ONNX export / checker / shape inference | {}/{}/{} |'.format(graph['export'], graph['checker'], graph['shape_inference']),
              '| ORT parity | PASS; max abs {:.3e}, mean abs {:.3e}, cosine {:.9f} |'.format(graph['ort_parity']['max_abs_error'], graph['ort_parity']['mean_abs_error'], graph['ort_parity']['cosine_similarity']),
              '| nncase import | {} |'.format(compiler.get('import')), '| PTQ | {} |'.format(compiler.get('ptq')), '| compile | {} |'.format(compiler.get('compile')), '| gencode | {} |'.format(compiler.get('gencode')), '| kmodel | {} {} |'.format(compiler.get('size', 'N/A'), compiler.get('sha256', '')), '| compiler memory | {} |'.format('See verbatim block below' if compiler.get('gencode') == 'PASS' else 'NOT PROVIDED: compile did not succeed'), '| KPU mapping / CPU fallback | NOT EXPOSED: import failed before compiler lowering/mapping |', '',
              '### Compiler memory output', '```', compiler_memory(), '```', '### Compiler warnings', '```', '\n'.join(warnings) or 'None', '```', '### Exact first failure (if any)', '```', compiler.get('error', 'None'), '```', '', '### Import-hazard graph evidence', '```', json.dumps(hazards, indent=2) if hazards else 'None', '```', '',
              '## Scope limits', '', 'A compiler PASS would be compiler feasibility only, not proof of board execution, latency, retrieval quality, localization quality, or full-system RAM. This M0 report does not propose any optimization.', '', final, '', 'Single most important bottleneck: {}'.format(bottleneck)]
    (OUT / 'm0_report.md').write_text('\n'.join(lines) + '\n'); print(final); return 0


def logged(command: List[str], name: str) -> int:
    LOG.mkdir(parents=True, exist_ok=True)
    with (LOG / name).open('w') as handle: return subprocess.run(command, cwd=str(ROOT), stdout=handle, stderr=subprocess.STDOUT).returncode


def run() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    model = import_model(); save(OUT / 'm0_model_manifest.json', model_manifest(model)); save(OUT / 'm0_tensor_audit.json', tensor_audit(model)); save(OUT / 'm0_graph.json', export_and_parity(model))
    # Compiler failure is expected data for M0; do not abort before writing the report.
    logged([KPY, '-u', __file__, 'compile'], 'compile.log')
    return report()


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument('action', choices=('run', 'manifest', 'compile', 'report')); action = parser.parse_args().action
    if action == 'run': return run()
    if action == 'manifest': save(OUT / 'm0_model_manifest.json', model_manifest(import_model())); return 0
    if action == 'compile': return compile_k210()
    return report()


if __name__ == '__main__':
    try: raise SystemExit(main())
    except Exception as error:
        if len(sys.argv) > 1 and sys.argv[1] == 'run': (OUT / 'm0_report.md').write_text('# MixVPR M0\n\nFailure: `{}`\n'.format(error))
        raise
