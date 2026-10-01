#!/usr/bin/env python3
"""R1 extension: evaluate only manifest-frozen 224x168 and 192x144 inputs."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
import traceback
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/superpoint_resolution_gate'
LOG = OUT / 'r1_logs'
ARTIFACT = ROOT / 'models/superpoint_k210_v1_fr12_l1init.pth'
KPY = '/home/quyet/miniconda3/envs/k210/bin/python'
NEW_RESOLUTIONS = ((224, 168), (192, 144))  # width, height only
CAPS = (256, 512, 1024)
CALIBRATION_SEED, CALIBRATION_COUNT = 20260906, 32


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + '\n', encoding='utf-8')


def resolution_name(width: int, height: int) -> str:
    return '{}x{}'.format(width, height)


def paths(width: int, height: int) -> Tuple[Path, Path, Path]:
    stem = 'superpoint_k210_v1_fr12_l1init_{}'.format(resolution_name(width, height))
    return ROOT / 'models' / (stem + '.onnx'), ROOT / 'models' / (stem + '_inferred.onnx'), ROOT / 'artifacts' / (stem + '.kmodel')


def expected_shapes(width: int, height: int) -> Tuple[List[int], List[int], List[int]]:
    assert width % 8 == 0 and height % 8 == 0
    return [1, 1, height, width], [1, 65, height // 8, width // 8], [1, 256, height // 8, width // 8]


def activation_records(model, image) -> List[Dict]:
    """Logical tensors only; deliberately not a peak compiler-memory estimate."""
    import torch
    rows = []
    def record(name, value):
        assert torch.isfinite(value).all()
        rows.append({'name': name, 'shape': list(value.shape), 'elements': value.numel(),
                     'fp32_bytes': value.numel() * 4, 'projected_int8_bytes': value.numel()})
        return value
    x = record('input', image)
    x = record('conv1a', model.conv1a(x)); x = record('relu1a', model.relu(x))
    x = record('conv1b', model.conv1b(x)); x = record('relu1b', model.relu(x)); x = record('pool1', model.pool(x))
    x = record('conv2a', model.conv2a(x)); x = record('relu2a', model.relu(x))
    x = record('conv2b', model.conv2b(x)); x = record('relu2b', model.relu(x)); x = record('pool2', model.pool(x))
    x = record('conv3a', model.conv3a(x)); x = record('relu3a', model.relu(x))
    x = record('conv3b', model.conv3b(x)); x = record('relu3b', model.relu(x)); x = record('pool3', model.pool(x))
    x = record('conv4a', model.conv4a(x)); x = record('relu4a', model.relu(x))
    x = record('conv4b', model.conv4b(x)); x = record('relu4b', model.relu(x))
    det = record('convPa', model.convPa(x)); det = record('reluPa', model.relu(det)); det = record('detector_logits', model.convPb(det))
    desc = record('convDa', model.convDa(x)); desc = record('reluDa', model.relu(desc)); desc = record('descriptor_map', model.convDb(desc))
    return rows


def export(width: int, height: int) -> int:
    import onnx
    import torch
    sys.path.insert(0, str(ROOT / 'scripts'))
    import create_superpoint_k210_w0 as w0
    onnx_path, inferred_path, _ = paths(width, height)
    input_shape, detector_shape, descriptor_shape = expected_shapes(width, height)
    torch.set_num_threads(1)
    model = w0.V1FR12().eval().cpu()
    model.load_state_dict(torch.load(ARTIFACT, map_location='cpu', weights_only=True))
    image = torch.linspace(0, 1, width * height).reshape(input_shape)
    with torch.inference_mode(): outputs = model(image)
    assert [list(value.shape) for value in outputs] == [detector_shape, descriptor_shape]
    records = activation_records(model, image)
    torch.onnx.export(model, image, str(onnx_path), opset_version=13, dynamo=False,
                      do_constant_folding=False, input_names=['image'],
                      output_names=['detector_logits', 'descriptor_map'])
    graph = onnx.load(str(onnx_path)); onnx.checker.check_model(graph)
    inferred = onnx.shape_inference.infer_shapes(graph, strict_mode=True); onnx.checker.check_model(inferred)
    onnx.save(inferred, str(inferred_path))
    assert inferred.graph.value_info
    operators = dict(Counter(node.op_type for node in inferred.graph.node))
    assert set(operators) <= {'Conv', 'Relu', 'MaxPool'}
    specs = {}
    for value in list(inferred.graph.input) + list(inferred.graph.value_info) + list(inferred.graph.output):
        shape = [dimension.dim_value for dimension in value.type.tensor_type.shape.dim]
        assert all(shape)
        specs[value.name] = shape
    assert specs['image'] == input_shape and specs['detector_logits'] == detector_shape and specs['descriptor_map'] == descriptor_shape
    name = resolution_name(width, height)
    save(OUT / ('r1_export_{}.json'.format(name)), {
        'status': 'PASS', 'input_shape': input_shape, 'detector_shape': detector_shape,
        'descriptor_shape': descriptor_shape, 'operators': operators, 'value_info_count': len(inferred.graph.value_info),
        'onnx_sha256': sha256(onnx_path), 'inferred_onnx_sha256': sha256(inferred_path),
        'artifact_sha256': sha256(ARTIFACT), 'parameters': sum(parameter.numel() for parameter in model.parameters()),
        'logical_activations': records,
        'largest_logical_fp32_activation_bytes': max(row['fp32_bytes'] for row in records),
        'largest_projected_int8_activation_bytes': max(row['projected_int8_bytes'] for row in records),
    })
    print('{} export/checker/shape/forward PASS'.format(name))
    return 0


def compile_resolution(width: int, height: int) -> int:
    import _nncase
    import nncase
    import numpy as np
    import onnx
    from importlib.metadata import version
    name = resolution_name(width, height)
    _, inferred_path, kmodel_path = paths(width, height)
    result = {'checker': 'NOT RUN', 'import': 'NOT RUN', 'ptq': 'NOT RUN', 'compile': 'NOT RUN', 'gencode': 'NOT RUN'}
    stage = 'checker'
    try:
        environment = {'python_executable': sys.executable, 'nncase': version('nncase'), '_nncase': _nncase.__version__,
                       'onnx': version('onnx'), 'numpy': version('numpy')}
        assert environment['nncase'] == '1.8.0.20220929', environment
        assert environment['_nncase'] == '1.8.0-55be52f', environment
        result['environment'] = environment
        graph = onnx.load(str(inferred_path)); onnx.checker.check_model(graph); assert graph.graph.value_info
        result[stage] = 'PASS'; stage = 'import'
        options = nncase.CompileOptions(); options.target = 'k210'; options.quant_type = 'uint8'; options.w_quant_type = 'uint8'
        dump = OUT / ('r1_nncase_{}'.format(name)); dump.mkdir(parents=True, exist_ok=True)
        options.dump_dir = str(dump); options.dump_ir = True; options.dump_asm = True; options.dump_quant_error = True
        compiler = nncase.Compiler(options); compiler.import_onnx(inferred_path.read_bytes(), nncase.ImportOptions())
        result[stage] = 'PASS'; stage = 'ptq'
        calibration = np.random.default_rng(CALIBRATION_SEED).uniform(0, 1, size=(CALIBRATION_COUNT, 1, height, width)).astype(np.float32)
        result['calibration_shape'] = list(calibration.shape); result['calibration_sha256'] = hashlib.sha256(calibration.tobytes(order='C')).hexdigest()
        ptq = nncase.PTQTensorOptions(); ptq.samples_count = CALIBRATION_COUNT; ptq.set_tensor_data(calibration.tobytes(order='C'))
        compiler.use_ptq(ptq); result[stage] = 'PASS'; stage = 'compile'
        compiler.compile(); result[stage] = 'PASS'; stage = 'gencode'
        blob = compiler.gencode_tobytes(); assert blob
        kmodel_path.write_bytes(blob); result.update(gencode='PASS', size=len(blob), sha256=sha256(kmodel_path))
        print('{} K210 compile PASS'.format(name))
    except Exception:
        result[stage] = 'FAIL'; result['error'] = traceback.format_exc(); traceback.print_exc()
    save(OUT / ('r1_compiler_{}.json'.format(name)), result)
    return 0 if result['gencode'] == 'PASS' else 1


def load_manifest():
    sys.path.insert(0, str(ROOT / 'scripts'))
    from r0_protocol_loader import load_r0_protocol
    return load_r0_protocol()


def verify_manifest_inputs(manifest: Dict) -> None:
    assert manifest['topk_values'] == list(CAPS)
    assert manifest['models']['v1_fr12_l1init']['checkpoint_sha256'] == sha256(ARTIFACT)
    assert len(manifest['ordered_images']) == 30 == len(manifest['homographies'])
    for image, homography in zip(manifest['ordered_images'], manifest['homographies']):
        path = Path(image['absolute_path'])
        assert path.is_file() and sha256(path) == image['sha256']
        assert homography['image_index'] == image['index'] and homography['relative_path'] == image['relative_path']
        assert all(resolution_name(w, h) in homography['matrices_source_to_warped_xy_pixels'] for w, h in NEW_RESOLUTIONS)


def evaluate() -> int:
    import cv2
    import numpy as np
    import torch
    sys.path.insert(0, str(ROOT / 'scripts'))
    import create_superpoint_k210_w0 as w0
    import superpoint_resolution_gate as r0
    manifest = load_manifest(); verify_manifest_inputs(manifest)
    torch.set_num_threads(1)
    original, sp_module = w0.import_original_superpoint()
    structured = w0.V1FR12().eval().cpu(); structured.load_state_dict(torch.load(ARTIFACT, map_location='cpu', weights_only=True))
    models = [('original', original), ('v1_l1', structured)]
    detector_rows, homography_rows = [], []
    for image, h_entry in zip(manifest['ordered_images'], manifest['homographies']):
        gray = r0.load_gray(Path(image['absolute_path']))
        for width, height in NEW_RESOLUTIONS:
            name = resolution_name(width, height)
            source = r0.resize_tensor(gray, width, height)
            homography = np.asarray(h_entry['matrices_source_to_warped_xy_pixels'][name], dtype=np.float64)
            warped = cv2.warpPerspective(source[0, 0].numpy(), homography, (width, height), flags=cv2.INTER_LINEAR,
                                         borderMode=cv2.BORDER_REFLECT_101)
            target = torch.from_numpy(warped).float()[None, None]
            outputs, transformed = {}, {}
            with torch.inference_mode():
                # Preserve R0 behavior exactly: Original threshold=0.005, V1 manual no threshold after NMS.
                original_raw = original({'image': source}); original_warped = original({'image': target})
                outputs['original'] = {'keypoints': original_raw['keypoints'][0], 'scores': original_raw['scores'][0], 'descriptors': original_raw['descriptors'][0]}
                transformed['original'] = {'keypoints': original_warped['keypoints'][0], 'scores': original_warped['scores'][0], 'descriptors': original_warped['descriptors'][0]}
                logits, dense = structured(source); warped_logits, warped_dense = structured(target)
                outputs['v1_l1'] = w0.postprocess_raw(logits, dense, sp_module)
                transformed['v1_l1'] = w0.postprocess_raw(warped_logits, warped_dense, sp_module)
            for cap in CAPS:
                original_cap = r0.cap_features(outputs['original'], cap)
                for model_name, _ in models:
                    current = r0.cap_features(outputs[model_name], cap)
                    current_warped = r0.cap_features(transformed[model_name], cap)
                    detector_rows.append({'image': image['relative_path'], 'image_index': image['index'], 'resolution': name,
                                          'model': model_name, 'cap': cap,
                                          **r0.detector_metrics(current, None if model_name == 'original' else original_cap, width, height)})
                    homography_rows.append({'image': image['relative_path'], 'image_index': image['index'], 'resolution': name,
                                            'model': model_name, 'cap': cap,
                                            **r0.homography_metrics(current, current_warped, homography, width, height)})
    detector_keys = ('score_mean', 'score_median', 'score_std', 'grid_occupancy', 'xy_spread', 'agreement_at3')
    homography_keys = ('repeatability_1px', 'repeatability_2px', 'repeatability_3px', 'localization_error_px', 'mnn_count', 'mnn_precision', 'geometric_descriptor_cosine')
    aggregates = {}
    for label, rows, keys in (('detector', detector_rows, detector_keys), ('homography', homography_rows, homography_keys)):
        buckets = defaultdict(list)
        for row in rows: buckets[(row['resolution'], row['model'], row['cap'])].append(row)
        aggregates[label] = {'|'.join(map(str, key)): {'images': len(value), **r0.mean_metrics(value, keys)} for key, value in buckets.items()}
    save(OUT / 'r1_extension_metrics.json', {'manifest_path': str(OUT / 'r0_protocol_manifest.json'),
                                              'manifest_sha256': sha256(OUT / 'r0_protocol_manifest.json'),
                                              'models': ['original', 'v1_l1'], 'resolutions': [resolution_name(*v) for v in NEW_RESOLUTIONS],
                                              'detector_per_image': detector_rows, 'homography_per_image': homography_rows,
                                              'detector_aggregate': aggregates['detector'], 'homography_aggregate': aggregates['homography']})
    with (OUT / 'r1_extension_per_image.csv').open('w', newline='', encoding='utf-8') as handle:
        rows = [{**row, 'category': 'detector'} for row in detector_rows] + [{**row, 'category': 'homography'} for row in homography_rows]
        fields = sorted({key for row in rows for key in row}); writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    print('R1 evaluate PASS: {} images, no historical resolution inference'.format(len(manifest['ordered_images'])))
    return 0


def compiler_memory(log: Path) -> str:
    if not log.exists(): return 'NOT PROVIDED'
    text = log.read_text(encoding='utf-8'); start = text.find('MEMORY USAGES')
    if start < 0: return 'NOT PROVIDED'
    tail = text[start:]
    return '\n'.join(line for line in tail.splitlines() if not line.endswith('K210 compile PASS')).strip()


def compiler_warnings(log: Path) -> str:
    if not log.exists(): return 'NOT PROVIDED'
    warnings = [line for line in log.read_text(encoding='utf-8').splitlines() if line.startswith('WARN:')]
    return '; '.join(warnings) if warnings else 'none'


def report() -> int:
    r0 = json.loads((OUT / 'r0_metrics.json').read_text())
    r1 = json.loads((OUT / 'r1_extension_metrics.json').read_text())
    rows_d = {**{key: value for key, value in r0['detector_aggregate'].items() if '|v1_random|' not in key}, **r1['detector_aggregate']}
    rows_h = {**{key: value for key, value in r0['homography_aggregate'].items() if '|v1_random|' not in key}, **r1['homography_aggregate']}
    exports, compilers = {}, {}
    for width, height in NEW_RESOLUTIONS:
        name = resolution_name(width, height)
        exports[name] = json.loads((OUT / 'r1_export_{}.json'.format(name)).read_text())
        compilers[name] = json.loads((OUT / 'r1_compiler_{}.json'.format(name)).read_text())
    lines = ['# SP-K210 R1 resolution extension', '',
             'Only `224x168` and `192x144` were evaluated. Historical `320x240`/`256x192` values below were loaded from `r0_metrics.json`; no inference was rerun for them. Inputs and homographies were loaded verbatim from `r0_protocol_manifest.json`.', '',
             '## Scope and fairness', '',
             '- This is **within-model resolution sensitivity only**. Do not interpret Original-vs-V1 absolute values below as a fair cross-model quality comparison.',
             '- Original preserved its historical R0 threshold (`0.005`). V1-L1 preserved historical R0 behavior: no threshold after NMS. Both use the same frozen image list, pre-frozen matrices, resize, NMS/border/descriptor methods, caps and metric definitions.',
             '- No training, fine-tuning, distillation, pruning, QAT, architecture/descriptor change, checkpoint modification, image discovery, homography generation, source-project modification or commit.', '',
             '## New-resolution compiler feasibility', '',
             '| Resolution | Expected V1 output detector / descriptor | Forward + ONNX checker/inference | nncase import/PTQ/compile/gencode | Largest logical FP32 / projected INT8 activation | kmodel bytes | SHA256 |',
             '|---|---|---|---|---:|---:|---|']
    for width, height in NEW_RESOLUTIONS:
        name = resolution_name(width, height); e, c = exports[name], compilers[name]
        status = '/'.join(c.get(key, 'NOT RUN') for key in ('import', 'ptq', 'compile', 'gencode'))
        lines.append('| {} | `{}` / `{}` | {} | {} | {} / {} | {} | `{}` |'.format(name, e['detector_shape'], e['descriptor_shape'], e['status'], status,
            e['largest_logical_fp32_activation_bytes'], e['largest_projected_int8_activation_bytes'], c.get('size', 'N/A'), c.get('sha256', 'N/A')))
        log = LOG / ('compile_{}.log'.format(name))
        lines += ['', '### {} compiler memory information'.format(name), '```', compiler_memory(log), '```',
                  '- Compiler warnings: `{}`.'.format(compiler_warnings(log))]
    lines += ['', 'Compiler PASS is graph feasibility only; it does not prove K210 board runtime, latency, camera integration, localization accuracy or full-system RAM fit.', '',
              '## Detector / spatial metrics (mean over frozen 30 images)', '',
              '| Resolution | Model | Cap | Score mean | Score median | Score std | 4x4 coverage | XY spread |', '|---|---|---:|---:|---:|---:|---:|---:|']
    for name in ('320x240', '256x192', '224x168', '192x144'):
        for model in ('original', 'v1_l1'):
            for cap in CAPS:
                d = rows_d['{}|{}|{}'.format(name, model, cap)]
                lines.append('| {} | {} | {} | {:.5f} | {:.5f} | {:.5f} | {:.3f} | {:.3f} |'.format(name, model, cap, d['score_mean'], d['score_median'], d['score_std'], d['grid_occupancy'], d['xy_spread']))
    lines += ['', '## Homography metrics (mean over frozen 30 images)', '',
              '| Resolution | Model | Cap | Repeat @1 | @2 | @3 | Localization error px | MNN count | MNN precision @3 | Descriptor cosine |', '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for name in ('320x240', '256x192', '224x168', '192x144'):
        for model in ('original', 'v1_l1'):
            for cap in CAPS:
                h = rows_h['{}|{}|{}'.format(name, model, cap)]
                lines.append('| {} | {} | {} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.1f} | {:.3f} | {:.3f} |'.format(name, model, cap, h['repeatability_1px'], h['repeatability_2px'], h['repeatability_3px'], h['localization_error_px'], h['mnn_count'], h['mnn_precision'], h['geometric_descriptor_cosine']))
    # Decision locked before reading new values: a >10pp adjacent resolution loss in either within-model R@3 or MNN precision at a required cap is a knee.
    order = ('320x240', '256x192', '224x168', '192x144'); deltas = []; knee_at = None
    for previous, current in zip(order, order[1:]):
        clear_knee = False
        for model in ('original', 'v1_l1'):
            for cap in CAPS:
                before, after = rows_h['{}|{}|{}'.format(previous, model, cap)], rows_h['{}|{}|{}'.format(current, model, cap)]
                dr, dp = after['repeatability_3px'] - before['repeatability_3px'], after['mnn_precision'] - before['mnn_precision']
                deltas.append((previous, current, model, cap, dr, dp))
                clear_knee |= dr < -0.10 or dp < -0.10
        if clear_knee and knee_at is None: knee_at = previous
    recommendation = knee_at or order[-1]
    lines += ['', '## Within-model resolution deltas and decision', '',
              'Predefined knee rule: for either model and any required cap, a drop greater than 0.10 absolute in repeatability@3px or MNN precision@3px between adjacent resolutions is a clear quality knee. Recommend the immediately larger resolution before the first knee; otherwise recommend the smallest tested resolution.', '',
              '| From → to | Model | Cap | Delta repeatability@3 | Delta MNN precision@3 |', '|---|---|---:|---:|---:|']
    for previous, current, model, cap, dr, dp in deltas:
        lines.append('| {} → {} | {} | {} | {:+.3f} | {:+.3f} |'.format(previous, current, model, cap, dr, dp))
    lines += ['', 'This recommendation uses only within-model deltas; it does not choose a model or state an absolute cross-model quality ranking.', '',
              '## Files', '',
              '- `scripts/superpoint_resolution_r1.py`.', '- `models/superpoint_k210_v1_fr12_l1init_224x168*.onnx` and `...192x144*.onnx`.',
              '- K210 kmodels/compiler logs and JSON under `reports/superpoint_resolution_gate/`.',
              '- `r1_extension_metrics.json` and `r1_extension_per_image.csv`.', '',
              'SP-K210 RESOLUTION FINAL RECOMMENDATION: {}'.format(recommendation)]
    (OUT / 'r1_extension.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('SP-K210 RESOLUTION FINAL RECOMMENDATION: {}'.format(recommendation))
    return 0


def logged(command: List[str], log_name: str) -> int:
    LOG.mkdir(parents=True, exist_ok=True)
    with (LOG / log_name).open('w', encoding='utf-8') as handle:
        return subprocess.run(command, cwd=str(ROOT), stdout=handle, stderr=subprocess.STDOUT).returncode


def run() -> int:
    for width, height in NEW_RESOLUTIONS:
        name = resolution_name(width, height)
        if logged([sys.executable, '-u', __file__, 'export', '--resolution', name], 'export_{}.log'.format(name)): raise RuntimeError('{} export failed'.format(name))
        if logged([KPY, '-u', __file__, 'compile', '--resolution', name], 'compile_{}.log'.format(name)): raise RuntimeError('{} compiler failed'.format(name))
    if logged([sys.executable, '-u', __file__, 'evaluate'], 'evaluate.log'): raise RuntimeError('R1 evaluation failed')
    return report()


def parse_resolution(value: str) -> Tuple[int, int]:
    try: width, height = (int(part) for part in value.split('x'))
    except ValueError as error: raise ValueError('Resolution must be WIDTHxHEIGHT') from error
    if (width, height) not in NEW_RESOLUTIONS: raise ValueError('Only 224x168 and 192x144 are allowed')
    return width, height


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument('action', choices=('run', 'export', 'compile', 'evaluate', 'report'))
    parser.add_argument('--resolution'); args = parser.parse_args()
    if args.action in ('export', 'compile') and not args.resolution: parser.error('--resolution required')
    if args.action == 'run': return run()
    if args.action == 'export': return export(*parse_resolution(args.resolution))
    if args.action == 'compile': return compile_resolution(*parse_resolution(args.resolution))
    if args.action == 'evaluate': return evaluate()
    return report()


if __name__ == '__main__':
    try: raise SystemExit(main())
    except Exception as error:
        if len(sys.argv) > 1 and sys.argv[1] == 'run':
            (OUT / 'r1_extension.md').write_text('# R1 extension\n\nFailure: `{}`\n'.format(error), encoding='utf-8')
        raise
