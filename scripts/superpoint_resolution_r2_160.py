#!/usr/bin/env python3
"""R2: one manifest-derived 160x120 resolution candidate; no historical reruns."""
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
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/superpoint_resolution_gate'
LOG = OUT / 'r2_160_logs'
ARTIFACT = ROOT / 'models/superpoint_k210_v1_fr12_l1init.pth'
KPY = '/home/quyet/miniconda3/envs/k210/bin/python'
WIDTH, HEIGHT, NAME = 160, 120, '160x120'
CAPS = (256, 512, 1024)
CALIBRATION_SEED, CALIBRATION_COUNT = 20260906, 32
R0_MANIFEST = OUT / 'r0_protocol_manifest.json'
EXTENSION = OUT / 'r2_160_protocol_extension.json'


def sha256(path: Path) -> str: return hashlib.sha256(path.read_bytes()).hexdigest()
def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
def model_paths():
    stem = ROOT / 'models' / 'superpoint_k210_v1_fr12_l1init_160x120'
    return Path(str(stem) + '.onnx'), Path(str(stem) + '_inferred.onnx'), ROOT / 'artifacts/superpoint_k210_v1_fr12_l1init_160x120.kmodel'


def scale(width: int, height: int):
    import numpy as np
    return np.array([[width - 1, 0, 0], [0, height - 1, 0], [0, 0, 1]], dtype=np.float64)


def extension_manifest() -> Dict[str, Any]:
    """Derive pixel H from frozen canonical H; do not import/call the R0 generator."""
    import numpy as np
    base = json.loads(R0_MANIFEST.read_text(encoding='utf-8'))
    assert base['protocol_version'] == 'SP-K210-R0-frozen-v1'
    assert base['topk_values'] == list(CAPS)
    entries = []
    source_scale = scale(320, 240); target_scale = scale(WIDTH, HEIGHT)
    for image, old in zip(base['ordered_images'], base['homographies']):
        matrices = old['matrices_source_to_warped_xy_pixels']
        h320 = np.asarray(matrices['320x240'], dtype=np.float64)
        h256 = np.asarray(matrices['256x192'], dtype=np.float64)
        canonical = np.linalg.inv(source_scale) @ h320 @ source_scale
        canonical_from_256 = np.linalg.inv(scale(256, 192)) @ h256 @ scale(256, 192)
        assert np.allclose(canonical, canonical_from_256, rtol=0.0, atol=1e-14), image['relative_path']
        h160 = target_scale @ canonical @ np.linalg.inv(target_scale)
        entries.append({'image_index': image['index'], 'relative_path': image['relative_path'],
                        'image_sha256': image['sha256'],
                        'canonical_normalized_source_to_warped': canonical.tolist(),
                        'matrix_source_to_warped_xy_pixels_160x120': h160.tolist()})
    return {'protocol_version': 'SP-K210-R2-160-extension-v1',
            'base_r0_manifest_path': str(R0_MANIFEST), 'base_r0_manifest_sha256': sha256(R0_MANIFEST),
            'resolution': NAME, 'image_count': len(entries),
            'derivation': {'source': 'frozen R0 320x240 pixel H',
                           'canonicalization': 'canonical = inverse(S_320x240) @ H_320x240 @ S_320x240',
                           'pixel_conversion': 'H_160x120 = S_160x120 @ canonical @ inverse(S_160x120)',
                           'S_width_height': '[[width-1,0,0],[0,height-1,0],[0,0,1]]',
                           'generator_called': False,
                           'semantic_check': 'canonical transforms reconstructed independently from frozen 320x240 and 256x192 matrices agree at atol=1e-14'},
            'images_and_homographies': entries,
            'postprocessing_behavior': {'original_superpoint': 'threshold=0.005 after NMS',
                                        'v1_l1': 'no threshold after NMS',
                                        'source': 'frozen R0 manifest; preserved exactly'},
            'topk_values': list(CAPS)}


def verify_extension(data: Dict[str, Any]) -> Dict[str, Any]:
    import math
    assert data['resolution'] == NAME and data['image_count'] == 30 and data['topk_values'] == list(CAPS)
    assert sha256(R0_MANIFEST) == data['base_r0_manifest_sha256']
    assert len(data['images_and_homographies']) == 30
    seen = set()
    for entry in data['images_and_homographies']:
        assert entry['image_index'] not in seen; seen.add(entry['image_index'])
        path = Path('/home/quyet/edge_ai_project') / entry['relative_path']
        assert path.is_file() and sha256(path) == entry['image_sha256']
        for key in ('canonical_normalized_source_to_warped', 'matrix_source_to_warped_xy_pixels_160x120'):
            matrix = entry[key]; assert len(matrix) == 3 and all(len(row) == 3 for row in matrix)
            assert all(math.isfinite(float(item)) for row in matrix for item in row)
    return {'status': 'PASS', 'images': len(seen), 'matrices': len(data['images_and_homographies'])}


def activation_records(model, image):
    import torch
    rows = []
    def add(name, value):
        assert torch.isfinite(value).all(); rows.append({'name': name, 'shape': list(value.shape), 'elements': value.numel(), 'fp32_bytes': value.numel() * 4, 'projected_int8_bytes': value.numel()}); return value
    x = add('input', image); x = add('conv1a', model.conv1a(x)); x = add('relu1a', model.relu(x)); x = add('conv1b', model.conv1b(x)); x = add('relu1b', model.relu(x)); x = add('pool1', model.pool(x))
    x = add('conv2a', model.conv2a(x)); x = add('relu2a', model.relu(x)); x = add('conv2b', model.conv2b(x)); x = add('relu2b', model.relu(x)); x = add('pool2', model.pool(x))
    x = add('conv3a', model.conv3a(x)); x = add('relu3a', model.relu(x)); x = add('conv3b', model.conv3b(x)); x = add('relu3b', model.relu(x)); x = add('pool3', model.pool(x))
    x = add('conv4a', model.conv4a(x)); x = add('relu4a', model.relu(x)); x = add('conv4b', model.conv4b(x)); x = add('relu4b', model.relu(x))
    d = add('convPa', model.convPa(x)); d = add('reluPa', model.relu(d)); add('detector_logits', model.convPb(d))
    q = add('convDa', model.convDa(x)); q = add('reluDa', model.relu(q)); add('descriptor_map', model.convDb(q))
    return rows


def export() -> int:
    import onnx, torch
    sys.path.insert(0, str(ROOT / 'scripts')); import create_superpoint_k210_w0 as w0
    onnx_path, inferred_path, _ = model_paths(); image = torch.linspace(0, 1, WIDTH * HEIGHT).reshape(1, 1, HEIGHT, WIDTH)
    torch.set_num_threads(1); model = w0.V1FR12().eval().cpu(); model.load_state_dict(torch.load(ARTIFACT, map_location='cpu', weights_only=True))
    with torch.inference_mode(): outputs = model(image)
    assert [list(item.shape) for item in outputs] == [[1, 65, 15, 20], [1, 256, 15, 20]]
    records = activation_records(model, image)
    torch.onnx.export(model, image, str(onnx_path), opset_version=13, dynamo=False, do_constant_folding=False, input_names=['image'], output_names=['detector_logits', 'descriptor_map'])
    graph = onnx.load(str(onnx_path)); onnx.checker.check_model(graph); inferred = onnx.shape_inference.infer_shapes(graph, strict_mode=True); onnx.checker.check_model(inferred); onnx.save(inferred, str(inferred_path)); assert inferred.graph.value_info
    ops = dict(Counter(node.op_type for node in inferred.graph.node)); assert set(ops) <= {'Conv', 'Relu', 'MaxPool'}
    save(OUT / 'r2_160_export.json', {'status': 'PASS', 'input_shape': [1, 1, 120, 160], 'detector_shape': [1, 65, 15, 20], 'descriptor_shape': [1, 256, 15, 20], 'operators': ops, 'value_info_count': len(inferred.graph.value_info), 'onnx_sha256': sha256(onnx_path), 'inferred_onnx_sha256': sha256(inferred_path), 'v1_checkpoint_sha256': sha256(ARTIFACT), 'logical_activations': records, 'largest_logical_fp32_activation_bytes': max(item['fp32_bytes'] for item in records), 'largest_projected_int8_activation_bytes': max(item['projected_int8_bytes'] for item in records)})
    print('160x120 forward/export/checker/shape PASS'); return 0


def compile_model() -> int:
    import _nncase, nncase, numpy as np, onnx
    from importlib.metadata import version
    _, inferred_path, kmodel = model_paths(); result = {'checker': 'NOT RUN', 'import': 'NOT RUN', 'ptq': 'NOT RUN', 'compile': 'NOT RUN', 'gencode': 'NOT RUN'}; stage = 'checker'
    try:
        env = {'python_executable': sys.executable, 'nncase': version('nncase'), '_nncase': _nncase.__version__, 'onnx': version('onnx'), 'numpy': version('numpy')}; assert env['nncase'] == '1.8.0.20220929' and env['_nncase'] == '1.8.0-55be52f'; result['environment'] = env
        graph = onnx.load(str(inferred_path)); onnx.checker.check_model(graph); assert graph.graph.value_info; result[stage] = 'PASS'; stage = 'import'
        options = nncase.CompileOptions(); options.target = 'k210'; options.quant_type = 'uint8'; options.w_quant_type = 'uint8'; dump = OUT / 'r2_160_nncase'; dump.mkdir(parents=True, exist_ok=True); options.dump_dir = str(dump); options.dump_ir = True; options.dump_asm = True; options.dump_quant_error = True
        compiler = nncase.Compiler(options); compiler.import_onnx(inferred_path.read_bytes(), nncase.ImportOptions()); result[stage] = 'PASS'; stage = 'ptq'
        calibration = np.random.default_rng(CALIBRATION_SEED).uniform(0, 1, size=(CALIBRATION_COUNT, 1, HEIGHT, WIDTH)).astype(np.float32); result['calibration_shape'] = list(calibration.shape); result['calibration_sha256'] = hashlib.sha256(calibration.tobytes(order='C')).hexdigest()
        ptq = nncase.PTQTensorOptions(); ptq.samples_count = CALIBRATION_COUNT; ptq.set_tensor_data(calibration.tobytes(order='C')); compiler.use_ptq(ptq); result[stage] = 'PASS'; stage = 'compile'
        compiler.compile(); result[stage] = 'PASS'; stage = 'gencode'; blob = compiler.gencode_tobytes(); assert blob; kmodel.write_bytes(blob); result.update(gencode='PASS', size=len(blob), sha256=sha256(kmodel)); print('160x120 K210 compile PASS')
    except Exception:
        result[stage] = 'FAIL'; result['error'] = traceback.format_exc(); traceback.print_exc()
    save(OUT / 'r2_160_compiler.json', result); return 0 if result['gencode'] == 'PASS' else 1


def evaluate() -> int:
    import cv2, numpy as np, torch
    sys.path.insert(0, str(ROOT / 'scripts')); import create_superpoint_k210_w0 as w0; import superpoint_resolution_gate as r0
    data = json.loads(EXTENSION.read_text()); verify_extension(data); torch.set_num_threads(1)
    original, sp = w0.import_original_superpoint(); v1 = w0.V1FR12().eval().cpu(); v1.load_state_dict(torch.load(ARTIFACT, map_location='cpu', weights_only=True))
    detector_rows, homography_rows = [], []
    for entry in data['images_and_homographies']:
        gray = r0.load_gray(Path('/home/quyet/edge_ai_project') / entry['relative_path']); source = r0.resize_tensor(gray, WIDTH, HEIGHT); H = np.asarray(entry['matrix_source_to_warped_xy_pixels_160x120'], dtype=np.float64)
        warped = cv2.warpPerspective(source[0, 0].numpy(), H, (WIDTH, HEIGHT), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101); target = torch.from_numpy(warped).float()[None, None]
        with torch.inference_mode():
            a, aw = original({'image': source}), original({'image': target}); logits, dense = v1(source); wlogits, wdense = v1(target)
            features = {'original': {'keypoints': a['keypoints'][0], 'scores': a['scores'][0], 'descriptors': a['descriptors'][0]}, 'v1_l1': w0.postprocess_raw(logits, dense, sp)}
            warped_features = {'original': {'keypoints': aw['keypoints'][0], 'scores': aw['scores'][0], 'descriptors': aw['descriptors'][0]}, 'v1_l1': w0.postprocess_raw(wlogits, wdense, sp)}
        for cap in CAPS:
            original_cap = r0.cap_features(features['original'], cap)
            for model in ('original', 'v1_l1'):
                current, transformed = r0.cap_features(features[model], cap), r0.cap_features(warped_features[model], cap)
                detector_rows.append({'image': entry['relative_path'], 'image_index': entry['image_index'], 'resolution': NAME, 'model': model, 'cap': cap, **r0.detector_metrics(current, None if model == 'original' else original_cap, WIDTH, HEIGHT)})
                homography_rows.append({'image': entry['relative_path'], 'image_index': entry['image_index'], 'resolution': NAME, 'model': model, 'cap': cap, **r0.homography_metrics(current, transformed, H, WIDTH, HEIGHT)})
    detector_keys = ('score_mean', 'score_median', 'score_std', 'grid_occupancy', 'xy_spread', 'agreement_at3'); homography_keys = ('repeatability_1px', 'repeatability_2px', 'repeatability_3px', 'localization_error_px', 'mnn_count', 'mnn_precision', 'geometric_descriptor_cosine')
    aggregates = {}
    for label, rows, keys in (('detector', detector_rows, detector_keys), ('homography', homography_rows, homography_keys)):
        groups = defaultdict(list)
        for row in rows: groups['|'.join(map(str, (row['resolution'], row['model'], row['cap'])))] .append(row)
        aggregates[label] = {key: {'images': len(value), **r0.mean_metrics(value, keys)} for key, value in groups.items()}
    save(OUT / 'r2_160_metrics.json', {'extension_manifest_sha256': sha256(EXTENSION), 'models': ['original', 'v1_l1'], 'resolution': NAME, 'detector_per_image': detector_rows, 'homography_per_image': homography_rows, 'detector_aggregate': aggregates['detector'], 'homography_aggregate': aggregates['homography']})
    with (OUT / 'r2_160_per_image.csv').open('w', newline='', encoding='utf-8') as handle:
        rows = [{**row, 'category': 'detector'} for row in detector_rows] + [{**row, 'category': 'homography'} for row in homography_rows]; fields = sorted({key for row in rows for key in row}); writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    print('160x120 evaluation PASS: manifest-only image/H inputs'); return 0


def compiler_memory(log: Path) -> str:
    if not log.exists(): return 'NOT PROVIDED'
    text = log.read_text(); start = text.find('MEMORY USAGES')
    return 'NOT PROVIDED' if start < 0 else '\n'.join(line for line in text[start:].splitlines() if not line.endswith('K210 compile PASS')).strip()


def compiler_warnings(log: Path) -> str:
    if not log.exists(): return 'NOT PROVIDED'
    warnings = [line for line in log.read_text().splitlines() if line.startswith('WARN:')]
    return '; '.join(warnings) if warnings else 'none'


def report() -> int:
    r0 = json.loads((OUT / 'r0_metrics.json').read_text()); r1 = json.loads((OUT / 'r1_extension_metrics.json').read_text()); r2 = json.loads((OUT / 'r2_160_metrics.json').read_text()); exp = json.loads((OUT / 'r2_160_export.json').read_text()); comp = json.loads((OUT / 'r2_160_compiler.json').read_text())
    detector = {**{key: value for key, value in r0['detector_aggregate'].items() if '|v1_random|' not in key}, **r1['detector_aggregate'], **r2['detector_aggregate']}; homo = {**{key: value for key, value in r0['homography_aggregate'].items() if '|v1_random|' not in key}, **r1['homography_aggregate'], **r2['homography_aggregate']}
    lines = ['# SP-K210 R2 160x120 extension', '',
             'Only `160x120` was evaluated. `320x240`, `256x192`, `224x168`, and `192x144` are loaded historical aggregates; none were rerun. The immutable R0 manifest was read only. A separate R2 extension manifest derives 160×120 matrices from frozen 320×240 canonical transforms.', '',
             '## Protocol', '', '- Same frozen ordered 30 images, preprocessing, NMS, border policy, descriptor sampling, Top-256/512/1024, metric implementations and checkpoints.', '- Original threshold=0.005; V1-L1 no threshold after NMS, exactly preserving historical per-model behavior.', '- Primary analysis is within-model resolution sensitivity only; no cross-model absolute quality conclusion.', '',
             '## V1 160x120 compiler feasibility', '',
             '- Forward outputs: detector `{}`, descriptor `{}`; ONNX checker/shape inference `{}`; operators `{}`.'.format(exp['detector_shape'], exp['descriptor_shape'], exp['status'], exp['operators']),
             '- nncase checker/import/PTQ/compile/gencode: `{}`.'.format('/'.join(comp.get(key, 'NOT RUN') for key in ('checker', 'import', 'ptq', 'compile', 'gencode'))),
             '- Largest logical activation: FP32 `{}` bytes; projected INT8 `{}` bytes. These are not peak KPU RAM.'.format(exp['largest_logical_fp32_activation_bytes'], exp['largest_projected_int8_activation_bytes']),
             '- kmodel: `{}` bytes; SHA256 `{}`.'.format(comp.get('size'), comp.get('sha256')), '- Compiler memory information:', '```', compiler_memory(LOG / 'compile.log'), '```', '- Compiler warnings: `{}`.'.format(compiler_warnings(LOG / 'compile.log')), '- Compiler PASS is not proof of board runtime/latency/localization/system RAM.', '',
             '## Five-resolution detector/spatial metrics (mean, frozen 30 images)', '', '| Resolution | Model | Cap | Score mean | Score median | Score std | 4x4 coverage | XY spread |', '|---|---|---:|---:|---:|---:|---:|---:|']
    order = ('320x240', '256x192', '224x168', '192x144', NAME)
    for resolution in order:
        for model in ('original', 'v1_l1'):
            for cap in CAPS:
                row = detector['{}|{}|{}'.format(resolution, model, cap)]; lines.append('| {} | {} | {} | {:.5f} | {:.5f} | {:.5f} | {:.3f} | {:.3f} |'.format(resolution, model, cap, row['score_mean'], row['score_median'], row['score_std'], row['grid_occupancy'], row['xy_spread']))
    lines += ['', '## Five-resolution homography metrics (mean, frozen 30 images)', '', '| Resolution | Model | Cap | Repeat @1 | @2 | @3 | Localization error px | MNN count | MNN precision @3 | Descriptor cosine |', '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for resolution in order:
        for model in ('original', 'v1_l1'):
            for cap in CAPS:
                row = homo['{}|{}|{}'.format(resolution, model, cap)]; lines.append('| {} | {} | {} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.1f} | {:.3f} | {:.3f} |'.format(resolution, model, cap, row['repeatability_1px'], row['repeatability_2px'], row['repeatability_3px'], row['localization_error_px'], row['mnn_count'], row['mnn_precision'], row['geometric_descriptor_cosine']))
    deltas, knee = [], False
    for model in ('original', 'v1_l1'):
        for cap in CAPS:
            before, after = homo['192x144|{}|{}'.format(model, cap)], homo['160x120|{}|{}'.format(model, cap)]
            dr, dp, dm = after['repeatability_3px'] - before['repeatability_3px'], after['mnn_precision'] - before['mnn_precision'], after['mnn_count'] - before['mnn_count']; knee |= dr < -0.10 or dp < -0.10; deltas.append((model, cap, dr, dp, dm))
    recommendation = '192x144' if knee else '160x120'; decision = '160x120 REJECTED — KEEP 192x144' if knee else '160x120 ACCEPTED'
    lines += ['', '## 192x144 → 160x120 knee check', '', 'Locked rule: a drop >0.10 absolute in repeatability@3 or MNN precision@3 at any cap for either model is a clear knee. MNN count is reported separately for later correspondence/PnP robustness.', '', '| Model | Cap | Delta repeatability@3 | Delta MNN precision@3 | Delta MNN count |', '|---|---:|---:|---:|---:|']
    for model, cap, dr, dp, dm in deltas: lines.append('| {} | {} | {:+.3f} | {:+.3f} | {:+.1f} |'.format(model, cap, dr, dp, dm))
    lines += ['', decision, '', '## Files', '', '- `r2_160_protocol_extension.json`, `r2_160_metrics.json`, and `r2_160_per_image.csv`.', '- `scripts/superpoint_resolution_r2_160.py`; static ONNX/kmodel/compiler logs.', '', 'FINAL RESOLUTION RECOMMENDATION: {}'.format(recommendation)]
    (OUT / 'r2_160_extension.md').write_text('\n'.join(lines) + '\n'); print('FINAL RESOLUTION RECOMMENDATION: {}'.format(recommendation)); return 0


def logged(command, name):
    LOG.mkdir(parents=True, exist_ok=True)
    with (LOG / name).open('w') as handle: return subprocess.run(command, cwd=str(ROOT), stdout=handle, stderr=subprocess.STDOUT).returncode


def run() -> int:
    data = extension_manifest(); result = verify_extension(data); save(EXTENSION, data); print('R2 extension manifest {}'.format(result['status']))
    for command, name in (([sys.executable, '-u', __file__, 'export'], 'export.log'), ([KPY, '-u', __file__, 'compile'], 'compile.log'), ([sys.executable, '-u', __file__, 'evaluate'], 'evaluate.log')):
        if logged(command, name): raise RuntimeError('{} failed'.format(name))
    return report()


def main():
    action = argparse.ArgumentParser(); action.add_argument('action', choices=('run', 'export', 'compile', 'evaluate', 'report')); args = action.parse_args()
    return {'run': run, 'export': export, 'compile': compile_model, 'evaluate': evaluate, 'report': report}[args.action]()


if __name__ == '__main__':
    try: raise SystemExit(main())
    except Exception as error:
        if len(sys.argv) > 1 and sys.argv[1] == 'run': (OUT / 'r2_160_extension.md').write_text('# R2 160x120\n\nFailure: `{}`\n'.format(error))
        raise
