#!/usr/bin/env python3
"""SP-K210-R0: resolution freeze gate; no training or model redesign."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import subprocess
import sys
import traceback
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
EDGE = Path('/home/quyet/edge_ai_project')
OUT = ROOT / 'reports/superpoint_resolution_gate'
LOG = OUT / 'logs'
ARTIFACT = ROOT / 'models/superpoint_k210_v1_fr12_l1init.pth'
ONNX = ROOT / 'models/superpoint_k210_v1_fr12_l1init_256x192.onnx'
INFERRED = ROOT / 'models/superpoint_k210_v1_fr12_l1init_256x192_inferred.onnx'
KMODEL = ROOT / 'artifacts/superpoint_k210_v1_fr12_l1init_256x192.kmodel'
KPY = '/home/quyet/miniconda3/envs/k210/bin/python'
MODEL_SEED = 20260905
CALIBRATION_SEED, CALIBRATION_COUNT = 20260906, 32
RESOLUTIONS = ((320, 240), (256, 192))  # width, height
CAPS = (256, 512, 1024)
BASELINE_CONFIG = dict(nms_radius=3, keypoint_threshold=0.005, max_keypoints=3072,
                       remove_borders=4, fix_sampling=False)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')


def export_256() -> int:
    import torch
    import onnx
    sys.path.insert(0, str(ROOT / 'scripts'))
    import create_superpoint_k210_w0 as w0
    torch.set_num_threads(1)
    model = w0.V1FR12().eval().cpu()
    model.load_state_dict(torch.load(ARTIFACT, map_location='cpu', weights_only=True))
    image = torch.linspace(0, 1, 192 * 256).reshape(1, 1, 192, 256)
    with torch.inference_mode():
        outputs = model(image)
    assert [list(t.shape) for t in outputs] == [[1, 65, 24, 32], [1, 256, 24, 32]]
    torch.onnx.export(model, image, str(ONNX), opset_version=13, dynamo=False,
                      do_constant_folding=False, input_names=['image'],
                      output_names=['detector_logits', 'descriptor_map'])
    graph = onnx.load(str(ONNX))
    onnx.checker.check_model(graph)
    inferred = onnx.shape_inference.infer_shapes(graph, strict_mode=True)
    onnx.checker.check_model(inferred)
    onnx.save(inferred, str(INFERRED))
    assert inferred.graph.value_info
    histogram = dict(Counter(node.op_type for node in inferred.graph.node))
    assert set(histogram) <= {'Conv', 'Relu', 'MaxPool'}, histogram
    specs = {}
    for value in list(inferred.graph.input) + list(inferred.graph.value_info) + list(inferred.graph.output):
        dims = value.type.tensor_type.shape.dim
        shape = [dim.dim_value for dim in dims]
        assert all(shape) and not any(dim.dim_param for dim in dims)
        specs[value.name] = shape
    assert specs['image'] == [1, 1, 192, 256]
    assert specs['detector_logits'] == [1, 65, 24, 32]
    assert specs['descriptor_map'] == [1, 256, 24, 32]
    save(OUT / 'onnx_export.json', {
        'status': 'PASS', 'input_shape': [1, 1, 192, 256],
        'output_shapes': {'detector_logits': [1, 65, 24, 32], 'descriptor_map': [1, 256, 24, 32]},
        'parameters': sum(p.numel() for p in model.parameters()), 'operators': histogram,
        'value_info_count': len(inferred.graph.value_info), 'onnx_sha256': digest(ONNX),
        'inferred_onnx_sha256': digest(INFERRED), 'artifact_sha256': digest(ARTIFACT),
    })
    print('256x192 ONNX export/checker/inference PASS', histogram)
    return 0


def compiler_environment() -> Dict[str, str]:
    from importlib.metadata import PackageNotFoundError, version
    result = {'executable': sys.executable, 'python': platform.python_version()}
    for package in ('onnx', 'numpy', 'nncase'):
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = 'NOT INSTALLED'
    import _nncase
    result['_nncase'] = _nncase.__version__
    return result


def compile_256() -> int:
    import nncase
    import numpy as np
    import onnx
    result = {'checker': 'NOT RUN', 'import': 'NOT RUN', 'ptq': 'NOT RUN',
              'compile': 'NOT RUN', 'gencode': 'NOT RUN'}
    stage = 'checker'
    try:
        env = compiler_environment()
        assert env['nncase'] == '1.8.0.20220929', env
        assert env['_nncase'] == '1.8.0-55be52f', env
        result['environment'] = env
        graph = onnx.load(str(INFERRED))
        onnx.checker.check_model(graph)
        assert graph.graph.value_info
        result[stage] = 'PASS'
        stage = 'import'
        options = nncase.CompileOptions()
        options.target = 'k210'
        options.quant_type = 'uint8'
        options.w_quant_type = 'uint8'
        dump = OUT / 'nncase_256x192'
        dump.mkdir(parents=True, exist_ok=True)
        options.dump_dir = str(dump)
        options.dump_ir = True
        options.dump_asm = True
        options.dump_quant_error = True
        compiler = nncase.Compiler(options)
        compiler.import_onnx(INFERRED.read_bytes(), nncase.ImportOptions())
        result[stage] = 'PASS'
        stage = 'ptq'
        calibration = np.random.default_rng(CALIBRATION_SEED).uniform(
            0, 1, size=(CALIBRATION_COUNT, 1, 192, 256)).astype(np.float32)
        result['calibration_shape'] = list(calibration.shape)
        result['calibration_sha256'] = hashlib.sha256(calibration.tobytes(order='C')).hexdigest()
        ptq = nncase.PTQTensorOptions()
        ptq.samples_count = CALIBRATION_COUNT
        ptq.set_tensor_data(calibration.tobytes(order='C'))
        compiler.use_ptq(ptq)
        result[stage] = 'PASS'
        stage = 'compile'
        compiler.compile()
        result[stage] = 'PASS'
        stage = 'gencode'
        blob = compiler.gencode_tobytes()
        assert blob, 'Empty kmodel'
        KMODEL.write_bytes(blob)
        result.update(gencode='PASS', size=len(blob), sha256=digest(KMODEL))
        print('256x192 K210 compile/gencode PASS', result['size'], result['sha256'])
    except Exception:
        result[stage] = 'FAIL'
        result['error'] = traceback.format_exc()
        traceback.print_exc()
    save(OUT / 'compiler_256x192.json', result)
    return 0 if result['gencode'] == 'PASS' else 1


def select_images() -> List[Path]:
    root = EDGE / 'assets/map_prepared_v2'
    groups = [sorted(directory.glob('*.jpg')) for directory in sorted(root.iterdir()) if directory.is_dir()]
    groups = [images for images in groups if images]
    if not groups:
        raise RuntimeError('No map_prepared_v2 images found')
    chosen = [images[len(images) // 2] for images in groups]
    positions = [0] * len(groups)
    while len(chosen) < 30:
        added = False
        for index, images in enumerate(groups):
            while positions[index] == len(images) // 2:
                positions[index] += 1
            if positions[index] < len(images):
                chosen.append(images[positions[index]])
                positions[index] += 1
                added = True
                if len(chosen) == 30:
                    break
        if not added:
            raise RuntimeError('Fewer than 30 project images')
    return chosen[:30]


def load_gray(path: Path):
    import cv2
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError('Cannot read {}'.format(path))
    return image


def resize_tensor(gray, width: int, height: int):
    import cv2
    import torch
    resized = cv2.resize(gray, (width, height), interpolation=cv2.INTER_AREA)
    return torch.from_numpy(resized).float().div(255.0)[None, None]


def cap_features(features: Dict, cap: int) -> Dict:
    import torch
    count = min(cap, int(features['scores'].numel()))
    order = torch.argsort(features['scores'], descending=True, stable=True)[:count]
    return {key: (value[:, order] if key == 'descriptors' else value[order])
            for key, value in features.items() if key in ('keypoints', 'scores', 'descriptors')}


def detector_metrics(features: Dict, original: Dict | None, width: int, height: int) -> Dict[str, float]:
    import torch
    keys, scores = features['keypoints'], features['scores']
    if not len(keys):
        return {key: 0.0 for key in ('score_mean', 'score_median', 'score_std', 'grid_occupancy',
                                    'xy_spread', 'agreement_at3')}
    # Fixed 4x4 occupancy after normalizing by the known input-image extent.
    xs = torch.clamp((keys[:, 0] / width * 4).long(), 0, 3)
    ys = torch.clamp((keys[:, 1] / height * 4).long(), 0, 3)
    occupied = torch.unique(ys * 4 + xs).numel() / 16.0
    spread = torch.sqrt((keys[:, 0].std(unbiased=False) / width) ** 2 +
                        (keys[:, 1].std(unbiased=False) / height) ** 2).item()
    agreement = 1.0
    if original is not None and len(original['keypoints']):
        nearest = torch.cdist(original['keypoints'], keys).min(dim=1).values
        agreement = float((nearest <= 3.0).float().mean())
    return {'score_mean': float(scores.mean()), 'score_median': float(scores.median()),
            'score_std': float(scores.std(unbiased=False)), 'grid_occupancy': float(occupied),
            'xy_spread': spread, 'agreement_at3': agreement}


def normalized_homography(index: int, width: int, height: int):
    """A fixed moderate projective perturbation in normalized coordinates."""
    import numpy as np
    theta = math.radians((-3 + index % 7) * 1.15)
    scale = 1.0 + ((index % 5) - 2) * 0.008
    c, s = math.cos(theta) * scale, math.sin(theta) * scale
    transform = np.array([[c, -s, 0.5 - 0.5 * c + 0.5 * s],
                          [s, c, 0.5 - 0.5 * s - 0.5 * c],
                          [((index % 3) - 1) * 0.006, ((index % 4) - 1.5) * 0.004, 1.0]], dtype=np.float64)
    transform[0, 2] += ((index % 5) - 2) * 0.012
    transform[1, 2] += ((index % 6) - 2.5) * 0.010
    scale_pixels = np.array([[width - 1, 0, 0], [0, height - 1, 0], [0, 0, 1]], dtype=np.float64)
    return scale_pixels @ transform @ np.linalg.inv(scale_pixels)


def homography_metrics(source: Dict, transformed: Dict, homography, width: int, height: int) -> Dict[str, float]:
    import numpy as np
    import torch
    source_keys, target_keys = source['keypoints'], transformed['keypoints']
    empty = {'repeatability_1px': 0.0, 'repeatability_2px': 0.0, 'repeatability_3px': 0.0,
             'localization_error_px': 0.0, 'mnn_count': 0.0, 'mnn_precision': 0.0,
             'geometric_descriptor_cosine': 0.0}
    if not len(source_keys) or not len(target_keys):
        return empty
    xy1 = np.concatenate([source_keys.numpy(), np.ones((len(source_keys), 1))], axis=1)
    projected = xy1 @ homography.T
    projected = projected[:, :2] / projected[:, 2:3]
    projected = torch.from_numpy(projected).float()
    distances = torch.cdist(projected, target_keys)
    nearest_distance, nearest_target = distances.min(dim=1)
    # Points mapped beyond the destination frame cannot repeat; exclude them from denominator.
    valid = ((projected[:, 0] >= 0) & (projected[:, 1] >= 0) &
             (projected[:, 0] < width) & (projected[:, 1] < height))
    if not valid.any():
        return empty
    repeat = {tolerance: float(((nearest_distance <= tolerance) & valid).float().sum() / valid.float().sum())
              for tolerance in (1.0, 2.0, 3.0)}
    repeated = (nearest_distance <= 3.0) & valid
    localization = float(nearest_distance[repeated].mean()) if repeated.any() else 0.0
    source_desc = torch.nn.functional.normalize(source['descriptors'].T, p=2, dim=1)
    target_desc = torch.nn.functional.normalize(transformed['descriptors'].T, p=2, dim=1)
    similarity = source_desc @ target_desc.T
    forward = similarity.argmax(dim=1)
    backward = similarity.argmax(dim=0)
    source_indices = torch.arange(len(source_desc))
    mutual = backward[forward] == source_indices
    matched_distances = distances[source_indices, forward]
    match_count = int(mutual.sum())
    precision = float(((matched_distances <= 3.0) & mutual & valid).float().sum() / max(match_count, 1))
    cosine = (source_desc[repeated] * target_desc[nearest_target[repeated]]).sum(dim=1)
    return {'repeatability_1px': repeat[1.0], 'repeatability_2px': repeat[2.0],
            'repeatability_3px': repeat[3.0], 'localization_error_px': localization,
            'mnn_count': float(match_count), 'mnn_precision': precision,
            'geometric_descriptor_cosine': float(cosine.mean()) if cosine.numel() else 0.0}


def mean_metrics(rows: List[Dict], keys: Tuple[str, ...]) -> Dict[str, float]:
    import numpy as np
    return {key: float(np.mean([row[key] for row in rows])) for key in keys}


def evaluate() -> int:
    import cv2
    import torch
    torch.set_num_threads(1)
    sys.path.insert(0, str(ROOT / 'scripts'))
    import create_superpoint_k210_w0 as w0
    original, sp_module = w0.import_original_superpoint()
    structured = w0.V1FR12().eval().cpu()
    structured.load_state_dict(torch.load(ARTIFACT, map_location='cpu', weights_only=True))
    torch.manual_seed(MODEL_SEED)
    random_model = w0.V1FR12().eval().cpu()
    models = [('original', original), ('v1_l1', structured), ('v1_random', random_model)]
    detector_rows, homography_rows = [], []
    for image_index, path in enumerate(select_images()):
        gray = load_gray(path)
        for width, height in RESOLUTIONS:
            source = resize_tensor(gray, width, height)
            H = normalized_homography(image_index, width, height)
            warped = cv2.warpPerspective(source[0, 0].numpy(), H, (width, height),
                                         flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
            target = torch.from_numpy(warped).float()[None, None]
            outputs = {}
            transformed_outputs = {}
            with torch.inference_mode():
                for name, model in models:
                    if name == 'original':
                        raw = model({'image': source})
                        raw_warped = model({'image': target})
                        outputs[name] = {'keypoints': raw['keypoints'][0], 'scores': raw['scores'][0],
                                         'descriptors': raw['descriptors'][0]}
                        transformed_outputs[name] = {'keypoints': raw_warped['keypoints'][0],
                                                     'scores': raw_warped['scores'][0],
                                                     'descriptors': raw_warped['descriptors'][0]}
                    else:
                        logits, dense = model(source)
                        warped_logits, warped_dense = model(target)
                        outputs[name] = w0.postprocess_raw(logits, dense, sp_module)
                        transformed_outputs[name] = w0.postprocess_raw(warped_logits, warped_dense, sp_module)
            for cap in CAPS:
                original_cap = cap_features(outputs['original'], cap)
                for name, _ in models:
                    current = cap_features(outputs[name], cap)
                    warped_current = cap_features(transformed_outputs[name], cap)
                    detector_rows.append({
                        'image': str(path.relative_to(EDGE)), 'resolution': '{}x{}'.format(width, height),
                        'model': name, 'cap': cap,
                        **detector_metrics(current, None if name == 'original' else original_cap, width, height)})
                    homography_rows.append({
                        'image': str(path.relative_to(EDGE)), 'resolution': '{}x{}'.format(width, height),
                        'model': name, 'cap': cap,
                        **homography_metrics(current, warped_current, H, width, height)})
    detector_keys = ('score_mean', 'score_median', 'score_std', 'grid_occupancy', 'xy_spread', 'agreement_at3')
    homography_keys = ('repeatability_1px', 'repeatability_2px', 'repeatability_3px',
                       'localization_error_px', 'mnn_count', 'mnn_precision', 'geometric_descriptor_cosine')
    grouped_detector, grouped_homography = {}, {}
    for rows, target_group, keys in ((detector_rows, grouped_detector, detector_keys),
                                     (homography_rows, grouped_homography, homography_keys)):
        buckets = defaultdict(list)
        for row in rows:
            buckets[(row['resolution'], row['model'], row['cap'])].append(row)
        for key, bucket in buckets.items():
            target_group['|'.join(map(str, key))] = {'images': len(bucket), **mean_metrics(bucket, keys)}
    save(OUT / 'r0_metrics.json', {'images': [str(path.relative_to(EDGE)) for path in select_images()],
                                   'baseline_config': BASELINE_CONFIG, 'caps': list(CAPS),
                                   'detector_per_image': detector_rows, 'homography_per_image': homography_rows,
                                   'detector_aggregate': grouped_detector,
                                   'homography_aggregate': grouped_homography})
    with (OUT / 'r0_per_image.csv').open('w', newline='', encoding='utf-8') as handle:
        rows = [{**row, 'category': 'detector'} for row in detector_rows] + [{**row, 'category': 'homography'} for row in homography_rows]
        fields = sorted({key for row in rows for key in row})
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    print('R0 PC evaluation PASS: {} images, {} detector rows, {} homography rows'.format(
        len(select_images()), len(detector_rows), len(homography_rows)))
    return 0


def metric(data: Dict, section: str, resolution: str, model: str, cap: int, name: str) -> float:
    return data[section]['{}|{}|{}'.format(resolution, model, cap)][name]


def report() -> int:
    export = json.loads((OUT / 'onnx_export.json').read_text())
    compiler = json.loads((OUT / 'compiler_256x192.json').read_text())
    data = json.loads((OUT / 'r0_metrics.json').read_text())
    compiler_log = (LOG / 'compile.log').read_text(encoding='utf-8') if (LOG / 'compile.log').exists() else ''
    warning_lines = [line for line in compiler_log.splitlines() if line.startswith('WARN:')]
    memory_start = compiler_log.find('MEMORY USAGES')
    memory_lines = [] if memory_start < 0 else compiler_log[memory_start:].split('256x192 K210 compile/gencode PASS')[0].strip().splitlines()
    lines = ['# SP-K210-R0 deployment-resolution gate', '',
             'This gate freezes a candidate training/deployment resolution. It performs no training, fine-tuning, distillation, pruning, QAT, architecture change, descriptor-dimension change, source-project modification or hardware runtime benchmark.', '',
             '## 256×192 compiler feasibility', '',
             '- Static input: `[1,1,192,256]`; outputs: detector `[1,65,24,32]`, descriptor `[1,256,24,32]`.',
             '- ONNX opset 13/checker/shape inference: `{}`; operators: `{}`.'.format(export.get('status', 'FAIL'), export.get('operators')),
             '- nncase checker/import/PTQ/compile/gencode: `{}/{}/{}/{}/{}`.'.format(*[compiler.get(key, 'NOT RUN') for key in ('checker', 'import', 'ptq', 'compile', 'gencode')]),
             '- kmodel: `{}` bytes, SHA256 `{}`.'.format(compiler.get('size', 'N/A'), compiler.get('sha256', 'N/A')),
             '- Compiler environment: `{}`.'.format(compiler.get('environment', {})),
             '- Compiler-reported memory information (verbatim):', '```', '\n'.join(memory_lines) or 'NOT PROVIDED', '```',
             '- Compiler warnings: `{}`.'.format('; '.join(warning_lines) if warning_lines else 'none'),
             '- Compiler PASS proves only this static graph compiled; it does **not** prove board runtime, latency, camera integration, localization accuracy or full-system RAM fit.', '',
             '## Evaluation protocol', '',
             '- Same deterministic 30 real `assets/map_prepared_v2` images at both resolutions; each image was grayscale-resized and evaluated FP32 in `[0,1]`.',
             '- Models: original pretrained SuperPoint; W0 V1-FR12 L1 structured-init; fixed-seed V1-FR12 random sanity control.',
             '- Shared semantics: NMS radius 3, detector threshold 0.005, border removal 4, descriptor sampling/normalization from vendored SuperPoint with `fix_sampling=False`.',
             '- Metrics use post-NMS caps Top-256, Top-512 and Top-1024. Random is a lower-control only, not a resolution-deciding model.',
             '- Homographies are deterministic moderate normalized-coordinate rotation/scale/translation/projective perturbations; the same normalized transform per source image is converted to each resolution pixel grid.', '',
             '## Detector and clean-image agreement (mean over 30 images)', '',
             '| Resolution | Model | Cap | Score mean | Score median | Score std | 4×4 occupied bins | XY spread | Agreement with Original ≤3 px |',
             '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for resolution in ('320x240', '256x192'):
        for model in ('original', 'v1_l1', 'v1_random'):
            for cap in CAPS:
                d = data['detector_aggregate']['{}|{}|{}'.format(resolution, model, cap)]
                lines.append('| {} | {} | {} | {:.5f} | {:.5f} | {:.5f} | {:.3f} | {:.3f} | {:.3f} |'.format(
                    resolution, model, cap, d['score_mean'], d['score_median'], d['score_std'],
                    d['grid_occupancy'], d['xy_spread'], d['agreement_at3']))
    lines += ['', '## Synthetic-homography metrics (mean over 30 images)', '',
              '| Resolution | Model | Cap | Repeat @1px | @2px | @3px | Repeat localization error px | MNN matches | MNN precision @3px | Geometric descriptor cosine |',
              '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for resolution in ('320x240', '256x192'):
        for model in ('original', 'v1_l1', 'v1_random'):
            for cap in CAPS:
                h = data['homography_aggregate']['{}|{}|{}'.format(resolution, model, cap)]
                lines.append('| {} | {} | {} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.1f} | {:.3f} | {:.3f} |'.format(
                    resolution, model, cap, h['repeatability_1px'], h['repeatability_2px'],
                    h['repeatability_3px'], h['localization_error_px'], h['mnn_count'],
                    h['mnn_precision'], h['geometric_descriptor_cosine']))
    # Resolution decision: conservative and explicit; original plus L1 must avoid >10pp R@3 loss.
    comparisons = []
    substantial = False
    for model in ('original', 'v1_l1'):
        for cap in CAPS:
            r320 = metric(data, 'homography_aggregate', '320x240', model, cap, 'repeatability_3px')
            r256 = metric(data, 'homography_aggregate', '256x192', model, cap, 'repeatability_3px')
            p320 = metric(data, 'homography_aggregate', '320x240', model, cap, 'mnn_precision')
            p256 = metric(data, 'homography_aggregate', '256x192', model, cap, 'mnn_precision')
            delta_r, delta_p = r256 - r320, p256 - p320
            comparisons.append((model, cap, delta_r, delta_p))
            substantial |= delta_r < -0.10 or delta_p < -0.10
    recommendation = '320x240' if substantial else '256x192'
    lines += ['', '## Resolution comparison and decision', '',
              'Decision rule fixed before reading results: label degradation **substantial** if either original or V1-L1 loses more than 0.10 absolute in homography repeatability@3px or MNN precision@3px at any required cap. Otherwise it is modest and 256×192 is recommended. Random is excluded from the decision.', '',
              '| Model | Cap | Δ repeatability@3px (256−320) | Δ MNN precision@3px (256−320) |', '|---|---:|---:|---:|']
    for model, cap, delta_r, delta_p in comparisons:
        lines.append('| {} | {} | {:+.3f} | {:+.3f} |'.format(model, cap, delta_r, delta_p))
    lines += ['', '## Same-resolution architecture/weight-transfer comparison', '',
              '| Resolution | Cap | V1-L1 detector agreement with Original ≤3px | Δ repeatability@3px (V1-L1−Original) | Δ MNN precision@3px (V1-L1−Original) |',
              '|---|---:|---:|---:|---:|']
    for resolution in ('320x240', '256x192'):
        for cap in CAPS:
            agreement = metric(data, 'detector_aggregate', resolution, 'v1_l1', cap, 'agreement_at3')
            repeat_delta = (metric(data, 'homography_aggregate', resolution, 'v1_l1', cap, 'repeatability_3px') -
                            metric(data, 'homography_aggregate', resolution, 'original', cap, 'repeatability_3px'))
            precision_delta = (metric(data, 'homography_aggregate', resolution, 'v1_l1', cap, 'mnn_precision') -
                               metric(data, 'homography_aggregate', resolution, 'original', cap, 'mnn_precision'))
            lines.append('| {} | {} | {:.3f} | {:+.3f} | {:+.3f} |'.format(
                resolution, cap, agreement, repeat_delta, precision_delta))
    lines += ['', 'This architecture/transfer loss must not be attributed to resolution alone. It is expected to be addressed only in a later authorized training/quality gate.',
              '', '## Scope limits', '',
              'This is not a localization/PnP/retrieval evaluation. It does not demonstrate K210 board performance. It only freezes the input resolution before a later authorized training gate.', '',
              '## Files created/modified', '',
              '- `scripts/superpoint_resolution_gate.py`.',
              '- `models/superpoint_k210_v1_fr12_l1init_256x192.onnx` and inferred ONNX.',
              '- `artifacts/superpoint_k210_v1_fr12_l1init_256x192.kmodel` when compiler PASS.',
              '- `reports/superpoint_resolution_gate/` (logs, compiler/export facts, per-image and aggregate metrics).',
              '', 'SP-K210-R0: {} RECOMMENDED'.format(recommendation)]
    (OUT / 'r0_report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('SP-K210-R0: {} RECOMMENDED'.format(recommendation))
    return 0


def run_logged(command: List[str], log_name: str) -> int:
    LOG.mkdir(parents=True, exist_ok=True)
    with (LOG / log_name).open('w', encoding='utf-8') as handle:
        return subprocess.run(command, cwd=str(ROOT), stdout=handle, stderr=subprocess.STDOUT).returncode


def run() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    steps = [([sys.executable, '-u', __file__, 'export'], 'export.log'),
             ([KPY, '-u', __file__, 'compile'], 'compile.log'),
             ([sys.executable, '-u', __file__, 'evaluate'], 'evaluate.log')]
    for command, log in steps:
        if run_logged(command, log):
            raise RuntimeError('{} failed; see {}'.format(command[-1], OUT / 'logs' / log))
    return report()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('run', 'export', 'compile', 'evaluate', 'report'))
    action = parser.parse_args().action
    if action == 'run': return run()
    if action == 'export': return export_256()
    if action == 'compile': return compile_256()
    if action == 'evaluate': return evaluate()
    return report()


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as error:
        if len(sys.argv) > 1 and sys.argv[1] == 'run':
            OUT.mkdir(parents=True, exist_ok=True)
            (OUT / 'r0_report.md').write_text('# SP-K210-R0\n\nFailure: `{}`\n'.format(error), encoding='utf-8')
        raise
