#!/usr/bin/env python3
"""Freeze and validate the non-inference protocol used by SP-K210-R0."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import platform
import sys
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[1]
EDGE = Path('/home/quyet/edge_ai_project')
OUT = ROOT / 'reports/superpoint_resolution_gate'
MANIFEST = OUT / 'r0_protocol_manifest.json'
AUDIT = OUT / 'r0_protocol_audit.md'
R0_SCRIPT = ROOT / 'scripts/superpoint_resolution_gate.py'
R0_METRICS = OUT / 'r0_metrics.json'
R0_CSV = OUT / 'r0_per_image.csv'
R0_REPORT = OUT / 'r0_report.md'
ORIGINAL_SOURCE = EDGE / 'third_party/SuperGluePretrainedNetwork/models/superpoint.py'
ORIGINAL_CHECKPOINT = EDGE / 'third_party/SuperGluePretrainedNetwork/models/weights/superpoint_v1.pth'
V1_CHECKPOINT = ROOT / 'models/superpoint_k210_v1_fr12_l1init.pth'
W0_CHANNEL_MAP = ROOT / 'reports/superpoint_weight_transfer/channel_map.json'
RESOLUTIONS = ((320, 240), (256, 192))
FUTURE_RESOLUTIONS = ((224, 168), (192, 144))
ALL_FROZEN_RESOLUTIONS = RESOLUTIONS + FUTURE_RESOLUTIONS


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_r0():
    spec = importlib.util.spec_from_file_location('r0_protocol_source', R0_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def environment() -> Dict[str, Any]:
    import cv2
    import numpy
    import onnx
    import torch
    try:
        import onnxruntime as ort
        ort_info: Any = {'version': ort.__version__, 'available_providers': ort.get_available_providers()}
    except ImportError:
        ort_info = 'NOT INSTALLED'
    return {
        'python_executable': sys.executable,
        'python_version': platform.python_version(),
        'platform': platform.platform(),
        'pytorch_version': torch.__version__,
        'torch_cuda_version': torch.version.cuda,
        'torch_cuda_available': torch.cuda.is_available(),
        'numpy_version': numpy.__version__,
        'opencv_version': cv2.__version__,
        'onnx_version': onnx.__version__,
        'onnxruntime': ort_info,
        'quality_evaluation_provider': 'PyTorch CPU; torch.cuda.is_available() was false. ONNX/ORT were not used for R0 quality inference.',
    }


def matrix_list(matrix) -> list:
    return [[float(value) for value in row] for row in matrix.tolist()]


def assert_matrix(matrix: Any) -> None:
    assert isinstance(matrix, list) and len(matrix) == 3
    assert all(isinstance(row, list) and len(row) == 3 for row in matrix)
    assert all(math.isfinite(float(value)) for row in matrix for value in row)
    assert abs(float(matrix[2][2])) > 0.0


def image_records(metrics: Dict[str, Any], r0) -> list:
    import cv2
    ordered = metrics['images']
    selected_now = [str(path.relative_to(EDGE)) for path in r0.select_images()]
    assert ordered == selected_now, 'r0_metrics ordered image list differs from current R0 selector'
    assert len(ordered) == 30 and len(set(ordered)) == 30
    records = []
    for index, relative in enumerate(ordered):
        path = (EDGE / relative).resolve()
        assert path.is_file() and str(path).startswith(str(EDGE.resolve()) + '/')
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        assert image is not None
        records.append({
            'index': index,
            'absolute_path': str(path),
            'relative_path': relative,
            'filename': path.name,
            'file_size_bytes': path.stat().st_size,
            'sha256': sha256(path),
            'source_image_shape': list(image.shape),
        })
    return records


def csv_audit(ordered: list) -> Dict[str, Any]:
    with R0_CSV.open(newline='', encoding='utf-8') as handle:
        rows = list(csv.DictReader(handle))
    csv_images = {row['image'] for row in rows}
    counts = {}
    for row in rows:
        counts[row['image']] = counts.get(row['image'], 0) + 1
    assert csv_images == set(ordered)
    assert all(counts[image] == 36 for image in ordered), counts
    return {
        'path': str(R0_CSV), 'sha256': sha256(R0_CSV), 'data_rows': len(rows),
        'unique_relative_paths': len(csv_images), 'rows_per_image': sorted(set(counts.values())),
        'unambiguous_relative_path_mapping': True,
    }


def homographies(images: list, r0) -> list:
    entries = []
    for image in images:
        matrices = {}
        for width, height in ALL_FROZEN_RESOLUTIONS:
            name = '{}x{}'.format(width, height)
            matrices[name] = matrix_list(r0.normalized_homography(image['index'], width, height))
        entries.append({'image_index': image['index'], 'relative_path': image['relative_path'],
                        'matrices_source_to_warped_xy_pixels': matrices})
    return entries


def manifest_data() -> Dict[str, Any]:
    metrics = json.loads(R0_METRICS.read_text(encoding='utf-8'))
    r0 = load_r0()
    images = image_records(metrics, r0)
    h = homographies(images, r0)
    assert metrics['caps'] == [256, 512, 1024]
    assert metrics['baseline_config'] == r0.BASELINE_CONFIG
    return {
        'protocol_version': 'SP-K210-R0-frozen-v1',
        'purpose': 'Frozen non-inference protocol for fair future SuperPoint resolution experiments.',
        'r0_artifact_provenance': {
            'r0_metrics_path': str(R0_METRICS), 'r0_metrics_sha256': sha256(R0_METRICS),
            'r0_per_image_csv_path': str(R0_CSV), 'r0_per_image_csv_sha256': sha256(R0_CSV),
            'r0_report_path': str(R0_REPORT), 'r0_report_sha256': sha256(R0_REPORT),
            'r0_metrics_models': sorted({row['model'] for row in metrics['detector_per_image']}),
            'r0_metrics_resolutions': sorted({row['resolution'] for row in metrics['detector_per_image']}),
        },
        'ordered_images': images,
        'image_count': len(images),
        'image_hashes': {image['relative_path']: image['sha256'] for image in images},
        'csv_mapping_audit': csv_audit(metrics['images']),
        'homography_seed': None,
        'homography_random_generator': None,
        'homography_assignment': {
            'count_per_image_per_resolution': 1,
            'assignment': 'image list index 0..29; one deterministic normalized transform per index, converted to each resolution pixel grid',
            'coordinate_convention': '3x3 source-resized-image (x,y,1) to warped-resized-image (x,y,1), dehomogenized after multiplication',
            'warp': 'cv2.warpPerspective(..., flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)',
        },
        'homography_generation': {
            'implementation': 'scripts/superpoint_resolution_gate.py:normalized_homography',
            'randomness': 'none; pure index-modulo arithmetic using math and NumPy float64',
            'theta_degrees': '(-3 + index % 7) * 1.15; values -3.45..+3.45 degrees',
            'scale': '1 + ((index % 5) - 2) * 0.008; values 0.984..1.016',
            'translation_x_normalized': '((index % 5) - 2) * 0.012; values -0.024..+0.024',
            'translation_y_normalized': '((index % 6) - 2.5) * 0.010; values -0.025..+0.025',
            'projective_x': '((index % 3) - 1) * 0.006; values -0.006, 0, +0.006',
            'projective_y': '((index % 4) - 1.5) * 0.004; values -0.006, -0.002, +0.002, +0.006',
            'pixel_conversion': 'S @ normalized_transform @ inverse(S), S=[[width-1,0,0],[0,height-1,0],[0,0,1]]',
        },
        'homographies': h,
        'homography_matrix_provenance': {
            'historical_R0_pixel_matrices': ['320x240', '256x192'],
            'pre_frozen_future_pixel_matrices': ['224x168', '192x144'],
            'note': 'Future matrices are frozen during this non-inference audit from the same deterministic normalized transform. Future runs must load these values rather than call a generator.',
        },
        'preprocessing': {
            'source_decode': 'cv2.imread(path, cv2.IMREAD_GRAYSCALE)',
            'source_color_policy': 'decode directly to one-channel grayscale; source color is discarded',
            'model_resize': 'cv2.resize(gray, (width,height), interpolation=cv2.INTER_AREA)',
            'model_resolutions_width_height': ['320x240', '256x192'],
            'aspect_ratio_policy': 'forced resize to requested width,height; no aspect-ratio preservation, crop, letterbox or pad',
            'normalization': 'uint8 grayscale converted by torch.from_numpy(...).float().div(255.0), range nominally [0,1]',
            'dtype_layout': 'torch.float32 NCHW [1,1,H,W] on CPU',
            'homography_timing': 'applied after resize to model-resolution single-channel FP32 array',
        },
        'postprocessing': {
            'declared_config': metrics['baseline_config'],
            'detector_score': 'softmax over 65 channels; discard dustbin channel; reshape 8x8 cell scores to full HxW score image',
            'nms': 'vendored superpoint.simple_nms, radius=3, two suppression refinement iterations',
            'original_superpoint_threshold': 'APPLIED: original vendored SuperPoint uses torch.nonzero(scores > 0.005) after NMS, then removes border 4',
            'v1_l1_threshold': 'NOT APPLIED in R0: create_superpoint_k210_w0.postprocess_raw uses torch.nonzero(score) after NMS without comparison to keypoint_threshold; it then removes border 4',
            'threshold_provenance': 'This split behavior is the exact R0 execution protocol. It explains V1 saturation and must be retained for strict historical comparability; it is not common threshold semantics across the two model families.',
            'coordinates': 'torch.nonzero produces (row,y; column,x), then torch.flip converts keypoints to (x,y) model-input pixel coordinates',
            'source_coordinate_scaling': 'none; no keypoints are scaled back to original source-image dimensions',
            'descriptor': 'dense descriptor map L2-normalized across 256 channels; vendored sample_descriptors bilinear grid_sample align_corners=True; sampled descriptors L2-normalized again',
            'fix_sampling': False,
            'pre_cap_max_keypoints': 3072,
            'evaluation_cap_order': 'torch.argsort(scores, descending=True, stable=True) then first min(cap,count)',
            'pre_cap_topk_order': 'vendored torch.topk for max_keypoints=3072; PyTorch tie order is not explicitly guaranteed by the source',
            'nms_tie_behavior': 'score == local max is retained by simple_nms before suppression; no custom tie breaker',
        },
        'topk_values': [256, 512, 1024],
        'metrics': {
            'detector_score_statistics': 'mean, median, population std of capped scores per image, then arithmetic mean across images',
            'spatial_coverage': 'grid_occupancy is occupied cells / 16 in a 4x4 grid over model input width,height; xy_spread=sqrt((std_x/width)^2+(std_y/height)^2)',
            'clean_image_agreement': 'for each Original keypoint, nearest candidate keypoint by torch.cdist; fraction distance <=3 px. Original self-agreement=1.',
            'repeatability': 'source keypoint is projected by stored H; valid denominator only projected points with 0<=x<width and 0<=y<height; numerator nearest transformed detection distance <=1, <=2, or <=3 px',
            'repeated_keypoint_localization_error': 'mean nearest transformed-detection distance over valid projected source points whose distance <=3 px; 0 if no repeats',
            'mnn_match_count': 'L2-normalize sampled descriptor columns, form source_desc @ transformed_desc.T, take argmax each direction; count mutual pairs',
            'mnn_precision': 'among MNN pairs, fraction that are valid source projections and whose source projected position is <=3 px from its matched transformed keypoint; denominator max(mnn_count,1)',
            'descriptor_cosine': 'mean cosine (dot product after L2 normalization) between each repeated source descriptor and descriptor at its nearest transformed geometric keypoint; repeat threshold <=3 px; 0 if none',
            'empty_case': 'all homography metrics are 0 if either side has no keypoints or no valid projected source points',
            'metric_source': 'scripts/superpoint_resolution_gate.py:detector_metrics and homography_metrics',
        },
        'models': {
            'original_superpoint': {
                'wrapper_implementation': str(EDGE / 'third_party/hloc/hloc/extractors/superpoint.py'),
                'network_implementation': str(ORIGINAL_SOURCE),
                'network_implementation_sha256': sha256(ORIGINAL_SOURCE),
                'checkpoint_path': str(ORIGINAL_CHECKPOINT), 'checkpoint_sha256': sha256(ORIGINAL_CHECKPOINT),
                'architecture': 'original SuperPoint encoder 64,64,128,128; detector 256->65; descriptor 256->256',
                'descriptor_dimension': 256, 'detector_output_dimension': 65, 'stride': 8,
            },
            'v1_fr12_l1init': {
                'implementation': str(ROOT / 'scripts/create_superpoint_k210_w0.py'),
                'implementation_sha256': sha256(ROOT / 'scripts/create_superpoint_k210_w0.py'),
                'checkpoint_path': str(V1_CHECKPOINT), 'checkpoint_sha256': sha256(V1_CHECKPOINT),
                'channel_map_path': str(W0_CHANNEL_MAP), 'channel_map_sha256': sha256(W0_CHANNEL_MAP),
                'architecture': 'V1-FR12 encoder 12,12,24,24,32,32,64,64; detector 64->128->65; descriptor 64->128->256',
                'descriptor_dimension': 256, 'detector_output_dimension': 65, 'stride': 8, 'parameter_count': 269989,
            },
            'r0_random_control': {'included_in_historical_R0_metrics': True, 'seed': 20260905,
                                  'not_required_for_future_resolution_decision': True},
        },
        'environment': environment(),
        'evaluation_script_path': str(R0_SCRIPT),
        'evaluation_script_sha256': sha256(R0_SCRIPT),
        'evaluation_script_hash_provenance': 'SHA256 of the audited current file. R0 did not record a contemporaneous script hash; report-only edits after R0 mean this is an audit snapshot, not a byte-level attestation of the historical execution file.',
    }


def verify(manifest: Dict[str, Any], check_current_selector: bool = True) -> Dict[str, Any]:
    assert manifest['protocol_version'] == 'SP-K210-R0-frozen-v1'
    images = manifest['ordered_images']
    assert manifest['image_count'] == len(images) == 30
    relatives = [image['relative_path'] for image in images]
    assert len(set(relatives)) == len(relatives)
    for image in images:
        path = Path(image['absolute_path'])
        assert path.is_file(), path
        assert path.stat().st_size == image['file_size_bytes'], path
        assert sha256(path) == image['sha256'], path
    h = manifest['homographies']
    assert len(h) == len(images)
    for image, entry in zip(images, h):
        assert entry['image_index'] == image['index'] and entry['relative_path'] == image['relative_path']
        assert set(entry['matrices_source_to_warped_xy_pixels']) == {'320x240', '256x192', '224x168', '192x144'}
        for matrix in entry['matrices_source_to_warped_xy_pixels'].values(): assert_matrix(matrix)
    assert manifest['topk_values'] == [256, 512, 1024]
    post = manifest['postprocessing']['declared_config']
    assert post == {'nms_radius': 3, 'keypoint_threshold': 0.005, 'max_keypoints': 3072,
                    'remove_borders': 4, 'fix_sampling': False}
    assert sha256(Path(manifest['models']['original_superpoint']['checkpoint_path'])) == manifest['models']['original_superpoint']['checkpoint_sha256']
    assert sha256(Path(manifest['models']['v1_fr12_l1init']['checkpoint_path'])) == manifest['models']['v1_fr12_l1init']['checkpoint_sha256']
    assert sha256(R0_SCRIPT) == manifest['evaluation_script_sha256'], 'Audited R0 script changed after manifest freeze'
    if check_current_selector:
        r0 = load_r0()
        assert relatives == [str(path.relative_to(EDGE)) for path in r0.select_images()]
        for image, entry in zip(images, h):
            for width, height in ALL_FROZEN_RESOLUTIONS:
                name = '{}x{}'.format(width, height)
                expected = matrix_list(r0.normalized_homography(image['index'], width, height))
                assert entry['matrices_source_to_warped_xy_pixels'][name] == expected
    return {'status': 'PASS', 'image_count': len(images), 'homography_entries': len(h),
            'current_selector_and_generator_checked': check_current_selector}


def write_audit(manifest: Dict[str, Any], validation: Dict[str, Any]) -> None:
    lines = ['# SP-K210-R0 protocol reproducibility audit', '',
             '## Answer', '',
             '**Yes for resolution-within-model comparison, provided future runs load `r0_protocol_manifest.json` without rediscovering images or independently generating homographies.** Exact 30-image identity, ordering, file hashes, and all required pixel-space homography matrices are frozen here. No model inference, compiler run, training, checkpoint modification or source-project change was performed for this audit.', '',
             '**Important qualification:** R0 did not apply the declared 0.005 detector threshold to V1-L1, while it did apply it to Original SuperPoint. Therefore Original-vs-V1 results are reproducible as executed but are not a strict same-threshold comparison. Future resolution runs must retain this exact split to compare against historical R0; correcting it would require rerunning all resolutions, which this audit does not authorize.', '',
             'Caveat: R0 did not save a script SHA at its execution time. The manifest records the audited current script SHA; it is sufficient to freeze future protocol use, but is not a byte-level provenance attestation of the historical R0 run. The relevant current image-selection, preprocessing, homography and metric functions were inspected directly and their reconstruction matches the R0 metrics image list.', '',
             '## Artifact consistency', '',
             '- R0 metric image count: `{}`; unique paths: `{}`.'.format(manifest['image_count'], manifest['image_count']),
             '- `r0_per_image.csv`: {} rows, {} unique relative paths, {} rows/image; mapping is unambiguous.'.format(
                 manifest['csv_mapping_audit']['data_rows'], manifest['csv_mapping_audit']['unique_relative_paths'], manifest['csv_mapping_audit']['rows_per_image']),
             '- Dry manifest validation: `{}`; every source file and both model checkpoint hashes match.'.format(validation['status']),
             '- R0 current-script SHA256: `{}`.'.format(manifest['evaluation_script_sha256']), '',
             '## Frozen ordered images', '',
             '| # | Relative path | Source shape | Bytes | SHA256 |', '|---:|---|---|---:|---|']
    for image in manifest['ordered_images']:
        lines.append('| {} | {} | {} | {} | `{}` |'.format(image['index'], image['relative_path'],
                     image['source_image_shape'], image['file_size_bytes'], image['sha256']))
    lines += ['', '## Homography protocol', '',
              '- No seed or random generator: transforms are deterministic index-modulo arithmetic; one source→warped transform per image per resolution.',
              '- Exact 3×3 float64 matrices for all 30 images at historical `320x240`/`256x192` and pre-frozen future `224x168`/`192x144` are stored in the manifest, not merely recoverable from a seed.',
              '- Normalized parameters: {}.'.format(manifest['homography_generation']),
              '- Coordinate space and warp: {}.'.format(manifest['homography_assignment']), '',
              '## Preprocessing and postprocessing', '',
              '- Preprocessing: {}.'.format(manifest['preprocessing']),
              '- Postprocessing: {}.'.format(manifest['postprocessing']), '',
              '## Metric definitions', '']
    for name, definition in manifest['metrics'].items(): lines.append('- `{}`: {}.'.format(name, definition))
    lines += ['', '## Model identities', '',
              '- Original SuperPoint checkpoint SHA256: `{}`; implementation: `{}`.'.format(
                  manifest['models']['original_superpoint']['checkpoint_sha256'], manifest['models']['original_superpoint']['network_implementation']),
              '- V1-FR12 L1-init checkpoint SHA256: `{}`; 256-D descriptor, 65 detector channels, stride 8.'.format(
                  manifest['models']['v1_fr12_l1init']['checkpoint_sha256']),
              '- Environment: `{}`.'.format(manifest['environment']), '',
              '## Required future use', '',
              'Future 224×168/192×144 code must load this manifest. It must use `ordered_images` exactly in index order, verify file hashes, use only the stored matrix entries appropriate to its resolution, retain all listed preprocessing/postprocessing/metric settings (including the historically split threshold behavior), and must not rediscover or resample images or transforms.', '',
              'R0 PROTOCOL AUDIT: PASS']
    AUDIT.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('freeze', 'verify'), nargs='?', default='freeze')
    action = parser.parse_args().action
    if action == 'freeze':
        manifest = manifest_data()
        validation = verify(manifest)
        MANIFEST.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
        write_audit(manifest, validation)
        print('R0 protocol manifest freeze PASS')
    else:
        manifest = json.loads(MANIFEST.read_text(encoding='utf-8'))
        print(json.dumps(verify(manifest), indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
