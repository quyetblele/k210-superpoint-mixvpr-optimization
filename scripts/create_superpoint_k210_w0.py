#!/usr/bin/env python3
"""SP-K210-W0: deterministic L1 channel transfer and zero-shot PC gate.

This script intentionally does no optimization.  It creates a V1-FR12 model
solely by copying structured subsets of the original SuperPoint checkpoint.
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
EDGE = Path('/home/quyet/edge_ai_project')
CHECKPOINT = EDGE / 'third_party/SuperGluePretrainedNetwork/models/weights/superpoint_v1.pth'
EXPECTED_CHECKPOINT_SHA256 = '52b6708629640ca883673b5d5c097c4ddad37d8048b33f09c8ca0d69db12c40e'
REPORT_DIR = ROOT / 'reports/superpoint_weight_transfer'
ARTIFACT = ROOT / 'models/superpoint_k210_v1_fr12_l1init.pth'
CHANNEL_MAP = REPORT_DIR / 'channel_map.json'
RESULTS = REPORT_DIR / 'w0_per_image.csv'
REPORT = REPORT_DIR / 'w0_report.md'
MODEL_SEED = 20260905
INPUT_SHAPE = (1, 1, 240, 320)
BASELINE_CONFIG = dict(nms_radius=3, keypoint_threshold=0.005, max_keypoints=3072,
                       remove_borders=4, fix_sampling=False)
TARGET_WIDTHS = OrderedDict([
    ('conv1a', (1, 12)), ('conv1b', (12, 12)),
    ('conv2a', (12, 24)), ('conv2b', (24, 24)),
    ('conv3a', (24, 32)), ('conv3b', (32, 32)),
    ('conv4a', (32, 64)), ('conv4b', (64, 64)),
    ('convPa', (64, 128)), ('convPb', (128, 65)),
    ('convDa', (64, 128)), ('convDb', (128, 256)),
])


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class V1FR12(nn.Module):
    """Exact V1-FR12 compiler-gate topology, with raw SuperPoint heads."""

    def __init__(self) -> None:
        super().__init__()
        self.relu = nn.ReLU(inplace=False)
        self.pool = nn.MaxPool2d(2, 2)
        for name, (cin, cout) in TARGET_WIDTHS.items():
            kernel = 1 if name in ('convPb', 'convDb') else 3
            padding = 0 if kernel == 1 else 1
            setattr(self, name, nn.Conv2d(cin, cout, kernel, 1, padding, bias=True))

    def forward_raw(self, image: Tensor) -> Tuple[Tensor, Tensor]:
        x = self.relu(self.conv1a(image))
        x = self.relu(self.conv1b(x))
        x = self.pool(x)
        x = self.relu(self.conv2a(x))
        x = self.relu(self.conv2b(x))
        x = self.pool(x)
        x = self.relu(self.conv3a(x))
        x = self.relu(self.conv3b(x))
        x = self.pool(x)
        x = self.relu(self.conv4a(x))
        x = self.relu(self.conv4b(x))
        detector_logits = self.convPb(self.relu(self.convPa(x)))
        descriptor_map = self.convDb(self.relu(self.convDa(x)))
        return detector_logits, descriptor_map

    def forward(self, image: Tensor) -> Tuple[Tensor, Tensor]:
        return self.forward_raw(image)


def import_original_superpoint():
    third_party = str(EDGE / 'third_party')
    hloc = str(EDGE / 'third_party/hloc')
    for path in (third_party, hloc):
        if path not in sys.path:
            sys.path.insert(0, path)
    from SuperGluePretrainedNetwork.models import superpoint
    from hloc import extractors
    from hloc.utils.base_model import dynamic_load
    model = dynamic_load(extractors, 'superpoint')(BASELINE_CONFIG).eval().cpu()
    return model, superpoint


def stable_topk(importance: Tensor, count: int) -> List[int]:
    # Python sorting explicitly resolves equal importance by original index.
    ranked = sorted(range(int(importance.numel())), key=lambda i: (-float(importance[i]), i))
    return ranked[:count]


def transfer_conv(source: nn.Conv2d, target: nn.Conv2d, source_name: str,
                  input_indices: List[int], keep_outputs: int, channel_map: Dict) -> List[int]:
    weight = source.weight.detach().cpu()
    assert weight.shape[1] >= len(input_indices)
    restricted = weight[:, input_indices, :, :]
    importance = restricted.abs().sum(dim=(1, 2, 3))
    selected = list(range(weight.shape[0])) if keep_outputs == weight.shape[0] else stable_topk(importance, keep_outputs)
    with torch.no_grad():
        target.weight.copy_(weight[selected][:, input_indices, :, :])
        target.bias.copy_(source.bias.detach().cpu()[selected])
    assert tuple(target.weight.shape) == (len(selected), len(input_indices), *weight.shape[2:])
    assert torch.equal(target.weight.detach().cpu(), weight[selected][:, input_indices, :, :])
    assert torch.equal(target.bias.detach().cpu(), source.bias.detach().cpu()[selected])
    selected_importance = importance[selected]
    channel_map[source_name] = {
        'original_input_indices': input_indices,
        'selected_output_indices': selected,
        'original_width': int(weight.shape[0]),
        'target_width': int(keep_outputs),
        'importance_statistics': {
            'restricted_all_min': float(importance.min()),
            'restricted_all_mean': float(importance.mean()),
            'restricted_all_max': float(importance.max()),
            'selected_min': float(selected_importance.min()),
            'selected_mean': float(selected_importance.mean()),
            'selected_max': float(selected_importance.max()),
        },
        'source_weight_shape': list(weight.shape),
        'target_weight_shape': list(target.weight.shape),
        'bias_copied': True,
    }
    return selected


def initialize_from_original(original_net: nn.Module) -> Tuple[V1FR12, Dict]:
    target = V1FR12().eval().cpu()
    cmap = {'algorithm': {
        'name': 'deterministic L1 filter importance after input-channel restriction',
        'tie_break': 'descending importance then ascending original channel index',
        'branch_rule': 'conv4b surviving outputs are input indices for both convPa and convDa',
    }, 'layers': OrderedDict()}
    previous = [0]
    for name in ('conv1a', 'conv1b', 'conv2a', 'conv2b', 'conv3a', 'conv3b', 'conv4a', 'conv4b'):
        previous = transfer_conv(getattr(original_net, name), getattr(target, name), name,
                                 previous, TARGET_WIDTHS[name][1], cmap['layers'])
    shared_conv4b = list(previous)
    detector = transfer_conv(original_net.convPa, target.convPa, 'convPa', shared_conv4b,
                             128, cmap['layers'])
    # Keep every original detector class. It is not a filter-selection operation.
    transfer_conv(original_net.convPb, target.convPb, 'convPb', detector, 65, cmap['layers'])
    descriptor = transfer_conv(original_net.convDa, target.convDa, 'convDa', shared_conv4b,
                               128, cmap['layers'])
    # Keep every descriptor coordinate; descriptor dimension remains 256.
    transfer_conv(original_net.convDb, target.convDb, 'convDb', descriptor, 256, cmap['layers'])
    cmap['shared_conv4b_output_indices'] = shared_conv4b
    cmap['all_target_parameters_copied'] = True
    return target, cmap


def postprocess_raw(logits: Tensor, dense: Tensor, sp_module) -> Dict[str, Tensor]:
    """The original SuperPoint postprocessing, with fix_sampling=False."""
    scores = torch.softmax(logits, 1)[:, :-1]
    batch, _, height, width = scores.shape
    scores = scores.permute(0, 2, 3, 1).reshape(batch, height, width, 8, 8)
    scores = scores.permute(0, 1, 3, 2, 4).reshape(batch, height * 8, width * 8)
    scores = sp_module.simple_nms(scores, BASELINE_CONFIG['nms_radius'])
    all_keys, all_scores, all_desc = [], [], []
    normalized_dense = F.normalize(dense, p=2, dim=1)
    for score, desc in zip(scores, normalized_dense):
        keys = torch.nonzero(score)
        key_scores = score[tuple(keys.t())]
        keys, key_scores = sp_module.remove_borders(keys, key_scores,
            BASELINE_CONFIG['remove_borders'], height * 8, width * 8)
        keys, key_scores = sp_module.top_k_keypoints(keys, key_scores,
            BASELINE_CONFIG['max_keypoints'])
        keys = torch.flip(keys, [1]).float()
        sampled = sp_module.sample_descriptors(keys[None], desc[None], 8)[0]
        all_keys.append(keys)
        all_scores.append(key_scores)
        all_desc.append(sampled)
    return {'keypoints': all_keys[0], 'scores': all_scores[0], 'descriptors': all_desc[0],
            'dense_norms': dense.norm(p=2, dim=1)[0]}


def select_images() -> List[Path]:
    root = EDGE / 'assets/map_prepared_v2'
    groups = []
    for directory in sorted(p for p in root.iterdir() if p.is_dir()):
        images = sorted(directory.glob('*.jpg'))
        if images:
            groups.append(images)
    if not groups:
        raise RuntimeError('No project image sequence found')
    # Prefer one median frame from every sequence.  This dataset has 26 sequences,
    # so fill the final four slots round-robin with the first non-median frame from
    # the earliest sequences.  The order is entirely deterministic and stays within
    # the project's prepared map imagery.
    chosen = [images[len(images) // 2] for images in groups]
    offsets = [0] * len(groups)
    while len(chosen) < 30:
        progress = False
        for group_index, images in enumerate(groups):
            median_index = len(images) // 2
            while offsets[group_index] < len(images) and offsets[group_index] == median_index:
                offsets[group_index] += 1
            if offsets[group_index] < len(images):
                chosen.append(images[offsets[group_index]])
                offsets[group_index] += 1
                progress = True
                if len(chosen) == 30:
                    break
        if not progress:
            raise RuntimeError('Project imagery contains fewer than 30 files')
    return chosen[:30]


def load_image(path: Path) -> Tensor:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError('Cannot read image: {}'.format(path))
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    interpolation = cv2.INTER_AREA if gray.shape[0] >= 240 or gray.shape[1] >= 320 else cv2.INTER_LINEAR
    gray = cv2.resize(gray, (320, 240), interpolation=interpolation)
    return torch.from_numpy(gray).float().div(255.0)[None, None]


def summary(values: Tensor) -> Dict[str, float]:
    if values.numel() == 0:
        return {'count': 0, 'min': None, 'mean': None, 'std': None, 'median': None, 'max': None}
    return {'count': int(values.numel()), 'min': float(values.min()), 'mean': float(values.mean()),
            'std': float(values.std(unbiased=False)), 'median': float(values.median()), 'max': float(values.max())}


def pair_metrics(reference: Dict[str, Tensor], candidate: Dict[str, Tensor]) -> Dict[str, object]:
    a, b = reference['keypoints'], candidate['keypoints']
    if not len(a) or not len(b):
        return {'mutual_near_count': 0, 'reference_overlap_fraction': 0.0,
                'nearest_coordinate_distance': summary(torch.empty(0)), 'descriptor_cosine': summary(torch.empty(0))}
    distance = torch.cdist(a, b)
    nearest_distance, nearest_b = distance.min(dim=1)
    nearest_a = distance.min(dim=0).indices
    mutual_a = torch.arange(len(a)) == nearest_a[nearest_b]
    mask = mutual_a & (nearest_distance <= 3.0)
    cosine = (reference['descriptors'][:, mask] * candidate['descriptors'][:, nearest_b[mask]]).sum(dim=0)
    return {'mutual_near_count': int(mask.sum()), 'reference_overlap_fraction': float(mask.float().mean()),
            'nearest_coordinate_distance': summary(nearest_distance), 'descriptor_cosine': summary(cosine)}


def aggregate(records: List[Dict], key: str) -> Dict[str, float]:
    vals = [r[key] for r in records]
    return {'images': len(vals), 'mean': float(np.mean(vals)), 'median': float(np.median(vals)),
            'min': float(np.min(vals)), 'max': float(np.max(vals))}


def evaluate(original, structured, random_model, sp_module) -> Tuple[List[Dict], Dict]:
    records = []
    for path in select_images():
        image = load_image(path)
        with torch.inference_mode():
            original_out = original({'image': image})
            sl_logits, sl_dense = structured.forward_raw(image)
            rd_logits, rd_dense = random_model.forward_raw(image)
            slim_out = postprocess_raw(sl_logits, sl_dense, sp_module)
            random_out = postprocess_raw(rd_logits, rd_dense, sp_module)
        # Original HLOC path uses the same vendored postprocessing semantics.
        original_post = {'keypoints': original_out['keypoints'][0], 'scores': original_out['scores'][0],
                         'descriptors': original_out['descriptors'][0],
                         'dense_norms': torch.empty(0)}
        slim_pair = pair_metrics(original_post, slim_out)
        random_pair = pair_metrics(original_post, random_out)
        record = {
            'image': str(path.relative_to(EDGE)),
            'original_keypoints': len(original_post['keypoints']),
            'structured_keypoints': len(slim_out['keypoints']),
            'random_keypoints': len(random_out['keypoints']),
            'original_score_mean': float(original_post['scores'].mean()) if len(original_post['scores']) else None,
            'structured_score_mean': float(slim_out['scores'].mean()) if len(slim_out['scores']) else None,
            'random_score_mean': float(random_out['scores'].mean()) if len(random_out['scores']) else None,
            'structured_mutual_near_count': slim_pair['mutual_near_count'],
            'structured_overlap_fraction': slim_pair['reference_overlap_fraction'],
            'structured_nearest_distance_mean': slim_pair['nearest_coordinate_distance']['mean'],
            'structured_descriptor_cosine_mean': slim_pair['descriptor_cosine']['mean'],
            'random_mutual_near_count': random_pair['mutual_near_count'],
            'random_overlap_fraction': random_pair['reference_overlap_fraction'],
            'random_nearest_distance_mean': random_pair['nearest_coordinate_distance']['mean'],
            'random_descriptor_cosine_mean': random_pair['descriptor_cosine']['mean'],
            'structured_dense_norm': summary(slim_out['dense_norms']),
            'random_dense_norm': summary(random_out['dense_norms']),
            'structured_descriptor_shape': list(slim_out['descriptors'].shape),
            'random_descriptor_shape': list(random_out['descriptors'].shape),
        }
        records.append(record)
    scalar_keys = ('original_keypoints', 'structured_keypoints', 'random_keypoints',
                   'structured_mutual_near_count', 'random_mutual_near_count',
                   'structured_overlap_fraction', 'random_overlap_fraction')
    return records, {key: aggregate(records, key) for key in scalar_keys}


def write_csv(records: List[Dict]) -> None:
    flat = []
    for row in records:
        item = {}
        for key, value in row.items():
            if isinstance(value, dict):
                for nested_key, nested_value in value.items():
                    item['{}_{}'.format(key, nested_key)] = nested_value
            elif isinstance(value, list):
                item[key] = 'x'.join(str(dimension) for dimension in value)
            else:
                item[key] = value
        flat.append(item)
    with RESULTS.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)


def write_report(cmap: Dict, structured: V1FR12, records: List[Dict], aggregates: Dict,
                 checkpoint_sha: str, model_sha: str, random_state_sha: str) -> None:
    parameter_count = sum(p.numel() for p in structured.parameters())
    lines = ['# SP-K210-W0 structured pretrained initialization', '',
             'This gate creates a deterministic structured initialization for the compiler-feasible V1-FR12 architecture. It performs no training, fine-tuning, distillation, iterative pruning, QAT or quantization-quality evaluation.', '',
             '## Sources and reproducibility', '',
             '- Original source: `/home/quyet/edge_ai_project/third_party/SuperGluePretrainedNetwork/models/superpoint.py`.',
             '- Original checkpoint: `{}`.'.format(CHECKPOINT),
             '- Checkpoint SHA256: `{}` (verified).'.format(checkpoint_sha),
             '- Model seed: `{}`. Random comparator uses this fixed seed.'.format(MODEL_SEED),
             '- Baseline configuration: `{}`.'.format(BASELINE_CONFIG),
             '- Evaluation preprocessing: project image → OpenCV grayscale → fixed 320×240 → FP32 [0,1]. Applied identically to all three models.', '',
             '## Exact V1-FR12 architecture', '',
             'Input FP32 `[1,1,240,320]`; conv1a/conv1b are full resolution 12-channel layers, followed by pool; then 24, 32 and 64-channel stages with the original remaining pooling schedule. Shared output is `[1,64,30,40]`. Detector head is `64→128→65`; descriptor head is `64→128→256`; all convolution biases are enabled.',
             '- Parameter count: `{}`.'.format(parameter_count),
             '- Initialized model: `{}`.'.format(ARTIFACT.relative_to(ROOT)),
             '- Initialized model SHA256: `{}`.'.format(model_sha),
             '- Deterministic random comparator state SHA256: `{}`.'.format(random_state_sha), '',
             '## Channel-selection algorithm', '',
             'For each convolution, W is first restricted to the surviving input channels. Each original output filter is scored by `sum(abs(W_restricted))`; top-K filters are selected by descending score and ascending original channel index on a tie. Selected outputs become the next encoder layer inputs. At conv4b, that one selected channel set feeds both convPa and convDa. convPb preserves all 65 detector outputs; convDb preserves all 256 descriptor dimensions.',
             '- Complete channel map and importance statistics: [channel_map.json](channel_map.json).',
             '- Structural copy verification: every target weight and bias is exact indexed data from the verified checkpoint; no random tensor remains in the initialized model.', '',
             '## Structural validation', '',
             '- All target parameter shapes match V1-FR12: PASS.',
             '- All target values re-verified against their indexed checkpoint sources: PASS.',
             '- `state_dict` save/load round-trip: PASS.',
             '- Forward FP32 `[1,1,240,320]`: PASS; finite outputs: PASS.',
             '- detector_logits: `[1,65,30,40]`; descriptor_map: `[1,256,30,40]`.', '',
             '## Zero-shot PC evaluation', '',
             'Thirty real images were selected deterministically from `assets/map_prepared_v2`: one median frame from each available sequence (26 in this dataset), then the first non-median frame round-robin from the earliest sequences to reach 30. The original pretrained model, structured initialization, and random V1-FR12 used the same input and SuperPoint postprocessing semantics (`fix_sampling=False`).', '',
             '| Metric | Original | Structured-init | Random V1 |', '|---|---:|---:|---:|']
    def number(key, model):
        return '{:.2f}'.format(aggregates[key]['mean']) if model else '{:.2f}'.format(aggregates[key]['mean'])
    lines += [
        '| Mean keypoint count | {} | {} | {} |'.format(number('original_keypoints', 1), number('structured_keypoints', 1), number('random_keypoints', 1)),
        '| Median keypoint count | {:.2f} | {:.2f} | {:.2f} |'.format(aggregates['original_keypoints']['median'], aggregates['structured_keypoints']['median'], aggregates['random_keypoints']['median']),
        '| Mean mutual near-keypoint count vs original (≤3 px) | — | {:.2f} | {:.2f} |'.format(aggregates['structured_mutual_near_count']['mean'], aggregates['random_mutual_near_count']['mean']),
        '| Mean original-keypoint overlap fraction | — | {:.4f} | {:.4f} |'.format(aggregates['structured_overlap_fraction']['mean'], aggregates['random_overlap_fraction']['mean']),
        '',
        'Descriptor output for both V1 models is `[256,N]` at extracted keypoints; dense raw-head output is `[1,256,30,40]`. Descriptor cosine is only reported for mutually nearest keypoints within 3 pixels. Full per-image score statistics, nearest-coordinate distance, dense descriptor norm statistics and descriptor cosine are in [w0_per_image.csv](w0_per_image.csv).',
        '', '## Optional localization-side check', '',
        'NOT RUN. Existing Point3D map descriptors and 2D–3D associations are in the original SuperPoint descriptor space. Testing the new descriptor space would require rebuilding map associations or introducing a new localization pipeline, outside this gate.', '',
        '## Limitations', '',
        'Zero-shot behavior is diagnostic only. Keypoint overlap and descriptor cosine do not establish retrieval, matching, PnP, localization accuracy, K210 latency, board execution, or full-system RAM fit. Random-network keypoint counts are not quality evidence.', '',
        '## Files created/modified', '',
        '- `scripts/create_superpoint_k210_w0.py`.',
        '- `models/superpoint_k210_v1_fr12_l1init.pth`.',
        '- `reports/superpoint_weight_transfer/channel_map.json`.',
        '- `reports/superpoint_weight_transfer/w0_per_image.csv`.',
        '- `reports/superpoint_weight_transfer/w0_report.md`.',
        '', 'SP-K210-W0 PASS']
    REPORT.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    if sha256_file(CHECKPOINT) != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError('Original checkpoint SHA256 mismatch')
    original, sp_module = import_original_superpoint()
    target, cmap = initialize_from_original(original.net)
    # Saved state contains no fields beyond exact channel-selected Conv parameters.
    torch.save(target.state_dict(), ARTIFACT)
    restored = V1FR12().eval().cpu()
    restored.load_state_dict(torch.load(ARTIFACT, map_location='cpu', weights_only=True))
    assert all(torch.equal(v, restored.state_dict()[k]) for k, v in target.state_dict().items())
    with torch.inference_mode():
        logits, dense = target.forward_raw(torch.linspace(0, 1, 240 * 320).reshape(INPUT_SHAPE))
    assert logits.shape == (1, 65, 30, 40) and dense.shape == (1, 256, 30, 40)
    assert torch.isfinite(logits).all() and torch.isfinite(dense).all()
    cmap['source_checkpoint_sha256'] = sha256_file(CHECKPOINT)
    cmap['initialized_artifact_sha256'] = sha256_file(ARTIFACT)
    cmap['target_parameter_count'] = sum(parameter.numel() for parameter in target.parameters())
    cmap['target_state_tensor_count'] = len(target.state_dict())
    CHANNEL_MAP.write_text(json.dumps(cmap, indent=2) + '\n', encoding='utf-8')
    torch.manual_seed(MODEL_SEED)
    random_model = V1FR12().eval().cpu()
    random_hasher = hashlib.sha256()
    for key, value in random_model.state_dict().items():
        random_hasher.update(key.encode())
        random_hasher.update(value.numpy().tobytes())
    records, aggregates = evaluate(original, target, random_model, sp_module)
    write_csv(records)
    write_report(cmap, target, records, aggregates, sha256_file(CHECKPOINT), sha256_file(ARTIFACT), random_hasher.hexdigest())
    print('SP-K210-W0 PASS')
    print('Initialized model:', ARTIFACT)
    print('Report:', REPORT)
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as error:
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        REPORT.write_text('# SP-K210-W0\n\nFailure: `{}`\n\nSP-K210-W0 FAIL\n'.format(error), encoding='utf-8')
        raise
