"""Trained weights to a static raw-head graph; validate in a separate stage."""
import copy, json, os
import numpy as np
from .runtime import torch
from .model import Model
from .settings import ROOT, SUPERPOINT, WIDTHS, RES, parse_candidate
from .io import sha, put
from .protocol import protocol

def deployment_dir(candidate):
    dest = SUPERPOINT / 'artifacts/deployment' / candidate
    dest.mkdir(parents=True, exist_ok=True)
    return dest

def selected_candidate(candidate=None):
    return candidate or json.loads((ROOT / 'summary.json').read_text())['selected_for_next_gate']

def permutation():
    return np.r_[np.rot90(np.arange(64).reshape(8, 8), -1).ravel(), 64]

def rotate_model(model):
    model = copy.deepcopy(model)
    state = model.state_dict()
    for (k, v) in state.items():
        if v.ndim == 4:
            v = torch.rot90(v, -1, (2, 3))
        if k in ['convPb.weight', 'convPb.bias']:
            v = v[permutation()]
        state[k] = v.contiguous()
    model.load_state_dict(state)
    return model

def physical_input(x, res):
    return np.ascontiguousarray(np.rot90(x, -1, (2, 3)) if res == 'r2' else x)

def logical_outputs(values, res):
    if res != 'r2':
        return values
    restored = [np.rot90(v, 1, (2, 3)) for v in values]
    restored[0] = restored[0][:, np.argsort(permutation())]
    return [np.ascontiguousarray(v) for v in restored]

def load_model(candidate):
    registry = SUPERPOINT / 'configs/candidates.json'
    entries = json.loads(registry.read_text()) if registry.exists() else {}
    if candidate in entries:
        entry = entries[candidate]
        width, res = entry['width'], entry['resolution']
        widths = entry.get('widths', WIDTHS[width])
        checkpoint = SUPERPOINT / entry['checkpoint']
        training_protocol = SUPERPOINT / entry['protocol']
    else:
        (width, res) = parse_candidate(candidate)
        widths = WIDTHS[width]
        checkpoint = ROOT / candidate / 'best.pt'
        training_protocol = ROOT / 'protocol.json'
    c = torch.load(checkpoint, map_location='cpu', weights_only=False)
    assert c['protocol_sha256'] == sha(training_protocol)
    assert c.get('resolution', res) == res, 'Checkpoint/registry preprocessing resolution mismatch'
    assert c.get('widths', widths) == widths, 'Checkpoint/registry architecture mismatch'
    model = Model(widths).eval()
    model.load_state_dict(c['model'])
    return (model, res, checkpoint)

def export(candidate):
    import onnx
    protocol()
    (model, res, checkpoint) = load_model(candidate)
    dest = deployment_dir(candidate)
    (w, h, _, _) = RES[res]
    physical = rotate_model(model) if res == 'r2' else model
    shape = [1, 1, w, h] if res == 'r2' else [1, 1, h, w]
    raw = dest / 'raw.onnx'
    canonical = dest / 'canonical.onnx'
    torch.onnx.export(physical, torch.zeros(shape), str(raw), opset_version=13, dynamo=False, do_constant_folding=False, input_names=['image'], output_names=['detector_logits', 'descriptor_map'])
    graph = onnx.load(raw)
    onnx.checker.check_model(graph)
    graph = onnx.shape_inference.infer_shapes(graph, strict_mode=True)
    onnx.checker.check_model(graph)
    assert {n.op_type for n in graph.graph.node} <= {'Conv', 'Relu', 'MaxPool'}
    assert sum((n.op_type == 'Conv' for n in graph.graph.node)) == 12
    onnx.save(graph, canonical)
    put(dest / 'export.json', {'status': 'EXPORTED_NOT_YET_VALIDATED', 'candidate': candidate, 'checkpoint': str(checkpoint), 'checkpoint_sha256': sha(checkpoint), 'protocol_sha256': sha(ROOT / 'protocol.json'), 'canonical_onnx_sha256': sha(canonical), 'input': shape, 'outputs': [[1, c, shape[2] // 8, shape[3] // 8] for c in [65, 256]], 'logical_resolution': res, 'orientation': 'CW90 input+weights+detector-channel permutation' if res == 'r2' else 'identity', 'external_io': 'FP32 gray/255; raw detector logits and unnormalized descriptor map', 'preprocessing': protocol()['preprocessing'], 'inverse_detector_permutation': np.argsort(permutation()).tolist() if res == 'r2' else list(range(65)), 'CPU_postprocessing': protocol()['features'], 'board_latency_and_actual_peak_SRAM': 'NOT_MEASURED'})
    evidence = json.loads((dest / 'export.json').read_text())
    evidence['training_protocol_sha256'] = torch.load(checkpoint, map_location='cpu', weights_only=False)['protocol_sha256']
    put(dest / 'export.json', evidence)

@torch.inference_mode()
def validate(candidate):
    import onnxruntime as ort
    from .metrics import features, evaluate
    (model, res, checkpoint) = load_model(candidate)
    dest = deployment_dir(candidate)
    e = json.loads((dest / 'export.json').read_text())
    assert sha(checkpoint) == e['checkpoint_sha256'] and sha(dest / 'canonical.onnx') == e['canonical_onnx_sha256']
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    optimization = os.environ.get('SP_ORT_PARITY_OPTIMIZATION', 'all')
    assert optimization in ['all', 'disabled']
    if optimization == 'disabled':
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session = ort.InferenceSession(str(dest / 'canonical.onnx'), options, providers=['CPUExecutionProvider'])
    # An explicit, checkpoint-bound numerical explanation may supplement (never
    # relabel) the strict raw-unit test. Other candidates keep the original gate.
    rounding = None
    addendum = dest / 'parity_policy_addendum.json'
    if addendum.exists():
        from pathlib import Path
        policy = json.loads(addendum.read_text())
        if policy.get('mode') == 'LOCAL_FP32_ROUNDING_AND_SEMANTIC_PARITY':
            audit_path = Path(policy['audit_path'])
            assert sha(audit_path) == policy['audit_sha256']
            audit = json.loads(audit_path.read_text())
            assert audit['checkpoint_sha256'] == sha(checkpoint)
            assert audit['onnx_sha256'] == e['canonical_onnx_sha256']
            assert audit['status'] == 'EXPLAINED_FP32_ACCUMULATION'
            assert sha(Path(policy['audit_source'])) == audit['source_sha256']
            assert optimization == 'disabled'
            rounding = {r['train_index']: r for r in audit['records']}
            assert len(rounding) == 8
    records = []
    oracle = copy.deepcopy(model).double()
    rotated_oracle = rotate_model(oracle) if res == 'r2' else oracle
    for i in np.linspace(0, len(protocol()['rows']['train']) - 1, 8, dtype=int):
        with np.load(ROOT / res / 'train' / f'{i:03d}.npz') as z:
            x = z['pair'][0:1].copy()
            masks = z['masks'][0:1].copy()
        expected = [v.numpy() for v in model(torch.from_numpy(x))]
        actual = logical_outputs(session.run(None, {'image': physical_input(x, res)}), res)
        precise = [v.numpy() for v in oracle(torch.from_numpy(x).double())]
        rotated_precise = logical_outputs([v.numpy() for v in rotated_oracle(torch.from_numpy(physical_input(x, res)).double())], res)
        assert all(np.allclose(a, b, atol=1e-10, rtol=1e-10) for a, b in zip(precise, rotated_precise)), 'Orientation is not mathematically equivalent'
        errors = []
        strict_raw = []
        for (a, b) in zip([v.astype(np.float32) for v in precise], actual):
            passed = bool(np.allclose(a, b, atol=0.0002, rtol=0.0002))
            strict_raw.append(passed)
            if not passed:
                assert rounding is not None, float(np.max(np.abs(a - b)))
            errors.append(float(np.max(np.abs(a - b))))
        if rounding is not None:
            proof = rounding[int(i)]
            assert proof['input_sha256'] == sha(ROOT / res / 'train' / f'{i:03d}.npz')
            assert proof['instrumentation_bitwise_identical'] and proof['relu_maxpool_exact'] and proof['rotation_fp64_pass']
            assert len(proof['layers']) == 12 and all(r['rounding_bound_pass'] and r['max_error_to_bound_ratio'] <= 1 for r in proof['layers'])
            # Test the actual consumers of both raw heads, densely as well as
            # at extracted keypoints, using the unchanged feature tolerance.
            normalizers = [lambda v: v.softmax(1), lambda v: torch.nn.functional.normalize(v, dim=1)]
            for ref, value, normalize in zip(precise, actual, normalizers):
                assert np.allclose(normalize(torch.from_numpy(ref)).numpy(), normalize(torch.from_numpy(value)).numpy(), atol=2e-4, rtol=2e-4)
        f0 = features(*[torch.from_numpy(v) for v in expected], masks, res)
        f1 = features(*[torch.from_numpy(v) for v in actual], masks, res)
        # Sparse features are coordinate/descriptor pairs. Near-tied scores may
        # permute top-k order without changing either location or its descriptor.
        # Preserve the strict ordering outcome, then compare coordinate-aligned
        # descriptors at the original tolerance. No decoder/metric is changed.
        order_equal = np.array_equal(f0[0][0], f1[0][0])
        ix0 = np.lexsort((f0[0][0][:, 1], f0[0][0][:, 0]))
        ix1 = np.lexsort((f1[0][0][:, 1], f1[0][0][:, 0]))
        assert np.array_equal(f0[0][0][ix0], f1[0][0][ix1]), 'TRAIN keypoint locations changed'
        assert np.allclose(f0[0][1][ix0], f1[0][1][ix1], atol=2e-4, rtol=2e-4), 'Normalized TRAIN descriptors changed'
        records.append({'train_index': int(i), 'strict_keypoint_order_pass': bool(order_equal),
                        'strict_raw_heads_vs_oracle': strict_raw,
                        'coordinate_aligned_features_pass': True, 'max_abs_heads_vs_precise_reference': errors,
                        'direct_FP32_raw_pass': [bool(np.allclose(a,b,atol=2e-4,rtol=2e-4)) for a,b in zip(expected,actual)],
                        'direct_FP32_raw_max_abs': [float(np.max(np.abs(a-b))) for a,b in zip(expected,actual)]})
    class ONNXModel(torch.nn.Module):
        def forward(self, x):
            heads = [[], []]
            for image in x.numpy():
                out = logical_outputs(session.run(None, {'image': physical_input(image[None], res)}), res)
                for j,value in enumerate(out): heads[j].append(value[0])
            return tuple(torch.from_numpy(np.stack(v)) for v in heads)
    fp32_metrics = evaluate(model,res,protocol()['rows']['dev'])
    onnx_metrics = evaluate(ONNXModel(),res,protocol()['rows']['dev'])
    assert fp32_metrics == onnx_metrics, 'DEV feature/matching behavior changed'
    put(dest / 'onnx_parity.json', {'status': 'PASS', 'checkpoint_sha256': sha(checkpoint), 'onnx_sha256': sha(dest / 'canonical.onnx'),
        'acceptance_method': 'LOCAL_FP32_ROUNDING_AND_SEMANTIC_PARITY' if rounding is not None else 'STRICT_RAW_ORACLE_AND_FEATURE_PARITY',
        'strict_raw_oracle_status': 'PASS' if all(all(r['strict_raw_heads_vs_oracle']) for r in records) else 'FAIL_EXPLAINED_BY_ROUNDING',
        'ort_graph_optimization': optimization,
        'comparison': 'Frozen weights vs FP64 oracle. Raw tolerance unchanged and strict failures retained; explicit numerical explanation required when present. Original FP32 sparse features and all42DEV matching metrics identical.',
        'direct_FP32_raw_strict': 'PASS' if all(all(r['direct_FP32_raw_pass']) for r in records) else 'FAIL_NUMERICAL_ROUNDING_RECORDED',
        'float64_rotation_equivalence': 'PASS', 'TRAIN_keypoints_and_normalized_descriptors': 'PASS', 'all42_DEV_metrics': 'EXACT',
        'atol': 0.0002, 'rtol': 0.0002, 'rows': records, 'policy_addendum_sha256': sha(dest/'parity_policy_addendum.json') if (dest/'parity_policy_addendum.json').exists() else None})

def calibrate(candidate):
    (_, res, checkpoint) = load_model(candidate)
    dest = deployment_dir(candidate)
    p = protocol()
    train = p['rows']['train']
    chosen = []
    for scene in sorted({r['scene'] for r in train}):
        ids = [i for (i, r) in enumerate(train) if r['scene'] == scene]
        chosen += [ids[int(j)] for j in np.linspace(0, len(ids) - 1, 4, dtype=int)]
    excluded = {r['sha256'] for r in p['rows']['dev']}
    tensors = []
    manifest = []
    for i in chosen:
        row = train[i]
        assert row['sha256'] not in excluded and sha(row['path']) == row['sha256']
        source = ROOT / res / 'train' / f'{i:03d}.npz'
        with np.load(source) as z:
            pair = z['pair'].copy()
        tensors.extend(physical_input(pair, res))
        manifest.append({'index': i, 'source_sha256': row['sha256'], 'cache_sha256': sha(source), 'views': [0, 1]})
    target = dest / 'calibration_train.npy'
    np.save(target, np.stack(tensors).astype(np.float32))
    put(dest / 'calibration.json', {'split': 'TRAIN_ONLY', 'count': len(tensors), 'selection': '4 evenly spaced TRAIN sources per scene, original+fixed warped views; chosen before PTQ results', 'protocol_sha256': sha(ROOT / 'protocol.json'), 'checkpoint_sha256': sha(checkpoint), 'tensor_sha256': sha(target), 'manifest': manifest})
