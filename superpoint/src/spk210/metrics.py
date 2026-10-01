"""CPU feature decoding and homography DEV evaluation in common coordinates."""
import numpy as np
from .runtime import torch, cv2, ops
import torch.nn.functional as F
from .settings import ROOT, RES
from .postprocess import default_nms, greedy_keypoints

def project(points, h):
    v = np.c_[points, np.ones(len(points))] @ h.T
    return v[:, :2] / v[:, 2:]

def features(a, b, masks, res, nms=None):
    (w, h, scale, left) = RES[res]
    score = a.softmax(1)[:, :64]
    (n, _, hh, ww) = score.shape
    score = score.permute(0, 2, 3, 1).reshape(n, hh, ww, 8, 8).permute(0, 1, 3, 2, 4).reshape(n, h, w)
    score = score * torch.from_numpy(masks)
    nms = default_nms() if nms is None else nms
    if nms not in ('original', 'greedy'): raise ValueError('Unknown NMS: ' + str(nms))
    if nms == 'original': score = ops.simple_nms(score, round(3 * scale))
    out = []
    for k in range(n):
        if nms == 'greedy':
            xy = torch.from_numpy(greedy_keypoints(score[k], round(3 * scale), 0.005, 160)).to(b.device)
        else:
            yx = torch.nonzero(score[k] > 0.005)
            conf = score[k][tuple(yx.t())]
            (yx, conf) = ops.top_k_keypoints(yx, conf, 160)
            xy = yx.flip([1]).float()
        desc = ops.sample_descriptors(xy[None], F.normalize(b[k:k + 1], dim=1), 8)[0].T.numpy()
        coords = xy.numpy()
        coords = (coords - np.array([left, 0])) / scale
        out.append((coords, desc))
    return out

def pair_metrics(f0, f1, h, rect):
    (x, d) = f0
    (y, e) = f1
    zero = {k: 0.0 for k in ['correct_mnn_count', 'mnn_precision', 'mnn_count', 'repeatability', 'coverage']}
    if not len(x) or not len(y):
        return zero
    pred = project(x, h)
    distances = np.linalg.norm(pred[:, None] - y[None], axis=2)
    sim = d @ e.T
    forward = sim.argmax(1)
    reverse = sim.argmax(0)
    ii = np.flatnonzero(reverse[forward] == np.arange(len(x)))
    correct = ii[distances[ii, forward[ii]] <= 3]
    coverage = 0
    if len(correct):
        uv = (x[correct] - rect[:2]) / rect[2:]
        bins = np.clip((uv * 4).astype(int), 0, 3)
        coverage = len(np.unique(bins, axis=0)) / 16
    return {'correct_mnn_count': float(len(correct)), 'mnn_precision': len(correct) / max(1, len(ii)), 'mnn_count': float(len(ii)), 'repeatability': float((distances.min(1) <= 3).mean()), 'coverage': coverage}

@torch.inference_mode()
def evaluate(model, res, rows, feature_extractor=None):
    was = model.training
    model.eval()
    records = []
    (w, h, scale, left) = RES[res]
    for (i, row) in enumerate(rows):
        with np.load(ROOT / res / 'dev' / f'{i:03d}.npz') as z:
            (a, b) = model(torch.from_numpy(z['pair']))
            masks = z['masks'].copy()
            hc = z['homography']
            s = np.array([[scale, 0, left], [0, scale, 0], [0, 0, 1.0]])
            tr = s @ hc @ np.linalg.inv(s)
            (m0, m1) = masks.copy()
            masks[0] &= cv2.warpPerspective(m1, np.linalg.inv(tr), (w, h), flags=cv2.INTER_NEAREST)
            masks[1] &= cv2.warpPerspective(m0, tr, (w, h), flags=cv2.INTER_NEAREST)
            f = features(a, b, masks, res) if feature_extractor is None else feature_extractor(a, b, masks, res, z['pair'])
            m = pair_metrics(*f, hc, z['content_rect'])
            m['mean_keypoints'] = float(np.mean([len(x[0]) for x in f]))
            records.append({'index': i, 'scene': row['scene'], 'domain': row['domain'], 'metrics': m})

    def avg(rr):
        return {k: float(np.mean([r['metrics'][k] for r in rr])) for k in records[0]['metrics']}
    scenes = {s: avg([r for r in records if r['scene'] == s]) for s in sorted({r['scene'] for r in records})}
    macro = {k: float(np.mean([v[k] for v in scenes.values()])) for k in records[0]['metrics']}
    model.train(was)
    return {'macro': macro, 'per_scene': scenes, 'per_domain': {d: avg([r for r in records if r['domain'] == d]) for d in ['indoor', 'project']}, 'rows': records}

def key(m):
    return tuple((m['macro'][k] for k in ['correct_mnn_count', 'mnn_precision', 'repeatability']))

def verify():
    x = np.array([[20.0, 20.0], [40.0, 30.0], [80.0, 70.0], [100.0, 150.0]])
    h = np.array([[1.0, 0.0, 3.0], [0.0, 1.0, -2.0], [0.0, 0.0, 1.0]])
    m = pair_metrics((x, np.eye(4)), (project(x, h), np.eye(4)), h, np.array([0, 0, 144, 256]))
    assert m['correct_mnn_count'] == 4 and m['mnn_precision'] == 1 and (m['repeatability'] == 1)
    for (_, (_, _, scale, left)) in RES.items():
        s = np.array([[scale, 0, left], [0, scale, 0], [0, 0, 1.0]])
        assert np.allclose(project(project(x, s), s @ h @ np.linalg.inv(s)), project(project(x, h), s))
    assert pair_metrics((x[:0], np.eye(4)[:0]), (x, np.eye(4)), h, np.array([0, 0, 144, 256]))['correct_mnn_count'] == 0
    print('METRIC_GEOMETRY_CHECKS_PASS', flush=True)
