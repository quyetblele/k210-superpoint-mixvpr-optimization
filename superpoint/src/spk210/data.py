"""Deterministic source-image pairs, teacher cache and matched geometry supervision."""
import gc, json
import numpy as np
from .runtime import torch, cv2, w0, SuperPointKPUWrapper
from .settings import ROOT, WIDTHS, RES, SEED
from .io import sha, put, save, guard
from .protocol import protocol, verify_cache
from .model import Model
from .metrics import project, evaluate

def prepare():
    p = protocol()
    if not (ROOT / 'plan.npy').exists():
        np.save(ROOT / 'plan.npy', np.random.default_rng(SEED).integers(0, len(p['rows']['train']), size=(400, 2)))
    if (ROOT / 'cache_done.json').exists():
        verify_cache()
        return
    (teacher, _) = w0.import_original_superpoint()
    raw = SuperPointKPUWrapper(teacher).eval().requires_grad_(False)
    for (name, width) in WIDTHS.items():
        m = Model(width)
        cmap = {}
        prev = [0]
        for (stage, co) in enumerate(width, 1):
            for suffix in 'ab':
                key = f'conv{stage}{suffix}'
                prev = w0.transfer_conv(getattr(teacher.net, key), getattr(m, key), key, prev, co, cmap)
        for (a, b, count) in [('convPa', 'convPb', 65), ('convDa', 'convDb', 256)]:
            head = w0.transfer_conv(getattr(teacher.net, a), getattr(m, a), a, prev, 128, cmap)
            w0.transfer_conv(getattr(teacher.net, b), getattr(m, b), b, head, count, cmap)
        save(ROOT / f'{name}_init.pt', m.state_dict())
        put(ROOT / f'{name}_channel_map.json', cmap)
    for (res, (w, h, scale, left)) in RES.items():
        folder = ROOT / res
        folder.mkdir(exist_ok=True)
        for (split, rows) in p['rows'].items():
            dest = folder / split
            dest.mkdir(exist_ok=True)
            cache = np.lib.format.open_memmap(dest / 'teacher_logits.npy', mode='w+', dtype=np.float16, shape=(len(rows), 2, 65, h // 8, w // 8))
            for (i, row) in enumerate(rows):
                guard()
                assert sha(row['path']) == row['sha256']
                image = cv2.imread(row['path'], 0)
                (ih, iw) = image.shape
                factor = min(288 / iw, 512 / ih)
                (nw, nh) = (round(iw * factor), round(ih * factor))
                (x0, y0) = ((288 - nw) // 2, (512 - nh) // 2)
                common = np.zeros((512, 288), np.uint8)
                mask = np.zeros_like(common)
                common[y0:y0 + nh, x0:x0 + nw] = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA)
                mask[y0:y0 + nh, x0:x0 + nw] = 1
                rng = np.random.default_rng(SEED + i + (10000 if split == 'dev' else 0))
                hc = np.eye(3)
                hc[:2] = cv2.getRotationMatrix2D((72, 128), rng.uniform(-10, 10), rng.uniform(0.92, 1.08))
                hc[:2, 2] += rng.uniform(-4, 4, 2)
                brightness = rng.uniform(0.8, 1.2)
                s = np.array([[scale, 0, left], [0, scale, 0], [0, 0, 1.0]])
                tr = s @ hc @ np.linalg.inv(s)
                im = np.zeros((h, w), np.uint8)
                valid = np.zeros_like(im)
                im[:, left:left + round(144 * scale)] = cv2.resize(common, (round(144 * scale), h), interpolation=cv2.INTER_AREA)
                valid[:, left:left + round(144 * scale)] = cv2.resize(mask, (round(144 * scale), h), interpolation=cv2.INTER_NEAREST)
                warped = cv2.warpPerspective(im, tr, (w, h))
                vm = cv2.warpPerspective(valid, tr, (w, h), flags=cv2.INTER_NEAREST)
                pair = np.stack([im, np.clip(warped.astype(float) * brightness, 0, 255).astype(np.uint8)])[:, None].astype(np.float32) / 255
                safe = cv2.erode(mask, np.ones((33, 33), np.uint8))
                corners = cv2.goodFeaturesToTrack(common, 100, 0.01, 12, mask=safe)
                if corners is None:
                    raise RuntimeError('No training corners: ' + row['path'])
                pts = corners[:, 0] / 2
                target = project(pts, hc)
                q = project(pts, s)
                tq = project(target, s)
                masks = np.stack([valid, vm])
                eroded = np.stack([cv2.erode(v, np.ones((2 * int(np.ceil(4 * scale)) + 1,) * 2, np.uint8), borderType=cv2.BORDER_CONSTANT, borderValue=0) for v in masks])
                keep = np.ones(len(q), bool)
                for (xy, v) in zip([q, tq], eroded):
                    ix = np.rint(xy).astype(int)
                    inside = (ix[:, 0] >= 0) & (ix[:, 0] < w) & (ix[:, 1] >= 0) & (ix[:, 1] < h)
                    good = np.zeros(len(ix), bool)
                    good[inside] = v[ix[inside, 1], ix[inside, 0]] > 0
                    keep &= good
                if keep.sum() < 8:
                    raise RuntimeError('Too few valid correspondences')
                cell = masks.reshape(2, h // 8, 8, w // 8, 8).min((2, 4)).astype(np.float32)
                np.savez_compressed(dest / f'{i:03d}.npz', pair=pair, points=np.stack([q[keep][:64], tq[keep][:64]]).astype(np.float32), homography=hc, masks=eroded, cell=cell, source_to_common=np.array([factor / 2, x0 / 2, y0 / 2]), content_rect=np.array([x0 / 2, y0 / 2, nw / 2, nh / 2]))
                with torch.inference_mode():
                    (a, b) = raw(torch.from_numpy(pair))
                    cache[i] = a.numpy().astype(np.float16)
                if i % 35 == 0:
                    print('CACHE', res, split, i, flush=True)
            cache.flush()
            del cache
        teacher_metrics = evaluate(raw, res, p['rows']['dev'])
        put(folder / 'teacher_dev.json', teacher_metrics)
    del teacher, raw
    gc.collect()
    match_correspondences()
    files = sorted((f for f in ROOT.rglob('*') if f.is_file() and f.suffix in ('.npy', '.npz', '.pt', '.json') and (f.name != 'cache_done.json')))
    put(ROOT / 'cache_done.json', {'protocol_sha256': sha(ROOT / 'protocol.json'), 'files': {str(f.relative_to(ROOT)): sha(f) for f in files}})

def match_correspondences():
    """Use the intersection, in identical order, of valid canonical points at both resolutions."""
    audit = []
    for (split, rows) in protocol()['rows'].items():
        for i in range(len(rows)):
            name = f'{i:03d}.npz'
            with np.load(ROOT / 'r1' / split / name) as z:
                a = {k: z[k] for k in z.files}
            with np.load(ROOT / 'r2' / split / name) as z:
                b = {k: z[k] for k in z.files}
            bp = (b['points'] - np.array([2, 0])) / 1.25
            distance = np.linalg.norm(a['points'][0, :, None] - bp[0, None], axis=2)
            keep = distance.min(1) < 0.0001
            common = a['points'][:, keep]
            assert len(common[0]) >= 8
            a['points'] = common.astype(np.float32)
            b['points'] = (common * 1.25 + np.array([2, 0])).astype(np.float32)
            np.savez_compressed(ROOT / 'r1' / split / name, **a)
            np.savez_compressed(ROOT / 'r2' / split / name, **b)
            audit.append({'split': split, 'index': i, 'points': int(keep.sum())})
    put(ROOT / 'correspondence_audit.json', {'status': 'PASS', 'canonical_coordinates_identical': True, 'rows': audit})
