"""Retained recipe preprocessing, with a reproducible fresh warp per training draw."""
import numpy as np
from spk210.runtime import cv2
from spk210.metrics import project
SEED=20260919

def make_pair(row,draw):
    w,h,scale,left=184,320,1.25,2
    image = cv2.imread(row['path'], 0)
    (ih, iw) = image.shape
    factor = min(288 / iw, 512 / ih)
    (nw, nh) = (round(iw * factor), round(ih * factor))
    (x0, y0) = ((288 - nw) // 2, (512 - nh) // 2)
    common = np.zeros((512, 288), np.uint8)
    mask = np.zeros_like(common)
    common[y0:y0 + nh, x0:x0 + nw] = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA)
    mask[y0:y0 + nh, x0:x0 + nw] = 1
    rng = np.random.default_rng(SEED + draw)
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
    return {'pair':pair,'points':np.stack([q[keep][:64],tq[keep][:64]]).astype(np.float32),'masks':eroded,'cell':cell,'H':tr}
