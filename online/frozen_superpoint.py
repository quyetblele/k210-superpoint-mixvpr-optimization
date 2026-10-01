"""Frozen CW90 ONNX + release decoder, with original-frame coordinates."""
import json
import sys
from pathlib import Path
import cv2
import numpy as np
from backends import identity

ROOT = Path('/home/quyet/k210_lab')

class FrozenSuperPoint:
    def __init__(self, manifest=None, threads=2):
        import onnxruntime as ort
        manifest = Path(manifest or ROOT/'superpoint/artifacts/full_training/freeze_manifest.json')
        release = json.loads(manifest.read_text())
        model = ROOT/'superpoint/artifacts/deployment/full_capacity_r2_cw/canonical.onnx'
        for path in (model, Path(release['postprocess']), Path(release['preprocessing'])):
            if identity(path, 'sha').split(':')[1] != release['files'][str(path)]:
                raise ValueError('frozen SuperPoint artifact hash mismatch: '+str(path))
        sys.path.insert(0, str(ROOT/'superpoint/src'))
        from spk210.metrics import features
        self.features = features
        self.identity = identity(model, 'superpoint-frozen-cw-r2')+':greedy160-letterbox-v1'
        options = ort.SessionOptions()
        options.intra_op_num_threads=threads
        options.inter_op_num_threads=1
        self.session=ort.InferenceSession(str(model), options, providers=['CPUExecutionProvider'])
        self.permutation=np.argsort(np.r_[np.rot90(np.arange(64).reshape(8,8),-1).ravel(),64])

    def infer_frame(self, bgr):
        import torch
        gray=cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY)
        h,w=gray.shape
        factor=min(288/w,512/h)
        nw,nh=round(w*factor),round(h*factor)
        x0,y0=(288-nw)//2,(512-nh)//2
        common=np.zeros((512,288),np.uint8)
        mask=np.zeros_like(common)
        common[y0:y0+nh,x0:x0+nw]=cv2.resize(gray,(nw,nh),interpolation=cv2.INTER_AREA)
        mask[y0:y0+nh,x0:x0+nw]=1
        logical=np.zeros((320,184),np.uint8)
        valid=np.zeros_like(logical)
        logical[:,2:182]=cv2.resize(common,(180,320),interpolation=cv2.INTER_AREA)
        valid[:,2:182]=cv2.resize(mask,(180,320),interpolation=cv2.INTER_NEAREST)
        valid=cv2.erode(valid,np.ones((11,11),np.uint8),borderType=cv2.BORDER_CONSTANT,borderValue=0)
        x=np.ascontiguousarray(np.rot90(logical,-1)[None,None],dtype=np.float32)/255
        raw,dense=self.session.run(None,{self.session.get_inputs()[0].name:x})
        raw=np.ascontiguousarray(np.rot90(raw,1,(2,3))[:,self.permutation])
        dense=np.ascontiguousarray(np.rot90(dense,1,(2,3)))
        with torch.inference_mode():
            xy,desc=self.features(torch.from_numpy(raw),torch.from_numpy(dense),valid[None],'r2',nms='greedy')[0]
        # Decoder returns canonical half-common coordinates. Invert both
        # actual rounded resize scales using pixel-center convention.
        xy=(xy*2+.8-np.array([x0,y0]))*np.array([w/nw,h/nh])-.5
        return xy.astype(np.float32),desc
