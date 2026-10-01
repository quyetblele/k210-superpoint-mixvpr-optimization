"""FP32 CPU ONNX raw-head adapter with existing SuperPoint postprocessing."""
import json,sys
from pathlib import Path
import numpy as np
import torch
from backends import identity
class SuperPointOnnx:
 def __init__(self,path,size,max_keypoints,threads=2):
  import onnxruntime as ort
  sys.path.insert(0,'/home/quyet/k210_lab/scripts')
  import superpoint_kpu_wrapper as sp
  sp.load_local_baseline() # initialize vendored imports; CPU only
  from SuperGluePretrainedNetwork.models import superpoint as ops
  self.ops=ops;self.cap=max_keypoints;self.size=size
  metadata=json.loads(Path(str(path)+'.json').read_text())
  if metadata['size']!=list(size) or metadata['onnx_identity']!=identity(path,'onnx'):raise ValueError('SuperPoint ONNX identity/shape mismatch')
  self.identity=metadata['local_identity'];options=ort.SessionOptions();options.intra_op_num_threads=threads;options.inter_op_num_threads=1
  self.session=ort.InferenceSession(str(path),sess_options=options,providers=['CPUExecutionProvider'])
 def infer(self,gray):
  if list(gray.shape)!=[self.size[1],self.size[0]]:raise ValueError('SuperPoint input size')
  raw,dense=self.session.run(None,{'gray':gray.astype(np.float32)[None,None]/255})
  with torch.inference_mode():
   score=torch.softmax(torch.from_numpy(raw),1)[:,:-1];b,_,h,w=score.shape
   score=score.permute(0,2,3,1).reshape(b,h,w,8,8).permute(0,1,3,2,4).reshape(b,h*8,w*8)
   score=self.ops.simple_nms(score,3)[0];xy=torch.nonzero(score>.005);value=score[tuple(xy.t())]
   xy,value=self.ops.remove_borders(xy,value,4,h*8,w*8);xy,value=self.ops.top_k_keypoints(xy,value,self.cap)
   order=value.argsort(descending=True);xy=xy[order].flip([1]).float();dense=torch.nn.functional.normalize(torch.from_numpy(dense),dim=1)
   desc=self.ops.sample_descriptors(xy[None],dense,8)[0]
  return xy.numpy(),desc.numpy().T
