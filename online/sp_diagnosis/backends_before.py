"""CPU adapters; kmodel requires a real board transport, never fake INT8."""
import hashlib,sys
from pathlib import Path
import numpy as np

def identity(path,prefix):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return prefix+':'+h.hexdigest()

def tensor(rgb):
 from PIL import Image
 # The caller provides independent global-branch RGB240.
 a=np.asarray(Image.fromarray(rgb),np.float32)/255.
 return ((a-np.array([.485,.456,.406],np.float32))/np.array([.229,.224,.225],np.float32)).transpose(2,0,1)[None].copy()

class TorchStudent:
 def __init__(self,path):
  import torch
  torch.set_num_threads(1)
  sys.path.insert(0,'/home/quyet/k210_lab/experiments/edge_vpr_students')
  import success_first_train_core as core
  self.model=core.build_student('B_C160_H96').cpu().eval()
  self.model.load_state_dict(torch.load(path,map_location='cpu',weights_only=False)['model_state_dict'],strict=True)
  self.identity=identity(path,'student-fp32')
 def infer(self,rgb):
  import torch
  with torch.inference_mode():return self.model.raw(torch.from_numpy(tensor(rgb))).numpy().reshape(512)

class OnnxStudent:
 def __init__(self,path):
  import onnxruntime as ort
  opts=ort.SessionOptions();opts.intra_op_num_threads=1;opts.inter_op_num_threads=1
  self.session=ort.InferenceSession(str(path),sess_options=opts,providers=['CPUExecutionProvider']);self.identity=identity(path,'student-onnx-prel2')
 def infer(self,rgb):return self.session.run(None,{self.session.get_inputs()[0].name:tensor(rgb)})[0].reshape(512)

class BoardStudent:
 def __init__(self,transport,model_identity):self.transport=transport;self.identity=model_identity
 def infer(self,rgb):
  # Transport owns verified board input quantization/output dequantization.
  # No emulation or fake quantization fallback.
  return self.transport.infer_pre_l2(rgb)

class SuperPointCPU:
 def __init__(self,size,max_keypoints):
  import torch
  torch.set_num_threads(1)
  sys.path.insert(0,'/home/quyet/k210_lab/scripts')
  import superpoint_kpu_wrapper as sp
  self.model=sp.load_local_baseline();self.model.conf['max_keypoints']=max_keypoints
  self.identity=identity(sp.CHECKPOINT_PATH,'superpoint')+':'+str(tuple(size))+':nms3-th0.005-border4-fixsamplingFalse'
 def infer(self,gray):
  import torch
  with torch.inference_mode():
   out=self.model({'image':torch.from_numpy(gray.astype(np.float32)/255)[None,None]})
  return out['keypoints'][0].cpu().numpy(),out['descriptors'][0].cpu().numpy().T
