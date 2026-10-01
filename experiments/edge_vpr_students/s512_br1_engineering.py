"""Single frozen S512-BR1 engineering gate; no training."""
from __future__ import annotations
import hashlib,json,sys,traceback
from pathlib import Path
import numpy as np,torch,torch.nn as nn,torch.nn.functional as F,onnx,onnxruntime
R=Path('/home/quyet/k210_lab');O=R/'reports/edge_vpr_students';M=R/'models';A=R/'artifacts/edge_vpr_students/s512_br1';A.mkdir(parents=True,exist_ok=True)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
class Mix(nn.Module):
 def __init__(self):super().__init__();self.n=nn.LayerNorm(66);self.a=nn.Linear(66,128);self.r=nn.ReLU();self.b=nn.Linear(128,66)
 def forward(self,x):return x+self.b(self.r(self.a(self.n(x))))
class BR1(nn.Module):
 def __init__(self):
  super().__init__();ws=[16,24,64,112,144];l=[];i=3
  for j,o in enumerate(ws):
   l += [nn.Conv2d(i,o,3,2 if j==0 else 1,1),nn.ReLU(),nn.Conv2d(o,o,3,1,1),nn.ReLU()]
   if j<4:l += [nn.MaxPool2d(2,2)]
   i=o
  self.backbone=nn.Sequential(*l);self.mix=nn.Sequential(Mix(),Mix());self.proj=nn.Linear(144,128);self.row=nn.Linear(66,4)
 def raw(self,x):x=self.backbone(x).flatten(2);x=self.mix(x);x=self.proj(x.permute(0,2,1)).permute(0,2,1);return self.row(x).flatten(1)
 def forward(self,x):return F.normalize(self.raw(x),p=2,dim=1)
def main():
 torch.manual_seed(20260906);m=BR1();x=torch.linspace(-1,1,3*352*192).reshape(1,3,352,192);y=m(x);loss=y.square().mean();loss.backward();g=sum(float(p.grad.abs().sum()) for p in m.parameters() if p.grad is not None);p=M/'s512_br1_192x352.onnx';torch.onnx.export(m,x,str(p),opset_version=13,dynamo=False,input_names=['normalized_rgb'],output_names=['global_descriptor']);q=onnx.load(p);onnx.checker.check_model(q);ip=M/'s512_br1_192x352_inferred.onnx';onnx.save(onnx.shape_inference.infer_shapes(q,strict_mode=True),ip);z=onnxruntime.InferenceSession(str(ip)).run(None,{'normalized_rgb':x.numpy()})[0];rep={'b_downsample':'stage-1 first Conv3x3 stride=2 plus four MaxPool2d stride=2 = total stride32','architecture':{'widths':[16,24,64,112,144],'input':[1,3,352,192],'backbone_output':[1,144,11,6],'tokens':66,'mixers':'2 x LN66 Linear66-128 ReLU Linear128-66 residual','head':'144->128; row66->4; raw512 then CPU L2 deployment'},'params':sum(v.numel() for v in m.parameters()),'forward':{'shape':list(y.shape),'finite':bool(torch.isfinite(y).all()),'norm':float(y.norm()),'grad_l1':g},'onnx':{'path':str(ip),'checker':'PASS','ort_max_abs':float(np.max(abs(y.detach().numpy()-z))),'cosine':float(y.detach().numpy().ravel()@z.ravel()/(np.linalg.norm(y.detach().numpy())*np.linalg.norm(z)))}}
 O.joinpath('s512_br1_engineering.json').write_text(json.dumps(rep,indent=2)+'\n');print(json.dumps(rep,indent=2))
if __name__=='__main__':main()
