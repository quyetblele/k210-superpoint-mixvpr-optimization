import sys,torch,torch.nn as nn,torch.nn.functional as F,onnx,numpy as np,onnxruntime,json,hashlib
from pathlib import Path
R=Path('/home/quyet/k210_lab');M=R/'models';O=R/'reports/edge_vpr_students'
class X(nn.Module):
 def __init__(self,t=264):super().__init__();self.n=nn.LayerNorm(t);self.a=nn.Linear(t,160);self.b=nn.Linear(160,t)
 def forward(self,x):return x+self.b(F.relu(self.a(self.n(x))))
class BR2M(nn.Module):
 def __init__(self):
  super().__init__();ws=[16,24,64,112,144];l=[];i=3
  for j,o in enumerate(ws):
   l += [nn.Conv2d(i,o,3,2 if j==0 else 1,1),nn.ReLU(),nn.Conv2d(o,o,3,1,1),nn.ReLU()]
   if j<3:l += [nn.MaxPool2d(2,2)]
   i=o
  self.backbone=nn.Sequential(*l);self.mix=nn.Sequential(*[X() for _ in range(4)]);self.proj=nn.Linear(144,128);self.row=nn.Linear(264,4)
 def raw(self,x):x=self.backbone(x).flatten(2);x=self.mix(x);x=self.proj(x.permute(0,2,1)).permute(0,2,1);return self.row(x).flatten(1)
 def forward(self,x):return F.normalize(self.raw(x),p=2,dim=1)
torch.manual_seed(20260906);m=BR2M();x=torch.linspace(-1,1,3*352*192).reshape(1,3,352,192);y=m(x);(y.square().mean()).backward();p=M/'s512_br2m_192x352.onnx';torch.onnx.export(m,x,str(p),opset_version=13,dynamo=False,input_names=['normalized_rgb'],output_names=['global_descriptor']);q=onnx.shape_inference.infer_shapes(onnx.load(p),strict_mode=True);onnx.checker.check_model(q);ip=M/'s512_br2m_192x352_inferred.onnx';onnx.save(q,ip);z=onnxruntime.InferenceSession(str(ip)).run(None,{'normalized_rgb':x.numpy()})[0];r={'params':sum(a.numel() for a in m.parameters()),'input':[1,3,352,192],'output':list(y.shape),'backbone':[1,144,22,12],'tokens':264,'ort_max':float(np.max(abs(y.detach().numpy()-z))),'cos':float(y.detach().numpy().ravel()@z.ravel()/(np.linalg.norm(y.detach().numpy())*np.linalg.norm(z)))};O.joinpath('s512_br2m_engineering.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
