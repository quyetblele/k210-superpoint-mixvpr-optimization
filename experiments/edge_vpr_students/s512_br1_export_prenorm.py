import sys,json,hashlib,numpy as np,torch,onnx,onnxruntime
from pathlib import Path
R=Path('/home/quyet/k210_lab');sys.path.insert(0,str(R/'experiments/edge_vpr_students'));from s512_br1_engineering import BR1
class Core(torch.nn.Module):
 def __init__(self,m):super().__init__();self.m=m
 def forward(self,x):return self.m.raw(x)
torch.manual_seed(20260906);m=BR1().eval();c=Core(m).eval();p=R/'models/s512_br1_prenorm_core_192x352.onnx';ip=R/'models/s512_br1_prenorm_core_192x352_inferred.onnx';x=torch.zeros(1,3,352,192);torch.onnx.export(c,x,str(p),opset_version=13,dynamo=False,input_names=['normalized_rgb'],output_names=['raw_descriptor']);q=onnx.load(p);onnx.checker.check_model(q);q=onnx.shape_inference.infer_shapes(q,strict_mode=True);onnx.save(q,ip);s=onnxruntime.InferenceSession(str(ip));rows=[]
for n,x in [('linspace',torch.linspace(-1,1,3*352*192).reshape(1,3,352,192)),('random',torch.from_numpy(np.random.default_rng(20260906).normal(size=(1,3,352,192)).astype('float32'))),('zeros',torch.zeros(1,3,352,192))]:
 with torch.no_grad():a=m(x).numpy()
 z=s.run(None,{'normalized_rgb':x.numpy()})[0];b=z/np.maximum(np.linalg.norm(z,axis=1,keepdims=True),1e-12);rows.append({'case':n,'raw_shape':list(z.shape),'raw_norm':float(np.linalg.norm(z)),'final_norm':float(np.linalg.norm(b)),'max_abs':float(np.max(abs(a-b))),'cosine':float(a.ravel()@b.ravel()/(np.linalg.norm(a)*np.linalg.norm(b)))})
r={'phase_a':'PASS','input':[1,3,352,192],'output':[1,512],'onnx':str(ip),'sha256':hashlib.sha256(ip.read_bytes()).hexdigest(),'parity':rows};(R/'reports/edge_vpr_students/s512_br1_prenorm_export.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
