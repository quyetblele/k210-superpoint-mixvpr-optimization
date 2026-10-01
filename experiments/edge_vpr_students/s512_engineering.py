#!/usr/bin/env python3
"""Untrained, structured MixVPR-derived S512 architecture/resource gate."""
from __future__ import annotations
import collections,hashlib,json,sys,traceback
from pathlib import Path
import numpy as np
ROOT=Path('/home/quyet/k210_lab');OUT=ROOT/'reports/edge_vpr_students';MOD=ROOT/'models';ART=ROOT/'artifacts/edge_vpr_students/s512';ART.mkdir(parents=True,exist_ok=True);SEED=20260906
def put(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
S={'S512-Q':{'wh':(192,352),'widths':[24,32,48,72,96],'depth':[1,1,1,1,1],'proj':128,'hidden':96,'mix':2},'S512-B':{'wh':(160,288),'widths':[16,24,40,64,80],'depth':[1,1,1,1,1],'proj':128,'hidden':64,'mix':1},'S512-C':{'wh':(144,256),'widths':[12,20,32,48,64],'depth':[1,1,1,1,1],'proj':128,'hidden':32,'mix':1}}
def build(c):
 import torch,torch.nn as nn,torch.nn.functional as F
 class Mixer(nn.Module):
  def __init__(self,t,h):super().__init__();self.norm=nn.LayerNorm(t);self.fc1=nn.Linear(t,h);self.relu=nn.ReLU();self.fc2=nn.Linear(h,t)
  def forward(self,x):return x+self.fc2(self.relu(self.fc1(self.norm(x))))
 class M(nn.Module):
  def __init__(self):
   super().__init__();ws=c['widths'];layers=[];i=3
   for j,o in enumerate(ws):
    layers += [nn.Conv2d(i,o,3,2 if j==0 else 1,1),nn.ReLU(),nn.Conv2d(o,o,3,1,1),nn.ReLU()]
    if j<len(ws)-1:layers += [nn.MaxPool2d(2,2)]
    i=o
   self.backbone=nn.Sequential(*layers); h,w=c['wh'][1]//32,c['wh'][0]//32;self.tokens=h*w;self.mix=nn.Sequential(*[Mixer(self.tokens,c['hidden']) for _ in range(c['mix'])]);self.proj=nn.Linear(ws[-1],c['proj']);self.row=nn.Linear(self.tokens,4)
  def forward(self,x):
   x=self.backbone(x);x=x.flatten(2);x=self.mix(x);x=self.proj(x.permute(0,2,1)).permute(0,2,1);x=self.row(x);return F.normalize(x.flatten(1),p=2,dim=1)
 return M().eval()
def prepare():
 import torch,onnx,onnxruntime
 out={'seed':SEED,'lineage':'MixVPR-derived: CNN feature map -> spatial token FeatureMixer residual MLP -> channel projection -> row projection -> L2-normalized 512D; not original MixVPR.' ,'candidates':{}}
 for n,c in S.items():
  torch.manual_seed(SEED);m=build(c);h,w=c['wh'][1],c['wh'][0];x=torch.linspace(-1,1,3*h*w).reshape(1,3,h,w);y=m(x);loss=(y*y).mean();m.train();loss=m(x).pow(2).mean();loss.backward();g=sum(float(p.grad.abs().sum()) for p in m.parameters() if p.grad is not None);m.eval()
  p=MOD/f's512_{n.lower().replace("-","_")}_{w}x{h}.onnx';torch.onnx.export(m,x,str(p),opset_version=13,dynamo=False,do_constant_folding=False,input_names=['normalized_rgb'],output_names=['global_descriptor']);o=onnx.load(str(p));onnx.checker.check_model(o);inf=onnx.shape_inference.infer_shapes(o,strict_mode=True);ip=p.with_name(p.stem+'_inferred.onnx');onnx.save(inf,str(ip))
  ort=onnxruntime.InferenceSession(str(ip),providers=['CPUExecutionProvider']).run(None,{'normalized_rgb':x.numpy()})[0];a=y.detach().numpy();params=lambda z:sum(q.numel() for q in z.parameters());ops=dict(collections.Counter(q.op_type for q in inf.graph.node));
  # Exact Conv/Linear MAC accounting from module geometry.
  mac=0;events=[];hs=[]
  def hk(name):
   def f(mod,ins,o):
    nonlocal mac
    if hasattr(o,'numel'):events.append({'name':name,'shape':list(o.shape),'bytes':o.numel()*4})
    if mod.__class__.__name__=='Conv2d':mac+=o.shape[1]*o.shape[2]*o.shape[3]*(mod.in_channels//mod.groups)*mod.kernel_size[0]*mod.kernel_size[1]
    if mod.__class__.__name__=='Linear':mac+=int(np.prod(o.shape))*mod.in_features
   return f
  for q,z in m.named_modules():
   if q and not list(z.children()):hs.append(z.register_forward_hook(hk(q)))
  m(x);[z.remove() for z in hs];largest=max(events,key=lambda z:z['bytes']);back=params(m.backbone);mix=params(m.mix);proj=params(m.proj);row=params(m.row)
  out['candidates'][n]={'input_wh':[w,h],'input_shape':[1,3,h,w],'backbone_widths':c['widths'],'backbone_output_shape':[1,c['widths'][-1],h//32,w//32],'tokens':m.tokens,'mixer_depth':c['mix'],'mixer_hidden':c['hidden'],'projection_width':c['proj'],'descriptor_dim':512,'parameters':params(m),'parameter_breakdown':{'backbone':back,'mixer':mix,'projection':proj,'output_row':row},'macs_exact_conv_linear':int(mac),'largest_logical_activation':largest,'forward':{'shape':list(y.shape),'finite':bool(torch.isfinite(y).all()),'norm':float(torch.linalg.norm(y))},'backward':{'finite_loss':bool(torch.isfinite(loss)),'gradient_l1':g},'onnx':{'path':str(ip),'sha256':sha(ip),'nodes':len(inf.graph.node),'operators':ops,'ort_max_abs':float(np.max(abs(a-ort))),'ort_mean_abs':float(np.mean(abs(a-ort))),'ort_cosine':float(a.reshape(-1)@ort.reshape(-1)/(np.linalg.norm(a)*np.linalg.norm(ort)))} }
 put(OUT/'s512_engineering_manifest.json',out);print('S512 PREPARE PASS')
def compiler(name):
 import _nncase,nncase,onnx
 from importlib.metadata import version
 m=json.loads((OUT/'s512_engineering_manifest.json').read_text())['candidates'][name];p=Path(m['onnx']['path']);shape=m['input_shape'];r={'name':name,'checker':'NOT RUN','import':'NOT RUN','ptq':'NOT RUN','compile':'NOT RUN','gencode':'NOT RUN'};stage='checker'
 try:
  assert version('nncase')=='1.8.0.20220929' and _nncase.__version__=='1.8.0-55be52f';onnx.checker.check_model(onnx.load(str(p)));r['checker']='PASS';stage='import';o=nncase.CompileOptions();o.target='k210';o.quant_type='uint8';o.w_quant_type='uint8';c=nncase.Compiler(o);c.import_onnx(p.read_bytes(),nncase.ImportOptions());r['import']='PASS';stage='ptq';a=np.random.default_rng(SEED).uniform(-1,1,size=(8,*shape[1:])).astype('float32');q=nncase.PTQTensorOptions();q.samples_count=8;q.set_tensor_data(a.tobytes());c.use_ptq(q);r['ptq']='PASS';stage='compile';c.compile();r['compile']='PASS';stage='gencode';b=c.gencode_tobytes();k=ART/f'{name}.kmodel';k.write_bytes(b);r.update(gencode='PASS',kmodel_bytes=len(b),sha256=sha(k))
 except Exception:r[stage]='FAIL';r['error']=traceback.format_exc()
 put(OUT/f's512_{name.lower()}_compiler.json',r);print(name,r['gencode']);return 0 if r['gencode']=='PASS' else 1
if __name__=='__main__':
 if len(sys.argv)>1 and sys.argv[1]=='compiler':raise SystemExit(compiler(sys.argv[2]))
 prepare()
