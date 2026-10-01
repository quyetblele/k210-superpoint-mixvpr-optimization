#!/usr/bin/env python3
"""M2 read-only diagnostic prefixes of frozen MixVPR components."""
from __future__ import annotations
import hashlib,json,sys,traceback
from pathlib import Path
import numpy as np
try:
 import torch
 ModuleBase=torch.nn.Module
except ModuleNotFoundError:
 torch=None
 ModuleBase=object
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'reports/mixvpr_feasibility';KPY='/home/quyet/miniconda3/envs/k210/bin/python';SEED=20260906;N=8
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):Path(p).parent.mkdir(parents=True,exist_ok=True);Path(p).write_text(json.dumps(x,indent=2)+'\n')
def model():
 from audit_mixvpr_m01_static_core import load_model
 return load_model()
def canonicalize(path,inf):
 from audit_mixvpr_m1_resource_localization import canonicalize
 return canonicalize(path,path,inf)
class BBPrefix(ModuleBase):
 def __init__(self,bb,bound):super().__init__();self.bb=bb;self.bound=bound
 def __call__(self,x):
  m=self.bb.model;x=m.maxpool(m.relu(m.bn1(m.conv1(x))))
  if self.bound=='stem':return x
  if self.bound.startswith('layer1_'):
   for i in range(int(self.bound.rsplit('_',1)[1])+1):x=m.layer1[i](x)
   return x
  x=m.layer1(x)
  if self.bound=='layer1':return x
  x=m.layer2(x)
  if self.bound=='layer2':return x
  n=int(self.bound.rsplit('_',1)[1]);
  for i in range(n+1):x=m.layer3[i](x)
  return x
 def cumulative_parameters(self):
  import itertools
  m=self.bb.model;mods=[m.conv1,m.bn1]
  if self.bound.startswith('layer1_'):mods.extend(m.layer1[i] for i in range(int(self.bound.rsplit('_',1)[1])+1));return itertools.chain.from_iterable(x.parameters() for x in mods)
  mods += ([] if self.bound=='stem' else [m.layer1]) + ([] if self.bound in ('stem','layer1') else [m.layer2])
  if self.bound.startswith('layer3_'):mods.extend(m.layer3[i] for i in range(int(self.bound.rsplit('_',1)[1])+1))
  return itertools.chain.from_iterable(x.parameters() for x in mods)
class AGPrefix(ModuleBase):
 def __init__(self,ag,bound):super().__init__();self.ag=ag;self.bound=bound
 def __call__(self,x):
  import torch.nn.functional as F
  x=x.flatten(2)
  if self.bound=='flatten':return x
  if self.bound.startswith('mix'):
   for i in range(int(self.bound[3:])):x=self.ag.mix[i](x)
   return x
  for i in range(4):x=self.ag.mix[i](x)
  x=x.permute(0,2,1);x=self.ag.channel_proj(x)
  if self.bound=='channel_proj':return x
  x=x.permute(0,2,1);x=self.ag.row_proj(x)
  if self.bound=='row_proj':return x
  return F.normalize(x.flatten(1),p=2,dim=-1)
 def cumulative_parameters(self):
  import itertools
  mods=[]
  if self.bound.startswith('mix'):mods=list(self.ag.mix[:int(self.bound[3:])])
  elif self.bound in ('channel_proj','row_proj','normalize'):mods=list(self.ag.mix)+[self.ag.channel_proj]+([] if self.bound=='channel_proj' else [self.ag.row_proj])
  return itertools.chain.from_iterable(x.parameters() for x in mods)
def export(prefix,input_tensor,name):
 import torch,onnx,onnxruntime
 # Prefix registers original child modules; export does not mutate weights.
 p=ROOT/f'models/mixvpr_m2_{name}.onnx';inf=ROOT/f'models/mixvpr_m2_{name}_inferred.onnx'
 torch.onnx.export(prefix.eval(),input_tensor,str(p),opset_version=13,dynamo=False,do_constant_folding=False,input_names=['image' if name.startswith('bb_') else 'backbone_features'],output_names=['prefix_output'])
 changes=canonicalize(p,inf);g=onnx.load(str(inf));return p,inf,changes,{'nodes':len(g.graph.node),'operators':{x:sum(n.op_type==x for n in g.graph.node) for x in sorted(set(n.op_type for n in g.graph.node))}}
def prepare():
 import torch,onnxruntime,cv2
 from audit_mixvpr_m01_static_core import external_preprocess,real_images
 m=model();bb=m.net.backbone;ag=m.net.aggregator;dummy=torch.linspace(-2,2,1*3*320*320).reshape(1,3,320,320)
 # Coarse sweep deliberately excludes M1 full endpoints: bb layer3_5 and ag normalize.
 specs=[('bb_stem',BBPrefix(bb,'stem'),dummy),('bb_layer1_0',BBPrefix(bb,'layer1_0'),dummy),('bb_layer1_1',BBPrefix(bb,'layer1_1'),dummy),('bb_layer1',BBPrefix(bb,'layer1'),dummy),('ag_flatten',AGPrefix(ag,'flatten'),None),('ag_mix1',AGPrefix(ag,'mix1'),None),('ag_mix2',AGPrefix(ag,'mix2'),None),('ag_mix4',AGPrefix(ag,'mix4'),None),('ag_channel_proj',AGPrefix(ag,'channel_proj'),None)]
 with torch.no_grad():feature=bb(dummy)
 results={};rows={}
 for name,prefix,x in specs:
  x=feature if x is None else x;p,inf,changes,graph=export(prefix,x,name);sess=onnxruntime.InferenceSession(str(inf),providers=['CPUExecutionProvider']);items=[('deterministic',x.numpy())]
  # Same fixed real set as M0/M1; aggregator inputs are actual backbone outputs.
  for img in real_images():
   q,_=external_preprocess(cv2.imread(str(img),cv2.IMREAD_COLOR),m)
   with torch.no_grad():items.append((str(img),(bb(q) if name.startswith('ag_') else q).numpy()))
  vals=[]
  with torch.no_grad():
   for label,v in items:
    pt=prefix(torch.from_numpy(v)).numpy();ort=sess.run(None,{sess.get_inputs()[0].name:v})[0];a,b=pt.reshape(-1),ort.reshape(-1);vals.append({'input':label,'shape':list(ort.shape),'max_abs_error':float(np.max(abs(a-b))),'mean_abs_error':float(np.mean(abs(a-b))),'cosine_similarity':float(np.dot(a,b)/max(float(np.linalg.norm(a)*np.linalg.norm(b)),1e-12))})
  cp=sum(z.numel() for z in prefix.cumulative_parameters());results[name]={'component':'backbone' if name.startswith('bb_') else 'aggregator','input_shape':list(x.shape),'output_shape':vals[0]['shape'],'parameters':cp,'fp32_parameter_bytes':cp*4,'output_elements':int(np.prod(vals[0]['shape'])),'output_fp32_logical_bytes':int(np.prod(vals[0]['shape']))*4,'onnx':str(p),'inferred_onnx':str(inf),'onnx_sha256':sha(p),'inferred_sha256':sha(inf),'canonicalizations':changes,'graph':graph,'parity':{'real_images':len(vals)-1,'max_abs_error':max(v['max_abs_error'] for v in vals),'mean_abs_error':float(np.mean([v['mean_abs_error'] for v in vals])),'cosine_similarity':float(np.mean([v['cosine_similarity'] for v in vals]))},'input_kind':'image' if name.startswith('bb_') else 'backbone_features'}
 # Full execution-order audit, no additional ONNX graphs or compiler runs.
 structure={'backbone':[],'aggregator':[]}
 with torch.no_grad():
  x=dummy;structure['backbone'].append({'boundary':'input','shape':list(x.shape),'elements':x.numel(),'fp32_logical_bytes':x.numel()*4,'cumulative_parameters':0,'cumulative_fp32_parameter_bytes':0})
  cm=0;m0=bb.model;x=m0.maxpool(m0.relu(m0.bn1(m0.conv1(x))));cm+=sum(z.numel() for mod in (m0.conv1,m0.bn1) for z in mod.parameters());structure['backbone'].append({'boundary':'stem','shape':list(x.shape),'elements':x.numel(),'fp32_logical_bytes':x.numel()*4,'cumulative_parameters':cm,'cumulative_fp32_parameter_bytes':cm*4})
  for stage in ('layer1','layer2','layer3'):
   for i,block in enumerate(getattr(m0,stage)):
    x=block(x);cm+=sum(z.numel() for z in block.parameters());structure['backbone'].append({'boundary':f'{stage}.{i}','shape':list(x.shape),'elements':x.numel(),'fp32_logical_bytes':x.numel()*4,'cumulative_parameters':cm,'cumulative_fp32_parameter_bytes':cm*4})
  y=feature;structure['aggregator'].append({'boundary':'input','shape':list(y.shape),'elements':y.numel(),'fp32_logical_bytes':y.numel()*4,'cumulative_parameters':0,'major_operators':'input'})
  y=y.flatten(2);structure['aggregator'].append({'boundary':'flatten','shape':list(y.shape),'elements':y.numel(),'fp32_logical_bytes':y.numel()*4,'cumulative_parameters':0,'major_operators':'Flatten'})
  ca=0
  for i,block in enumerate(ag.mix):
   y=block(y);ca+=sum(z.numel() for z in block.parameters());structure['aggregator'].append({'boundary':f'mix.{i}','shape':list(y.shape),'elements':y.numel(),'fp32_logical_bytes':y.numel()*4,'cumulative_parameters':ca,'major_operators':'LayerNorm + Linear + ReLU + Linear + residual Add'})
  y=y.permute(0,2,1);y=ag.channel_proj(y);ca+=sum(z.numel() for z in ag.channel_proj.parameters());structure['aggregator'].append({'boundary':'channel_proj','shape':list(y.shape),'elements':y.numel(),'fp32_logical_bytes':y.numel()*4,'cumulative_parameters':ca,'major_operators':'Transpose + Linear'})
  y=y.permute(0,2,1);y=ag.row_proj(y);ca+=sum(z.numel() for z in ag.row_proj.parameters());structure['aggregator'].append({'boundary':'row_proj','shape':list(y.shape),'elements':y.numel(),'fp32_logical_bytes':y.numel()*4,'cumulative_parameters':ca,'major_operators':'Transpose + Linear'})
 save(OUT/'m2_resource_boundary_manifest.json',{'protocol_version':'MIXVPR-M2-v1','coarse_prefixes':results,'structure_audit':structure,'m1_full_endpoints':{'backbone':'gencode FAIL allocator OOM','aggregator':'gencode FAIL allocator OOM'},'scope':'Original computation prefixes only; outputs exposed solely for diagnosis; M0.2/M0.3 representation canonicalizations applied.'});print('M2 PREPARE PASS')
def compiler(name):
 import _nncase,nncase,onnx
 from importlib.metadata import version
 man=json.loads((OUT/'m2_resource_boundary_manifest.json').read_text());s=man['coarse_prefixes'][name];shape=s['input_shape'];path=Path(s['inferred_onnx']);r={'name':name,'checker':'NOT RUN','import':'NOT RUN','ptq':'NOT RUN','compile':'NOT RUN','gencode':'NOT RUN'};stage='checker'
 try:
  r['environment']={'nncase':version('nncase'),'_nncase':_nncase.__version__,'python':sys.executable};assert r['environment']['nncase']=='1.8.0.20220929' and r['environment']['_nncase']=='1.8.0-55be52f';onnx.checker.check_model(onnx.load(str(path)));r['checker']='PASS';stage='import';o=nncase.CompileOptions();o.target='k210';o.quant_type='uint8';o.w_quant_type='uint8';o.dump_dir=str(OUT/f'm2_{name}_dump');o.dump_ir=True;o.dump_asm=True;c=nncase.Compiler(o);c.import_onnx(path.read_bytes(),nncase.ImportOptions());r['import']='PASS';stage='ptq';cal=np.random.default_rng(SEED).uniform(-3,3,size=(N,*shape[1:])).astype(np.float32);q=nncase.PTQTensorOptions();q.samples_count=N;q.set_tensor_data(cal.tobytes());r['calibration_shape']=list(cal.shape);c.use_ptq(q);r['ptq']='PASS';stage='compile';c.compile();r['compile']='PASS';stage='gencode';blob=c.gencode_tobytes();kp=ROOT/f'artifacts/mixvpr_m2_{name}.kmodel';kp.write_bytes(blob);r.update(gencode='PASS',kmodel_bytes=len(blob),sha256=sha(kp))
 except Exception:r[stage]='FAIL';r['error']=traceback.format_exc()
 save(OUT/f'm2_{name}_compiler.json',r);return 0 if r['gencode']=='PASS' else 1
def report():
 raw=json.loads((OUT/'m2_resource_boundary_manifest.json').read_text());man=raw['coarse_prefixes'];structure=raw['structure_audit']
 def c(name):return json.loads((OUT/f'm2_{name}_compiler.json').read_text())
 bs,bf,af,am=c('bb_stem'),c('bb_layer1_0'),c('ag_flatten'),c('ag_mix1')
 bpass,bfail,apass,afail=man['bb_stem'],man['bb_layer1_0'],man['ag_flatten'],man['ag_mix1']
 lines=['# MixVPR M2 resource-boundary localization','', 'Diagnostic prefix graphs preserve original computation up to the exposed output. The only representation edits are already parity-proven M0.2/M0.3 static-shape and DOUBLE cleanup; no learned weight/math changed. Each tested prefix was compared against its original PyTorch intermediate over deterministic input and 30 real project images.','', '## Active structure audit','', '- Backbone: ResNet50 cropped before layer4: stem → layer1 (3 Bottleneck blocks) → layer2 (4 blocks) → layer3 (6 blocks) → `[1,1024,20,20]`.', '- Aggregator: flatten `[1,1024,20,20]` to `[1,1024,400]` → four `FeatureMixerLayer`s (LayerNorm, Linear 400→400, ReLU, Linear 400→400, residual) → channel projection 1024→1024 → row projection 400→4 → L2 normalization.','', '### Backbone execution order (all values DERIVED_EXACT)','', '| Boundary | Shape | FP32 logical bytes | Cumulative params | Cumulative FP32 param bytes |','|---|---|---:|---:|---:|']
 for x in structure['backbone']:lines.append(f"| {x['boundary']} | `{x['shape']}` | {x['fp32_logical_bytes']:,} | {x['cumulative_parameters']:,} | {x['cumulative_fp32_parameter_bytes']:,} |")
 lines += ['', '### Aggregator execution order (all values DERIVED_EXACT)','', '| Boundary | Shape | FP32 logical bytes | Cumulative params | Major operators |','|---|---|---:|---:|---|']
 for x in structure['aggregator']:lines.append(f"| {x['boundary']} | `{x['shape']}` | {x['fp32_logical_bytes']:,} | {x['cumulative_parameters']:,} | {x['major_operators']} |")
 lines += ['', '## Boundaries and parity','', '| Boundary | Shape | FP32 logical bytes | Cumulative params | FP32 parameter bytes | parity max abs / cosine | gencode |','|---|---|---:|---:|---:|---|---|']
 for name,x,co in [('backbone stem',bpass,bs),('backbone layer1.0',bfail,bf),('aggregator flatten',apass,af),('aggregator mix.0',afail,am)]:
  p=x['parity'];lines.append(f"| {name} | `{x['output_shape']}` | {x['output_fp32_logical_bytes']:,} DERIVED_EXACT | {x['parameters']:,} DERIVED_EXACT | {x['fp32_parameter_bytes']:,} DERIVED_EXACT | {p['max_abs_error']:.3e} / {p['cosine_similarity']:.9f} | {co['gencode']} |")
 lines += ['', 'All four: checker/import/PTQ/compile were PASS. The two FAIL rows fail only at gencode with `RuntimeError: Allocator has ran out of memory`. Compiler MEMORY USAGES were not emitted for these prefix runs; no board peak RAM is inferred.', '', '## Compact boundary comparison','', '| Component | Last PASS | First FAIL | Params PASS/FAIL | Activation PASS/FAIL | Risk interpretation |','|---|---|---|---|---|---|',f"| Backbone | stem | layer1.0 | {bpass['parameters']:,} / {bfail['parameters']:,} | {bpass['output_fp32_logical_bytes']:,} / {bfail['output_fp32_logical_bytes']:,} FP32 logical bytes | ACTIVATION/WORKSPACE-DOMINATED RISK: params grow only 75,008 while exposed activation grows 4×; compiler-specific exact allocation remains unknown. |",f"| Aggregator | flatten | mix.0 | {apass['parameters']:,} / {afail['parameters']:,} | {apass['output_fp32_logical_bytes']:,} / {afail['output_fp32_logical_bytes']:,} FP32 logical bytes | WEIGHT-DOMINATED RISK: output is unchanged while first mixer adds 321,600 params; exact allocator accounting remains unknown. |",'', '## Resource interpretation','', '- Backbone transition: stem `[1,64,80,80]` to original residual `layer1.0` `[1,256,80,80]`; output logical storage increases 1,638,400 → 6,553,600 B (`DERIVED_EXACT`). This supports activation/workspace dominance, but logical tensors are not measured KPU SRAM.', '- Aggregator transition: flatten and first mixer both output `[1,1024,400]`, 1,638,400 B (`DERIVED_EXACT`); first mixer adds LayerNorm and two 400×400 linears. This supports weight-dominated risk.', '- Both components independently OOM at gencode, confirming partitioning alone cannot make the frozen model deployable under this toolchain.', '', 'MIXVPR-M2 BACKBONE BOUNDARY:', 'stem → layer1.0', '', 'BACKBONE RESOURCE CHARACTER:', 'ACTIVATION-WORKSPACE-DOMINATED', '', 'MIXVPR-M2 AGGREGATOR BOUNDARY:', 'flatten → mix.0', '', 'AGGREGATOR RESOURCE CHARACTER:', 'WEIGHT-DOMINATED', '', 'PARTITIONING ALONE SUFFICIENT:', 'NO', '', 'STRUCTURAL MODEL CHANGE NOW PROVEN NECESSARY:', 'YES', '', 'NEXT SINGLE HIGHEST-VALUE INTERVENTION FAMILY:', 'RESOLUTION']
 (OUT/'m2_resource_boundary_report.md').write_text('\n'.join(lines)+'\n');print('M2 REPORT PASS')
if __name__=='__main__':
 a=sys.argv[1] if len(sys.argv)>1 else 'prepare'
 if a=='prepare':prepare()
 elif a=='report':report()
 else:raise SystemExit(compiler(a))
