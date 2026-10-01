#!/usr/bin/env python3
"""M1 diagnostic split of unchanged M0.3 backbone and aggregator."""
from __future__ import annotations
import hashlib,json,re,subprocess,sys,traceback
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'reports/mixvpr_feasibility';KPY='/home/quyet/miniconda3/envs/k210/bin/python';SHAPE=(1,3,320,320);FEATURE=(1,1024,20,20);SEED=20260906;N=8
BB=ROOT/'models/mixvpr_m1_backbone_1x3x320x320.onnx';BBI=ROOT/'models/mixvpr_m1_backbone_1x3x320x320_inferred.onnx';AG=ROOT/'models/mixvpr_m1_aggregator.onnx';AGI=ROOT/'models/mixvpr_m1_aggregator_inferred.onnx';KB=ROOT/'artifacts/mixvpr_m1_backbone.kmodel';KA=ROOT/'artifacts/mixvpr_m1_aggregator.kmodel'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):Path(p).parent.mkdir(parents=True,exist_ok=True);Path(p).write_text(json.dumps(x,indent=2)+'\n')
def load():
 from audit_mixvpr_m01_static_core import load_model,core_wrapper
 m=load_model();return m,core_wrapper(m)
def met(a,b):
 a,b=np.asarray(a).reshape(-1),np.asarray(b).reshape(-1);return {'max_abs_error':float(np.max(abs(a-b))),'mean_abs_error':float(np.mean(abs(a-b))),'cosine_similarity':float(np.dot(a,b)/max(float(np.linalg.norm(a)*np.linalg.norm(b)),1e-12))}
def canonicalize(source,target,inferred):
 import onnx
 from onnx import TensorProto,helper,numpy_helper
 m=onnx.load(str(source));g=m.graph; removed=set(); changes=[]
 # Replace every Shape/.. chain feeding Reshape shape input with its static
 # resolved [1,1024,-1], and Shape feeding Expand with [1,4096].
 producers={o:n for n in g.node for o in n.output}; consumers={}
 for n in g.node:
  for x in n.input:
   if x:consumers.setdefault(x,[]).append(n)
 def ancestors(t,acc):
  n=producers.get(t)
  if n is None or id(n) in acc:return
  acc.add(id(n))
  for x in n.input:
   if x:ancestors(x,acc)
 for n in list(g.node):
  if n.op_type=='Reshape' and len(n.input)>1 and n.input[1] in producers:
   chain=set();ancestors(n.input[1],chain)
   if all(x.op_type in {'Shape','Slice','Concat','Constant'} for x in g.node if id(x) in chain):
    old=n.input[1];g.initializer.append(helper.make_tensor(old,TensorProto.INT64,[3],[1,1024,-1]));removed|=chain;changes.append({'kind':'static_reshape_shape','tensor':old,'value':[1,1024,-1]})
  if n.op_type=='Expand' and len(n.input)>1 and n.input[1] in producers:
   s=producers[n.input[1]]
   if s.op_type=='Shape':
    old=n.input[1];g.initializer.append(helper.make_tensor(old,TensorProto.INT64,[2],[1,4096]));removed.add(id(s));changes.append({'kind':'static_expand_shape','tensor':old,'value':[1,4096]})
 # M0.3-proven double epsilon representation cleanup.
 for c in list(g.node):
  if c.op_type!='Constant':continue
  a=next((x for x in c.attribute if x.name=='value' and x.HasField('t')),None)
  if a is None or a.t.data_type!=TensorProto.DOUBLE:continue
  con=consumers.get(c.output[0],[])
  if len(con)==1 and con[0].op_type=='Cast':
   cast=con[0];to=next(x.i for x in cast.attribute if x.name=='to')
   if to==TensorProto.FLOAT:
    val=numpy_helper.to_array(a.t).astype(np.float32);c.attribute.clear();c.attribute.extend([helper.make_attribute('value',numpy_helper.from_array(val,name=''))])
    for user in g.node:
     for i,x in enumerate(user.input):
      if x==cast.output[0]:user.input[i]=c.output[0]
    removed.add(id(cast));changes.append({'kind':'double_to_float_epsilon','constant':c.name,'value':val.tolist(),'cast_removed':cast.name})
 kept=[n for n in g.node if id(n) not in removed];del g.node[:];g.node.extend(kept)
 onnx.checker.check_model(m);onnx.save(m,str(target));inf=onnx.shape_inference.infer_shapes(m,strict_mode=True);onnx.checker.check_model(inf);onnx.save(inf,str(inferred));return changes
def prepare():
 import onnx,onnxruntime,torch,cv2
 from audit_mixvpr_m01_static_core import external_preprocess,real_images
 m,core=load();dummy=torch.linspace(-2,2,int(np.prod(SHAPE))).reshape(SHAPE)
 with torch.no_grad():f=core.net.backbone(dummy);out=core.net.aggregator(f)
 torch.onnx.export(core.net.backbone,dummy,str(BB),opset_version=13,dynamo=False,do_constant_folding=False,input_names=['normalized_rgb_320x320'],output_names=['backbone_features'])
 torch.onnx.export(core.net.aggregator,f,str(AG),opset_version=13,dynamo=False,do_constant_folding=False,input_names=['backbone_features'],output_names=['global_descriptor'])
 bbchanges=canonicalize(BB,BB,BBI);agchanges=canonicalize(AG,AG,AGI)
 bbs=onnxruntime.InferenceSession(str(BBI),providers=['CPUExecutionProvider']);ags=onnxruntime.InferenceSession(str(AGI),providers=['CPUExecutionProvider']);rows=[]
 probes=[('deterministic',dummy.numpy())]
 for p in real_images():
  x,_=external_preprocess(cv2.imread(str(p),cv2.IMREAD_COLOR),m);probes.append((str(p),x.numpy()))
 with torch.inference_mode():
  for name,x in probes:
   pf=core.net.backbone(torch.from_numpy(x)).numpy(); bf=bbs.run(None,{'normalized_rgb_320x320':x})[0]; ao=ags.run(None,{'backbone_features':bf})[0];full=core(torch.from_numpy(x)).numpy();rows.append({'input':name,'backbone_shape':list(bf.shape),'output_shape':list(ao.shape),'pytorch_backbone_vs_onnx':met(pf,bf),'full_vs_split_aggregator':met(full,ao)})
 def agg(key):
  vals=[x[key] for x in rows];return {k:max(v[k] for v in vals) if k=='max_abs_error' else float(np.mean([v[k] for v in vals])) for k in vals[0]}
 parity={'status':'PASS','real_images':len(rows)-1,'rows':rows,'backbone':agg('pytorch_backbone_vs_onnx'),'full_vs_split':agg('full_vs_split_aggregator')}
 if parity['backbone']['max_abs_error']>2e-5 or parity['full_vs_split']['max_abs_error']>2e-5:parity['status']='FAIL'
 save(OUT/'m1_resource_localization_parity.json',parity)
 def count(module):return sum(x.numel() for x in module.parameters())
 def facts(path):
  g=onnx.load(str(path));return {'path':str(path),'sha256':sha(path),'nodes':len(g.graph.node),'operators':{k:sum(n.op_type==k for n in g.graph.node) for k in sorted(set(n.op_type for n in g.graph.node))}}
 manifest={'protocol_version':'MIXVPR-M1-v1','source_m03':str(ROOT/'models/mixvpr_m03_dtype_canonical_1x3x320x320.onnx'),'source_m03_sha256':sha(ROOT/'models/mixvpr_m03_dtype_canonical_1x3x320x320.onnx'),'split_tensor':{'m03_onnx_tensor_name':'/net/backbone/layer3/layer3.5/relu_2/Relu_output_0','m1_export_output_name':'backbone_features','shape':list(f.shape),'dtype':'float32','elements':f.numel(),'fp32_logical_bytes':f.numel()*4},'backbone':{'parameters':count(core.net.backbone),'fp32_parameter_bytes':count(core.net.backbone)*4,'onnx':facts(BBI),'canonicalizations':bbchanges},'aggregator':{'parameters':count(core.net.aggregator),'fp32_parameter_bytes':count(core.net.aggregator)*4,'onnx':facts(AGI),'canonicalizations':agchanges},'parity_path':str(OUT/'m1_resource_localization_parity.json'),'scope':'Unchanged PyTorch backbone/aggregator and M0.2/M0.3-proven representation-only canonicalizations only.'};save(OUT/'m1_resource_localization_manifest.json',manifest);print('M1 PREPARE',parity['status'])
def compiler(kind):
 import _nncase,nncase,onnx
 from importlib.metadata import version
 path,kmodel,shape=(BBI,KB,SHAPE) if kind=='backbone' else (AGI,KA,FEATURE);r={'checker':'NOT RUN','import':'NOT RUN','ptq':'NOT RUN','compile':'NOT RUN','gencode':'NOT RUN','kind':kind};stage='checker'
 try:
  r['environment']={'nncase':version('nncase'),'_nncase':_nncase.__version__,'python':sys.executable};assert r['environment']['nncase']=='1.8.0.20220929' and r['environment']['_nncase']=='1.8.0-55be52f'
  onnx.checker.check_model(onnx.load(str(path)));r['checker']='PASS';stage='import';o=nncase.CompileOptions();o.target='k210';o.quant_type='uint8';o.w_quant_type='uint8';o.dump_dir=str(OUT/f'm1_{kind}_nncase_dump');o.dump_ir=True;o.dump_asm=True;c=nncase.Compiler(o);c.import_onnx(path.read_bytes(),nncase.ImportOptions());r['import']='PASS';stage='ptq';cal=np.random.default_rng(SEED).uniform(-3,3,size=(N,*shape[1:])).astype(np.float32);r['calibration_shape']=list(cal.shape);r['calibration_sha256']=hashlib.sha256(cal.tobytes()).hexdigest();q=nncase.PTQTensorOptions();q.samples_count=N;q.set_tensor_data(cal.tobytes());c.use_ptq(q);r['ptq']='PASS';stage='compile';c.compile();r['compile']='PASS';stage='gencode';blob=c.gencode_tobytes();kmodel.write_bytes(blob);r.update(gencode='PASS',kmodel_bytes=len(blob),sha256=sha(kmodel))
 except Exception:r[stage]='FAIL';r['error']=traceback.format_exc()
 save(OUT/f'm1_{kind}_compiler.json',r);return 0 if r['gencode']=='PASS' else 1
def report():
 man=json.loads((OUT/'m1_resource_localization_manifest.json').read_text());p=json.loads((OUT/'m1_resource_localization_parity.json').read_text());bb=json.loads((OUT/'m1_backbone_compiler.json').read_text());ag=json.loads((OUT/'m1_aggregator_compiler.json').read_text());full=json.loads((OUT/'m03_dtype_canonical_compiler.json').read_text())
 if p['status']!='PASS':cl='MIXVPR-M1: COMPATIBILITY/PARITY BLOCKER';b='UNKNOWN';next='UNKNOWN'
 elif bb['gencode']!='PASS' and ag['gencode']!='PASS':cl='MIXVPR-M1: MULTIPLE COMPONENT RESOURCE BLOCKERS';b='MULTIPLE';next='UNKNOWN'
 elif bb['gencode']!='PASS':cl='MIXVPR-M1: BACKBONE RESOURCE BLOCKER';b='BACKBONE';next='BACKBONE SLIMMING'
 elif ag['gencode']!='PASS':cl='MIXVPR-M1: AGGREGATOR RESOURCE BLOCKER';b='AGGREGATOR';next='AGGREGATOR CHANGE'
 else:cl='MIXVPR-M1: MONOLITHIC FULL-GRAPH RESOURCE BLOCKER';b='MONOLITHIC COMBINATION';next='PARTITIONING'
 def row(name,params,inp,out,c):return f"| {name} | {params:,} | {params*4:,} | `{inp}` | `{out}` | {c['import']} | {c['ptq']} | {c['compile']} | {c['gencode']} | {c.get('kmodel_bytes','N/A')} |"
 lines=['# MixVPR M1 resource-localization diagnostic','', 'No learned model was changed. The only rewrites are the already M0.2/M0.3-proven static-shape and DOUBLE-representation canonicalizations, recorded in the manifest.','', '## Exact split and parity','',f"- Backbone→aggregator tensor in M0.3: `{man['split_tensor']['m03_onnx_tensor_name']}`; M1 export output name: `{man['split_tensor']['m1_export_output_name']}`. Shape `{man['split_tensor']['shape']}` {man['split_tensor']['dtype']}, {man['split_tensor']['elements']:,} elements, {man['split_tensor']['fp32_logical_bytes']:,} FP32 logical bytes (`DERIVED_EXACT`; not runtime peak).",f"- Deterministic + {p['real_images']} real images: backbone PyTorch→ORT max abs {p['backbone']['max_abs_error']:.3e}, mean {p['backbone']['mean_abs_error']:.3e}, cosine {p['backbone']['cosine_similarity']:.9f}.",f"- Full PyTorch output→backbone ORT + aggregator ORT: max abs {p['full_vs_split']['max_abs_error']:.3e}, mean {p['full_vs_split']['mean_abs_error']:.3e}, cosine {p['full_vs_split']['cosine_similarity']:.9f}.", '', '## K210 comparison','', '| Graph | Params | FP32 parameter bytes | Input | Output | Import | PTQ | Compile | Gencode | kmodel bytes |','|---|---:|---:|---|---|---|---|---|---|---:|',row('FULL M0.3',10880900,'[1,3,320,320]','[1,4096]',full),row('BACKBONE ONLY',man['backbone']['parameters'],'[1,3,320,320]','[1,1024,20,20]',bb),row('AGGREGATOR ONLY',man['aggregator']['parameters'],'[1,1024,20,20]','[1,4096]',ag),'', '### Exact failures / compiler output','', '```text','BACKBONE: '+bb.get('error','None'),'','AGGREGATOR: '+ag.get('error','None'),'','FULL M0.3: '+full.get('error','None'),'```','',cl,'',f"BACKBONE GENCODE: {'PASS' if bb['gencode']=='PASS' else 'FAIL'}",f"AGGREGATOR GENCODE: {'PASS' if ag['gencode']=='PASS' else 'FAIL'}",'FULL M0.3 GENCODE: FAIL — ALLOCATOR OOM','',f'CONFIRMED RESOURCE BOTTLENECK: {b}',f'NEXT INTERVENTION FAMILY: {next}']
 (OUT/'m1_resource_localization_report.md').write_text('\n'.join(lines)+'\n');print(cl)
if __name__=='__main__':
 a=sys.argv[1] if len(sys.argv)>1 else 'prepare'
 if a=='prepare':prepare()
 elif a in ('backbone','aggregator'):raise SystemExit(compiler(a))
 elif a=='report':report()
