#!/usr/bin/env python3
from __future__ import annotations
import hashlib,json,sys,traceback
from pathlib import Path
import numpy as np
ROOT=Path('/home/quyet/k210_lab');OUT=ROOT/'reports/edge_vpr_students';SRC=ROOT/'models/s512_b_static_canonical_160x288_inferred.onnx';DST=ROOT/'models/s512_b_double_canonical_160x288.onnx';INF=ROOT/'models/s512_b_double_canonical_160x288_inferred.onnx';ART=ROOT/'artifacts/edge_vpr_students/s512';SEED=20260906
def put(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def prepare():
 import onnx,onnxruntime,torch
 from onnx import TensorProto,helper,numpy_helper
 sys.path.insert(0,str(ROOT/'experiments/edge_vpr_students'));from s512_engineering import S,build
 m=onnx.load(str(SRC));g=m.graph; c=next(n for n in g.node if n.name=='/Constant_4');a=next(x for x in c.attribute if x.name=='value');old=numpy_helper.to_array(a.t);c.attribute.clear();c.attribute.extend([helper.make_attribute('value',numpy_helper.from_array(np.asarray(old,dtype=np.float32),name=''))]);cast=next(n for n in g.node if n.name=='/Cast')
 for n in g.node:
  for i,x in enumerate(n.input):
   if x=='/Cast_output_0':n.input[i]='/Constant_4_output_0'
 kept=[n for n in g.node if n.name!='/Cast'];del g.node[:];g.node.extend(kept);vi=[v for v in g.value_info if v.name not in {'/Constant_4_output_0','/Cast_output_0'}];del g.value_info[:];g.value_info.extend(vi);onnx.checker.check_model(m);onnx.save(m,str(DST));inf=onnx.shape_inference.infer_shapes(m,strict_mode=True);onnx.checker.check_model(inf);onnx.save(inf,str(INF))
 torch.manual_seed(SEED);pt=build(S['S512-B']);paths={'clean':ROOT/'models/s512_s512_b_160x288_inferred.onnx','shape':SRC,'double':INF};ss={k:onnxruntime.InferenceSession(str(v),providers=['CPUExecutionProvider']) for k,v in paths.items()};rows=[]
 for name,x in [('linspace',torch.linspace(-1,1,3*288*160).reshape(1,3,288,160)),('random',torch.from_numpy(np.random.default_rng(SEED).normal(size=(1,3,288,160)).astype('float32'))),('zeros',torch.zeros(1,3,288,160))]:
  with torch.inference_mode():p=pt(x).numpy()
  z={k:v.run(None,{'normalized_rgb':x.numpy()})[0] for k,v in ss.items()};met=lambda u,v:{'max_abs':float(np.max(abs(u-v))),'mean_abs':float(np.mean(abs(u-v))),'cosine':float(u.reshape(-1)@v.reshape(-1)/(np.linalg.norm(u)*np.linalg.norm(v))),'norm':float(np.linalg.norm(v))};rows.append({'input':name,'pytorch_vs_double':met(p,z['double']),'clean_vs_double':met(z['clean'],z['double']),'shape_vs_double':met(z['shape'],z['double'])})
 if max(x['shape_vs_double']['max_abs'] for x in rows)>1e-6:raise RuntimeError('parity fail')
 rep={'double_trace':{'node':'/Constant_4','value':float(old),'shape':list(old.shape),'consumer':'/Cast -> /Clip(min)','purpose':'L2-normalization numerical-stability epsilon','activations':'FP32 /ReduceL2 output and /Flatten output','shape_logic':'NO'},'rewrite':{'constant':'DOUBLE 1e-12 -> FLOAT32 1e-12','cast_removed':'/Cast FLOAT conversion','learned_weights_changed':'NO','math_changed':'NO'},'before_nodes':len(onnx.load(str(SRC)).graph.node),'after_nodes':len(inf.graph.node),'input':[1,3,288,160],'output':[1,512],'parity':rows,'sha256':sha(INF)};put(OUT/'s512_b_double_canonical_report.json',rep);print('DOUBLE CANONICAL PARITY PASS')
def compiler():
 import _nncase,nncase,onnx
 from importlib.metadata import version
 r={'checker':'NOT RUN','import':'NOT RUN','ptq':'NOT RUN','compile':'NOT RUN','gencode':'NOT RUN'};stage='checker'
 try:
  assert version('nncase')=='1.8.0.20220929' and _nncase.__version__=='1.8.0-55be52f';onnx.checker.check_model(onnx.load(str(INF)));r['checker']='PASS';stage='import';o=nncase.CompileOptions();o.target='k210';o.quant_type='uint8';o.w_quant_type='uint8';c=nncase.Compiler(o);c.import_onnx(INF.read_bytes(),nncase.ImportOptions());r['import']='PASS';stage='ptq';a=np.random.default_rng(SEED).uniform(-1,1,size=(8,3,288,160)).astype('float32');q=nncase.PTQTensorOptions();q.samples_count=8;q.set_tensor_data(a.tobytes());c.use_ptq(q);r['ptq']='PASS';stage='compile';c.compile();r['compile']='PASS';stage='gencode';b=c.gencode_tobytes();p=ART/'s512_b_double_canonical.kmodel';p.write_bytes(b);r.update(gencode='PASS',kmodel_bytes=len(b),sha256=sha(p))
 except Exception:r[stage]='FAIL';r['error']=traceback.format_exc()
 put(OUT/'s512_b_double_canonical_compiler.json',r);print('DOUBLE NNCASE',r['gencode']);return 0 if r['gencode']=='PASS' else 1
if __name__=='__main__':
 if len(sys.argv)>1:raise SystemExit(compiler())
 prepare()
