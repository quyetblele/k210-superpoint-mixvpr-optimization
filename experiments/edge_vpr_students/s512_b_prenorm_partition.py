#!/usr/bin/env python3
from __future__ import annotations
import hashlib,json,sys,traceback
from pathlib import Path
import numpy as np
ROOT=Path('/home/quyet/k210_lab');OUT=ROOT/'reports/edge_vpr_students';SRC=ROOT/'models/s512_b_double_canonical_160x288_inferred.onnx';DST=ROOT/'models/s512_b_prenorm_core_160x288.onnx';INF=ROOT/'models/s512_b_prenorm_core_160x288_inferred.onnx';ART=ROOT/'artifacts/edge_vpr_students/s512';SEED=20260906
def put(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def norm(z):
 z=np.asarray(z,dtype=np.float32);return z/np.maximum(np.linalg.norm(z,axis=1,keepdims=True),np.float32(1e-12))
def prepare():
 import onnx,onnxruntime,torch
 from onnx import helper,TensorProto
 sys.path.insert(0,str(ROOT/'experiments/edge_vpr_students'));from s512_engineering import S,build
 m=onnx.load(str(SRC));g=m.graph;produ={o:n for n in g.node for o in n.output};pre='/Flatten_output_0'; chain=['/ReduceL2','/Constant_4','/Clip','/Shape_1','/Expand','/Div'];keep=[n for n in g.node if n.name not in chain];del g.node[:];g.node.extend(keep);del g.output[:];g.output.extend([helper.make_tensor_value_info(pre,TensorProto.FLOAT,[1,512])]);vi=[v for v in g.value_info if v.name not in {'/ReduceL2_output_0','/Constant_4_output_0','/Clip_output_0','/Shape_1_output_0','/Expand_output_0','global_descriptor'}];del g.value_info[:];g.value_info.extend(vi);onnx.checker.check_model(m);onnx.save(m,str(DST));inf=onnx.shape_inference.infer_shapes(m,strict_mode=True);onnx.checker.check_model(inf);onnx.save(inf,str(INF))
 torch.manual_seed(SEED);pt=build(S['S512-B']);full=onnxruntime.InferenceSession(str(SRC),providers=['CPUExecutionProvider']);core=onnxruntime.InferenceSession(str(INF),providers=['CPUExecutionProvider']);rows=[]
 for name,x in [('linspace',torch.linspace(-1,1,3*288*160).reshape(1,3,288,160)),('random',torch.from_numpy(np.random.default_rng(SEED).normal(size=(1,3,288,160)).astype('float32'))),('zeros',torch.zeros(1,3,288,160))]:
  with torch.inference_mode():a=pt(x).numpy()
  b=full.run(None,{'normalized_rgb':x.numpy()})[0];z=core.run(None,{'normalized_rgb':x.numpy()})[0];c=norm(z)
  def met(u,v):return {'max_abs':float(np.max(abs(u-v))),'mean_abs':float(np.mean(abs(u-v))),'cosine':float(u.reshape(-1)@v.reshape(-1)/(np.linalg.norm(u)*np.linalg.norm(v)) if np.linalg.norm(u)*np.linalg.norm(v)>0 else 1.0),'norm':float(np.linalg.norm(v))}
  rows.append({'input':name,'raw_shape':list(z.shape),'raw_norm':float(np.linalg.norm(z)),'pytorch_vs_partition':met(a,c),'full_onnx_vs_partition':met(b,c),'normalized_norm':float(np.linalg.norm(c))})
 if max(x['full_onnx_vs_partition']['max_abs'] for x in rows)>1e-6:raise RuntimeError('partition parity fail')
 rep={'boundary':{'tensor':pre,'shape':[1,512],'dtype':'FLOAT','producer':'/Flatten (Flatten) from /row/Add_output_0','cpu_postprocess':'d=z/max(||z||_2,1e-12)'},'removed_nodes':chain,'before_nodes':len(onnx.load(str(SRC)).graph.node),'after_nodes':len(inf.graph.node),'input':[1,3,288,160],'output':[1,512],'parity':rows,'sha256':sha(INF)};put(OUT/'s512_b_prenorm_partition_report.json',rep);print('PRENORM PARITY PASS')
def compiler():
 import _nncase,nncase,onnx
 from importlib.metadata import version
 r={'checker':'NOT RUN','import':'NOT RUN','ptq':'NOT RUN','compile':'NOT RUN','gencode':'NOT RUN','output_quantization_metadata':'NOT EXPOSED BY nncase Python API'};stage='checker'
 try:
  assert version('nncase')=='1.8.0.20220929' and _nncase.__version__=='1.8.0-55be52f';onnx.checker.check_model(onnx.load(str(INF)));r['checker']='PASS';stage='import';o=nncase.CompileOptions();o.target='k210';o.quant_type='uint8';o.w_quant_type='uint8';c=nncase.Compiler(o);c.import_onnx(INF.read_bytes(),nncase.ImportOptions());r['import']='PASS';stage='ptq';a=np.random.default_rng(SEED).uniform(-1,1,size=(8,3,288,160)).astype('float32');q=nncase.PTQTensorOptions();q.samples_count=8;q.set_tensor_data(a.tobytes());c.use_ptq(q);r['ptq']='PASS';stage='compile';c.compile();r['compile']='PASS';stage='gencode';b=c.gencode_tobytes();p=ART/'s512_b_prenorm_core.kmodel';p.write_bytes(b);r.update(gencode='PASS',kmodel_bytes=len(b),sha256=sha(p))
 except Exception:r[stage]='FAIL';r['error']=traceback.format_exc()
 put(OUT/'s512_b_prenorm_partition_compiler.json',r);print('PRENORM NNCASE',r['gencode']);return 0 if r['gencode']=='PASS' else 1
if __name__=='__main__':
 if len(sys.argv)>1:raise SystemExit(compiler())
 prepare()
