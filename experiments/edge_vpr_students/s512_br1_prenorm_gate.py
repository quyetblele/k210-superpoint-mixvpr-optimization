from pathlib import Path
import sys,json,hashlib,traceback,numpy as np,torch,onnx,onnxruntime
from onnx import helper,TensorProto
R=Path('/home/quyet/k210_lab');M=R/'models';O=R/'reports/edge_vpr_students';A=R/'artifacts/edge_vpr_students/s512_br1';sys.path.insert(0,str(R/'experiments/edge_vpr_students'));from s512_br1_engineering import BR1
src=M/'s512_br1_192x352_inferred.onnx';dst=M/'s512_br1_prenorm_core_192x352.onnx';inf=M/'s512_br1_prenorm_core_192x352_inferred.onnx'
m=onnx.load(src);g=m.graph;flat=[n for n in g.node if n.op_type=='Flatten'][-1];raw=flat.output[0];keep=[];seen=False
for n in g.node:
 if n is flat:seen=True;keep.append(n)
 elif not seen:keep.append(n)
del g.node[:];g.node.extend(keep);del g.output[:];g.output.extend([helper.make_tensor_value_info(raw,TensorProto.FLOAT,[1,512])]);onnx.checker.check_model(m);onnx.save(m,dst);q=onnx.shape_inference.infer_shapes(m,strict_mode=True);onnx.save(q,inf)
torch.manual_seed(20260906);pt=BR1().eval();sess=onnxruntime.InferenceSession(str(inf));rows=[]
for n,x in [('linspace',torch.linspace(-1,1,3*352*192).reshape(1,3,352,192)),('random',torch.from_numpy(np.random.default_rng(20260906).normal(size=(1,3,352,192)).astype('float32'))),('zeros',torch.zeros(1,3,352,192))]:
 with torch.no_grad():a=pt(x).numpy()
 z=sess.run(None,{'normalized_rgb':x.numpy()})[0];b=z/np.maximum(np.linalg.norm(z,axis=1,keepdims=True),1e-12);rows.append({'case':n,'raw_norm':float(np.linalg.norm(z)),'final_norm':float(np.linalg.norm(b)),'max_abs':float(np.max(abs(a-b))),'cosine':float(a.ravel()@b.ravel()/(np.linalg.norm(a)*np.linalg.norm(b)))})
r={'checker':'PASS','output':[1,512],'parity':rows,'import':'NOT RUN','ptq':'NOT RUN','compile':'NOT RUN','gencode':'NOT RUN'}
try:
 import nncase,_nncase
 c=nncase.Compiler(nncase.CompileOptions(target='k210',quant_type='uint8',w_quant_type='uint8'));c.import_onnx(inf.read_bytes(),nncase.ImportOptions());r['import']='PASS';p=nncase.PTQTensorOptions();p.samples_count=8;p.set_tensor_data(np.random.default_rng(20260906).uniform(-1,1,(8,3,352,192)).astype('float32').tobytes());c.use_ptq(p);r['ptq']='PASS';c.compile();r['compile']='PASS';b=c.gencode_tobytes();k=A/'s512_br1_prenorm_core.kmodel';k.write_bytes(b);r.update(gencode='PASS',kmodel=str(k),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
except Exception:r['error']=traceback.format_exc()
O.joinpath('s512_br1_nncase_report.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
