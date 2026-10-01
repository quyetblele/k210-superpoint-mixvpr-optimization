import json,hashlib,traceback,numpy as np,onnx,nncase
from pathlib import Path
R=Path('/home/quyet/k210_lab');p=R/'models/s512_br1_prenorm_core_192x352_inferred.onnx';a=R/'artifacts/edge_vpr_students/s512_br1';a.mkdir(parents=True,exist_ok=True);r={'checker':'NOT RUN','import':'NOT RUN','ptq':'NOT RUN','compile':'NOT RUN','gencode':'NOT RUN'};stage='checker'
try:
 onnx.checker.check_model(onnx.load(p));r['checker']='PASS';stage='import';o=nncase.CompileOptions();o.target='k210';o.quant_type='uint8';o.w_quant_type='uint8';c=nncase.Compiler(o);c.import_onnx(p.read_bytes(),nncase.ImportOptions());r['import']='PASS';stage='ptq';q=nncase.PTQTensorOptions();q.samples_count=8;q.set_tensor_data(np.random.default_rng(20260906).uniform(-1,1,(8,3,352,192)).astype('float32').tobytes());c.use_ptq(q);r['ptq']='PASS';stage='compile';c.compile();r['compile']='PASS';stage='gencode';b=c.gencode_tobytes();k=a/'s512_br1_prenorm_core.kmodel';k.write_bytes(b);r.update(gencode='PASS',kmodel=str(k),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
except Exception:r[stage]='FAIL';r['error']=traceback.format_exc()
(R/'reports/edge_vpr_students/s512_br1_nncase_report.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
