"""Untrained, isolated nncase graph probes; no candidate registry or training changes."""
import _bootstrap
import json,sys,subprocess,re
from pathlib import Path
import numpy as np
from spk210.settings import SUPERPOINT,WORKSPACE
from spk210.io import sha,put
OUT=SUPERPOINT/'artifacts/architecture_headroom_probe'
GRAPHS={'control':[24,32,64,96],'deep128':[24,48,96,128],'deep192':[24,64,128,192],'wide192':[32,64,128,192]}
CAL=SUPERPOINT/'artifacts/deployment/rank_control_r2_cw/calibration_train.npy'

def compile_one(name):
    import nncase,_nncase
    from importlib.metadata import version
    assert version('nncase')=='1.8.0.20220929' and _nncase.__version__=='1.8.0-55be52f'
    d=OUT/name;o=nncase.CompileOptions();o.target='k210';o.quant_type='uint8';o.w_quant_type='uint8';o.dump_asm=True;o.dump_dir=str(d/'compiler_dump')
    c=nncase.Compiler(o);c.import_onnx((d/'model.onnx').read_bytes(),nncase.ImportOptions())
    data=np.load(CAL);q=nncase.PTQTensorOptions();q.samples_count=len(data);q.set_tensor_data(data.tobytes());c.use_ptq(q);c.compile();blob=c.gencode_tobytes();assert blob;(d/'model.kmodel').write_bytes(blob)
    inspect_one(name)

def inspect_one(name):
    import nncase
    d=OUT/name;blob=(d/'model.kmodel').read_bytes();data=np.load(CAL)
    cpu=(d/'compiler_dump/stackvm/main/runtime_ops.txt').read_text();kpu='\n'.join(f.read_text() for f in (d/'compiler_dump/k210').rglob('runtime_ops.txt'))
    info=(d/'compiler_dump/kmodel_info.txt').read_text();mem={k:int(v) for k,v in re.findall(r'(input|output|data|MODEL|TOTAL):.*?\((\d+) B\)',info)};assert len(mem)==5
    sim=nncase.Simulator();sim.load_model(blob);sim.set_input_tensor(0,nncase.RuntimeTensor.from_numpy(np.ascontiguousarray(data[0:1])));sim.run();shapes=[]
    for i,shape in enumerate([(1,65,23,40),(1,256,23,40)]):
        a=sim.get_output_tensor(i).to_numpy();assert a.shape==shape and np.isfinite(a).all();shapes.append(list(a.shape))
    put(d/'compile.json',{'status':'COMPILE_AND_SIMULATOR_PASS_UNTRAINED','cpu_conv':cpu.count('[Conv2D]'),'kpu_conv':kpu.count('[KPUConv2D]'),'memory':mem,'kmodel_sha256':sha(d/'model.kmodel'),'calibration_sha256':sha(CAL),'output_shapes':shapes,'board_peak_and_latency':'NOT_MEASURED','quality':'NOT_EVALUATED_RANDOM_WEIGHTS'})

def run():
    import torch,onnx,onnxruntime as ort
    from spk210.model import Model
    from spk210.onnx_export import rotate_model
    OUT.mkdir(exist_ok=True)
    spec={'graphs':GRAPHS,'seed':20260919,'resolution':'same r2 CW90, 1x1x184x320','head_width':128,'descriptor_dim':256,'training':False,'calibration_sha256':sha(CAL),'screening_policy':'All12KPUConv, zeroCPUConv; compiler general total<=3MiB leaves nominal >=3MiB for rest of6MiB bank. This is an engineering screening allowance, not verified full-system reserve. KPU reserve cannot be inferred from general total.','hardware_source':'https://wiki.sipeed.com/ai/zh/deploy/k210.html','source_sha256':sha(__file__)}
    if (OUT/'protocol.json').exists():
        old=json.loads((OUT/'protocol.json').read_text());assert {k:v for k,v in old.items() if k!='source_sha256'}=={k:v for k,v in spec.items() if k!='source_sha256'}
        if old!=spec:put(OUT/'previous_protocol.json',old);put(OUT/'protocol.json',spec)
    else:put(OUT/'protocol.json',spec)
    records=[]
    for name,widths in GRAPHS.items():
        d=OUT/name;d.mkdir(exist_ok=True);torch.manual_seed(20260919);m=rotate_model(Model(widths).eval());x=torch.from_numpy(np.load(CAL)[0:1].copy());params=sum(v.numel() for v in m.parameters());macs=[0]
        def hook(layer,args,out):macs[0]+=out.numel()*layer.in_channels*layer.kernel_size[0]*layer.kernel_size[1]
        handles=[v.register_forward_hook(hook) for v in m.modules() if isinstance(v,torch.nn.Conv2d)]
        with torch.inference_mode():expected=[v.numpy() for v in m(x)]
        for h in handles:h.remove()
        torch.onnx.export(m,x,str(d/'model.onnx'),opset_version=13,dynamo=False,do_constant_folding=False,input_names=['image'],output_names=['detector_logits','descriptor_map']);graph=onnx.shape_inference.infer_shapes(onnx.load(d/'model.onnx'),strict_mode=True);onnx.checker.check_model(graph);onnx.save(graph,d/'model.onnx')
        o=ort.SessionOptions();o.intra_op_num_threads=2;o.inter_op_num_threads=1;s=ort.InferenceSession(str(d/'model.onnx'),o,providers=['CPUExecutionProvider']);actual=s.run(None,{'image':x.numpy()});assert all(np.allclose(a,b,atol=2e-4,rtol=2e-4) for a,b in zip(expected,actual))
        with (d/'compile.log').open('w') as f:r=subprocess.run(['/home/quyet/miniconda3/envs/k210/bin/python',str(Path(__file__).resolve()),'compile',name],stdout=f,stderr=subprocess.STDOUT)
        row={'name':name,'widths':widths,'parameters':params,'MACs':macs[0],'onnx_parity':'PASS','returncode':r.returncode}
        if r.returncode==0:
            row.update(json.loads((d/'compile.json').read_text()));row['general_screen_pass']=row['cpu_conv']==0 and row['kpu_conv']==12 and row['memory']['TOTAL']<=3*1024**2;row['nominal_general_remaining_bytes']=6*1024**2-row['memory']['TOTAL']
        else:row['status']='FAILED';row['error_tail']=(d/'compile.log').read_text()[-2000:]
        records.append(row);put(OUT/'summary.json',{'records':records,'quality_and_full_system_reserve':'NOT_ESTABLISHED'});print(json.dumps(row),flush=True)
if __name__=='__main__':
    if len(sys.argv)>1:
        (inspect_one if sys.argv[1]=='inspect' else compile_one)(sys.argv[2])
    else:run()
