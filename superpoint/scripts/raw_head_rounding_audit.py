"""Explain FP32 ONNX raw-head discrepancies with local IEEE rounding bounds."""
import _bootstrap
import copy,json
import numpy as np
import onnx,onnxruntime as ort
from onnx import numpy_helper,TensorProto
from spk210.runtime import torch
import torch.nn.functional as F
from spk210.onnx_export import load_model,deployment_dir,physical_input,logical_outputs,rotate_model
from spk210.protocol import protocol
from spk210.settings import ROOT,SUPERPOINT
from spk210.io import put,sha
NAME='full_capacity_r2_cw';OUT=SUPERPOINT/'artifacts/full_training/parity_resolution'

@torch.inference_mode()
def run():
    OUT.mkdir(exist_ok=True);model,res,checkpoint=load_model(NAME);dest=deployment_dir(NAME);graph=onnx.load(dest/'canonical.onnx');weights={v.name:numpy_helper.to_array(v).copy() for v in graph.graph.initializer};aug=copy.deepcopy(graph);known={v.name:v for v in list(aug.graph.value_info)+list(aug.graph.output)};original_outputs=[v.name for v in aug.graph.output]
    for node in aug.graph.node:
        for name in node.output:
            if name not in original_outputs:aug.graph.output.append(known[name])
    options=ort.SessionOptions();options.intra_op_num_threads=2;options.inter_op_num_threads=1;options.graph_optimization_level=ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session=ort.InferenceSession(aug.SerializeToString(),options,providers=['CPUExecutionProvider']);plain=ort.InferenceSession(str(dest/'canonical.onnx'),options,providers=['CPUExecutionProvider']);oracle=copy.deepcopy(model).double();rot=rotate_model(oracle);records=[]
    for i in np.linspace(0,len(protocol()['rows']['train'])-1,8,dtype=int):
        with np.load(ROOT/res/'train'/f'{i:03d}.npz') as z:x=z['pair'][0:1].copy()
        physical=physical_input(x,res);out=session.run(None,{'image':physical});actual={v.name:a for v,a in zip(session.get_outputs(),out)};actual['image']=physical
        original=plain.run(None,{'image':physical});assert all(np.array_equal(a,actual[n]) for a,n in zip(original,original_outputs)),'Instrumenting graph changed raw outputs'
        layers=[]
        for node in aug.graph.node:
            inp=actual[node.input[0]];a=torch.from_numpy(inp.astype(np.float64));got=actual[node.output[0]].astype(np.float64)
            if node.op_type=='Conv':
                w=weights[node.input[1]];bias=weights[node.input[2]];attrs={v.name:onnx.helper.get_attribute_value(v) for v in node.attribute};pads=attrs.get('pads',[0,0,0,0]);assert pads[:2]==pads[2:];stride=attrs.get('strides',[1,1]);groups=attrs.get('group',1)
                kw={'stride':stride,'padding':pads[:2],'groups':groups};wt=torch.from_numpy(w.astype(np.float64));bt=torch.from_numpy(bias.astype(np.float64));exact=F.conv2d(a,wt,bt,**kw).numpy();magnitude=F.conv2d(a.abs(),wt.abs(),bt.abs(),**kw).numpy()
                n=2*int(np.prod(w.shape[1:]))+1;u=2**-24;gamma=n*u/(1-n*u);bound=gamma*magnitude;error=np.abs(got-exact);assert np.all(error<=bound+1e-12),node.name
                layers.append({'node':node.name,'op':'Conv','n':n,'gamma':gamma,'max_local_error':float(error.max()),'max_rounding_bound':float(bound.max()),'max_error_to_bound_ratio':float((error/(bound+1e-30)).max()),'rounding_bound_pass':True})
            elif node.op_type=='Relu':assert np.array_equal(got,np.maximum(inp,0))
            elif node.op_type=='MaxPool':
                attrs={v.name:onnx.helper.get_attribute_value(v) for v in node.attribute};exact=F.max_pool2d(a,attrs['kernel_shape'],attrs.get('strides'),attrs.get('pads',[0,0])[0]).numpy();assert np.array_equal(got,exact)
            else:raise AssertionError(node.op_type)
        precise=[v.numpy() for v in oracle(torch.from_numpy(x).double())];rp=logical_outputs([v.numpy() for v in rot(torch.from_numpy(physical).double())],res);assert all(np.allclose(a,b,atol=1e-10,rtol=1e-10) for a,b in zip(precise,rp))
        heads=logical_outputs(original,res);bad=[]
        for ref,a in zip(precise,heads):
            ref=ref.astype(np.float32);mask=np.abs(ref-a)>(2e-4+2e-4*np.abs(a));bad.append({'violations':int(mask.sum()),'max_abs':float(np.abs(ref-a).max()),'reference':ref[mask].tolist(),'actual':a[mask].tolist()})
        records.append({'input_sha256':sha(ROOT/res/'train'/f'{i:03d}.npz'),'train_index':int(i),'instrumentation_bitwise_identical':True,'relu_maxpool_exact':True,'rotation_fp64_pass':True,'layers':layers,'strict_raw_heads':bad})
    result={'status':'EXPLAINED_FP32_ACCUMULATION','checkpoint_sha256':sha(checkpoint),'onnx_sha256':sha(dest/'canonical.onnx'),'source_sha256':sha(__file__),'backend':'ORT CPU, graph optimization disabled','formula':'For each Conv evaluated on its actual FP32 inputs: abs(y_fp32-y_fp64_exact)<=gamma_n*(abs(W)*abs(x)+abs(b)); u=2^-24,n=2*Cin*Kh*Kw+1,gamma_n=n*u/(1-n*u). +1e-12 protects FP64 comparison roundoff only. ReLU/MaxPool exact. This is a local rounding explanation, not a tighter global bound or independent model-quality test.','strict_raw_gate':'FAIL_RETAINED','records':records}
    put(OUT/'rounding_audit.json',result);print(result['status'],'layers',sum(len(r['layers']) for r in records),'strict violations',sum(h['violations'] for r in records for h in r['strict_raw_heads']),flush=True)
if __name__=='__main__':run()
