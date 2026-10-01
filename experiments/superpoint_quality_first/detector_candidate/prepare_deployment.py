import os
os.environ.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
import json,sys,hashlib,time,resource
from pathlib import Path
import numpy as np,torch,onnx,onnxruntime as ort
from diagnose import BASE,p,digest
OUT=Path(__file__).parent
def write(name,x):
    t=OUT/(name+'.tmp');t.write_text(json.dumps(x,indent=2)+'\n');t.replace(OUT/name)

def main():
    diag=json.loads((OUT/'diagnosis.json').read_text());identity=json.loads((OUT/'protocol.json').read_text())
    branch=diag['selected_existing_pilot'];ident=identity['checkpoint_identities'][branch]
    assert digest(ident['path'])==ident['sha256']
    model=p.w0.V1FR12().eval();ck=torch.load(ident['path'],map_location='cpu',weights_only=False);model.load_state_dict(ck['model'])
    raw=OUT/'pilot_raw.onnx';canonical=OUT/'pilot_canonical.onnx'
    torch.onnx.export(model,torch.zeros(1,1,p.H,p.W),str(raw),opset_version=13,input_names=['image'],output_names=['detector_logits','descriptor_map'],dynamo=False,do_constant_folding=False)
    graph=onnx.load(raw);onnx.checker.check_model(graph)
    graph=onnx.shape_inference.infer_shapes(graph,strict_mode=True);onnx.checker.check_model(graph)
    assert set(n.op_type for n in graph.graph.node)<={'Conv','Relu','MaxPool'}
    onnx.save(graph,canonical)
    so=ort.SessionOptions();so.intra_op_num_threads=2;so.inter_op_num_threads=1
    session=ort.InferenceSession(str(canonical),so,providers=['CPUExecutionProvider'])
    prot=json.loads((BASE/'protocol.json').read_text());selected=[]
    for scene in sorted({r['scene'] for r in prot['TRAIN']}):
        ids=[i for i,r in enumerate(prot['TRAIN']) if r['scene']==scene]
        selected.extend(ids[int(j)] for j in np.linspace(0,len(ids)-1,4,dtype=int))
    cal=[];manifest=[];parity=[]
    with torch.inference_mode():
        for i in selected:
            row=prot['TRAIN'][i];assert digest(row['path'])==row['sha256']
            path=BASE/'train'/('%03d.npz'%i)
            with np.load(path) as z:x=z['pair'][0:1].copy()
            cal.append(x[0]);manifest.append({'index':i,'scene':row['scene'],'source_path':row['path'],'source_sha256':row['sha256'],'cache_sha256':digest(path)})
            if len(parity)<8:
                a=model(torch.from_numpy(x));b=session.run(None,{'image':x});diff=[]
                for ta,ob in zip(a,b):
                    aa=ta.numpy();err=float(np.max(np.abs(aa-ob)));assert np.allclose(aa,ob,atol=2e-4,rtol=2e-4),err
                    diff.append(err)
                parity.append({'train_index':i,'max_abs_each_head':diff})
    assert not {r['source_sha256'] for r in manifest}&{r['sha256'] for r in prot['DEV']}
    np.save(OUT/'calibration_train.npy',np.stack(cal).astype(np.float32))
    write('calibration.json',{'split':'TRAIN','count':len(cal),'selection':'four evenly spaced existing TRAIN items per scene','preprocessing':'cached OpenCV grayscale INTER_AREA resize W256 H192, float32 /255; original view only','manifest':manifest,'tensor_sha256':digest(OUT/'calibration_train.npy')})
    write('export.json',{'status':'PASS','selected_branch':branch,'checkpoint':ident,'input':[1,1,p.H,p.W],'outputs':[[1,65,p.H//8,p.W//8],[1,256,p.H//8,p.W//8]],'raw_onnx_sha256':digest(raw),'canonical_onnx_sha256':digest(canonical),'canonicalization':'shape inference/checker; native Conv/ReLU/MaxPool graph needs no algebraic rewrite','parity':parity,'CPU_postprocessing':'softmax, dustbin removal, pixel decode, NMS3, threshold0.005, border4, top160, original descriptor sampling fix_sampling=False, L2','status_scope':'existing pilot weights, not final deployment-quality candidate','rss_peak_MiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024})
    print('EXPORT_PARITY_PASS',branch,len(parity),'CALIBRATION_TRAIN',len(cal),flush=True)
if __name__=='__main__':main()
