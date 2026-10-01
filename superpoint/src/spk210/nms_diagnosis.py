"""Bounded DEV diagnostic: greedy suppression on faithful cached kmodel outputs."""
import json
import numpy as np
from .runtime import torch, cv2, ops
import torch.nn.functional as F
from .settings import ROOT, RES, SUPERPOINT
from .io import sha, put
from .protocol import protocol
from .metrics import evaluate
from .onnx_export import load_model, deployment_dir

def greedy(a, b, masks, res, images, corner=False):
    w,h,scale,left=RES[res]
    scores=a.softmax(1)[:,:64]
    n,_,hh,ww=scores.shape
    scores=scores.permute(0,2,3,1).reshape(n,hh,ww,8,8).permute(0,1,3,2,4).reshape(n,h,w).numpy()
    result=[]; radius=round(3*scale)
    for k in range(n):
        y,x=np.nonzero((scores[k]>.005)&(masks[k]>0))
        secondary=cv2.cornerMinEigenVal(np.ascontiguousarray(images[k,0]),3,3)[y,x] if corner else np.zeros(len(y))
        order=np.lexsort((x,y,-secondary,-scores[k,y,x]))
        blocked=np.zeros((h,w),bool); selected=[]
        for i in order:
            yy,xx=y[i],x[i]
            if blocked[yy,xx]: continue
            selected.append((xx,yy))
            blocked[max(0,yy-radius):yy+radius+1,max(0,xx-radius):xx+radius+1]=True
            if len(selected)==160: break
        xy=torch.tensor(selected,dtype=torch.float32).reshape(-1,2)
        desc=ops.sample_descriptors(xy[None],F.normalize(b[k:k+1],dim=1),8)[0].T.numpy()
        result.append(((xy.numpy()-[left,0])/scale,desc))
    return result

def run(candidate='wider_r2_cw'):
    model,res,checkpoint=load_model(candidate); dep=deployment_dir(candidate)
    dest=SUPERPOINT/'artifacts/nms_diagnosis'/candidate;dest.mkdir(parents=True,exist_ok=True)
    sim=json.loads((dep/'simulation.json').read_text())
    assert sim['checkpoint_sha256']==sha(checkpoint) and sim['kmodel_sha256']==sha(dep/'model.kmodel')
    for row in sim['rows']:
        assert sha(ROOT/res/'dev'/f"{row['index']:03d}.npz")==row['input_sha256']
        assert sha(dep/'sim_outputs'/f"{row['index']:03d}.npz")==row['output_sha256']
    spec={'checkpoint_sha256':sha(checkpoint),'kmodel_sha256':sha(dep/'model.kmodel'),'source_sha256':sha(__file__),
          'data_protocol_sha256':sha(ROOT/'protocol.json'),'variants':['original','greedy_score','greedy_corner_exact_ties'],
          'fixed':'threshold .005, cap160, square suppression radius round(3*scale), unchanged descriptor sampling',
          'gate':'INT8 correct >= original*1.10; precision >= original-.02; coverage >= original-.02',
          'scope':'engineering DEV diagnosis; no GT1, no training, no automatic promotion; no board timing'}
    path=dest/'protocol.json'
    if path.exists(): assert json.loads(path.read_text())==spec
    else: put(path,spec)
    class Actual(torch.nn.Module):
        def __init__(self):super().__init__();self.index=0
        def forward(self,x):
            with np.load(dep/'sim_outputs'/f'{self.index:03d}.npz') as z:
                out=tuple(torch.from_numpy(z[k].copy()) for k in ('logits','desc'))
            self.index+=1;return out
    results={}
    for name,fn in [('original',None),('greedy_score',greedy),('greedy_corner_exact_ties',lambda a,b,m,r,x:greedy(a,b,m,r,x,True))]:
        results[name]={'fp32':evaluate(model,res,protocol()['rows']['dev'],fn),'int8':evaluate(Actual(),res,protocol()['rows']['dev'],fn)}
    assert results['original']['int8']==json.loads((dep/'int8_quality.json').read_text())['int8']
    c=results['original']['int8']['macro']
    for name,result in results.items():
        t=result['int8']['macro']
        result['gate_pass']=t['correct_mnn_count']>=c['correct_mnn_count']*1.1 and t['mnn_precision']>=c['mnn_precision']-.02 and t['coverage']>=c['coverage']-.02
    put(dest/'result.json',{'protocol_sha256':sha(path),'results':results})
    print(json.dumps({k:{'fp32':v['fp32']['macro'],'int8':v['int8']['macro'],'gate_pass':v['gate_pass']} for k,v in results.items()}),flush=True)

if __name__=='__main__': run()
