"""Experimental exact-score tie ordering; no change to graph or unequal-score ranks."""
import json,time
import numpy as np
from .runtime import torch,cv2,ops
import torch.nn.functional as F
from .settings import ROOT,RES,SUPERPOINT
from .io import sha,put
from .protocol import protocol
from .metrics import evaluate,features
from .onnx_export import load_model,deployment_dir

def order_exact_ties(scores,strength,yx):
    # np.lexsort uses the last key as primary. Corner strength cannot outrank a higher model score.
    return np.lexsort((yx[:,1],yx[:,0],-strength,-scores))

def corner_tie_features(a,b,masks,res,images):
    w,h,scale,left=RES[res]
    scores=a.softmax(1)[:,:64];n,_,hh,ww=scores.shape
    scores=scores.permute(0,2,3,1).reshape(n,hh,ww,8,8).permute(0,1,3,2,4).reshape(n,h,w)
    scores=ops.simple_nms(scores*torch.from_numpy(masks),round(3*scale));result=[]
    for k in range(n):
        yx=torch.nonzero(scores[k]>.005);values=scores[k][tuple(yx.t())].numpy();coords=yx.numpy()
        corner=cv2.cornerMinEigenVal(np.ascontiguousarray(images[k,0],dtype=np.float32),blockSize=3,ksize=3)
        strength=corner[coords[:,0],coords[:,1]]
        indices=order_exact_ties(values,strength,coords)[:160] if len(values)>160 else np.arange(len(values))
        assert len(values)<=160 or np.all(np.diff(values[indices])<=0),'Unequal score ordering changed'
        xy=yx[torch.from_numpy(indices.copy())].flip([1]).float()
        desc=ops.sample_descriptors(xy[None],F.normalize(b[k:k+1],dim=1),8)[0].T.numpy()
        result.append(((xy.numpy()-np.array([left,0]))/scale,desc))
    return result

def run(candidate):
    _,res,checkpoint=load_model(candidate);deployment=deployment_dir(candidate)
    dest=SUPERPOINT/'artifacts/tie_breaking'/candidate;dest.mkdir(parents=True,exist_ok=True)
    simulated=json.loads((deployment/'simulation.json').read_text())
    assert simulated['checkpoint_sha256']==sha(checkpoint) and simulated['kmodel_sha256']==sha(deployment/'model.kmodel')
    p={'purpose':'Test exact-score tie ordering after unchanged INT8 detector NMS',
       'candidate':candidate,'checkpoint_sha256':sha(checkpoint),'kmodel_sha256':sha(deployment/'model.kmodel'),
       'control':'current torch top_k_keypoints, threshold.005, NMS round(3*scale),cap160',
       'treatment':'same NMS and threshold; rank by model score, then image cornerMinEigenVal(block3,ksize3), then y,x. Only exact model-score ties may change ordering.',
       'data_protocol_sha256':sha(ROOT/'protocol.json'),'code_sha256':sha(__file__),
       'gate':'correct matches>=control*1.10 AND precision>=control-0.02 AND coverage>=control-0.02',
       'scope':'one engineering DEV postprocessing diagnostic using actual kmodel outputs, no training or GT1; new postprocessing identity, not automatic promotion',
       'board_latency':'NOT_MEASURED'}
    path=dest/'protocol.json'
    if path.exists():assert json.loads(path.read_text())==p
    else:put(path,p)
    # A strong corner must not overtake even a slightly higher unequal model score.
    test=order_exact_ties(np.array([.2,.3,.3]),np.array([100.,1.,2.]),np.array([[0,0],[0,1],[0,2]]))
    assert test.tolist()==[2,1,0]
    for row in simulated['rows']:
        assert sha(ROOT/res/'dev'/f"{row['index']:03d}.npz")==row['input_sha256']
        assert sha(deployment/'sim_outputs'/f"{row['index']:03d}.npz")==row['output_sha256']
    class ActualOutputs(torch.nn.Module):
        def __init__(self):super().__init__();self.index=0
        def forward(self,x):
            with np.load(deployment/'sim_outputs'/f'{self.index:03d}.npz') as z:out=tuple(torch.from_numpy(z[k].copy()) for k in ['logits','desc'])
            self.index+=1;return out
    timings={'control':[],'corner_ties':[]}
    def timed(fn,label):
        def call(a,b,masks,res,images):
            start=time.perf_counter();out=fn(a,b,masks,res,images);timings[label].append((time.perf_counter()-start)*1000/len(images));return out
        return call
    control=evaluate(ActualOutputs(),res,protocol()['rows']['dev'],timed(lambda a,b,m,r,x:features(a,b,m,r),'control'))
    previous=json.loads((deployment/'int8_quality.json').read_text())['int8'];assert control==previous,'Control evaluator did not reproduce actual INT8 metrics'
    treatment=evaluate(ActualOutputs(),res,protocol()['rows']['dev'],timed(corner_tie_features,'corner_ties'))
    c=control['macro'];t=treatment['macro'];passed=t['correct_mnn_count']>=c['correct_mnn_count']*1.1 and t['mnn_precision']>=c['mnn_precision']-.02 and t['coverage']>=c['coverage']-.02
    put(dest/'result.json',{'status':'COMPLETE','gate_pass':passed,'control':control,'corner_ties':treatment,'protocol_sha256':sha(path),
       'rough_postprocess_median_ms_per_image':{k:float(np.median(v)) for k,v in timings.items()},
       'timing_scope':'single pass42pairs,includes all feature extraction not just tie sorting; laptop CPU,not formal board benchmark',
       'deployment_status':'experimental CPU postprocessing with actual INT8 outputs; model itself unchanged'})
    print(json.dumps({'candidate':candidate,'gate_pass':passed,'control':c,'corner_ties':t}),flush=True)
