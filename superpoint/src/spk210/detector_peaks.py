"""Read-only spatial peak stability audit using actual compiled output tensors."""
import json
import numpy as np
from .runtime import torch
from .settings import ROOT,RES
from .protocol import protocol
from .io import sha,put
from .metrics import features
from .onnx_export import load_model,deployment_dir

def heatmap(logits):
    score=logits.softmax(1)[:,:64];n,_,h,w=score.shape
    return score.permute(0,2,3,1).reshape(n,h,w,8,8).permute(0,1,3,2,4).reshape(n,h*8,w*8).numpy()

@torch.inference_mode()
def run(candidate):
    model,res,checkpoint=load_model(candidate);dest=deployment_dir(candidate)
    simulation=json.loads((dest/'simulation.json').read_text())
    assert simulation['checkpoint_sha256']==sha(checkpoint) and simulation['kmodel_sha256']==sha(dest/'model.kmodel')
    _,_,scale,left=RES[res];radius=round(3*scale)
    margins=[];noise=[];survive=[];quant_margins=[];overlap=[];ties=[];all_points=0
    for row in simulation['rows']:
        i=row['index'];source=ROOT/res/'dev'/f'{i:03d}.npz';output=dest/'sim_outputs'/f'{i:03d}.npz'
        assert sha(source)==row['input_sha256'] and sha(output)==row['output_sha256']
        with np.load(source) as z:x=torch.from_numpy(z['pair']);masks=z['masks'].copy()
        a,b=model(x)
        with np.load(output) as z:qa=torch.from_numpy(z['logits']);qb=torch.from_numpy(z['desc'])
        maps=heatmap(a)*masks;qmaps=heatmap(qa)*masks
        fp=features(a,b,masks,res);q=features(qa,qb,masks,res)
        for j in range(2):
            points=np.rint(fp[j][0]*scale+np.array([left,0])).astype(int)
            qpoints=np.rint(q[j][0]*scale+np.array([left,0])).astype(int)
            all_points+=len(points)
            if len(points) and len(qpoints):
                dist=np.linalg.norm(fp[j][0][:,None]-q[j][0][None],axis=2)
                overlap.extend((dist.min(1)<=1).tolist())
            elif len(points):overlap.extend([False]*len(points))
            if len(qpoints):
                values=qmaps[j,qpoints[:,1],qpoints[:,0]]
                ties.append(1-len(np.unique(values))/len(values))
            for px,py in points:
                patch=maps[j,py-radius:py+radius+1,px-radius:px+radius+1].copy()
                qpatch=qmaps[j,py-radius:py+radius+1,px-radius:px+radius+1].copy()
                if patch.shape!=(2*radius+1,2*radius+1):continue
                value=patch[radius,radius];qvalue=qpatch[radius,radius]
                patch[radius,radius]=-np.inf;qpatch[radius,radius]=-np.inf
                margin=float(value-patch.max())
                # Iterated NMS can select secondary peaks; restrict local-winner analysis explicitly.
                if margin>=0:
                    margins.append(margin);noise.append(abs(float(qvalue-value)));survive.append(bool(qvalue>=qpatch.max()))
                    quant_margins.append(float(qvalue-qpatch.max()))
    result={'status':'COMPLETE','checkpoint_sha256':sha(checkpoint),'kmodel_sha256':sha(dest/'model.kmodel'),
        'selected_fp32_points':all_points,'strict_local_peak_subset':len(margins),
        'median_fp32_peak_margin':float(np.median(margins)),
        'median_quant_score_change_at_fp32_peaks':float(np.median(noise)),
        'quant_change_exceeds_fp32_margin_fraction':float(np.mean(np.array(noise)>np.array(margins))),
        'fp32_local_winners_remaining_local_winners_after_quant':float(np.mean(survive)),
        'quantized_strict_local_winner_fraction':float(np.mean(np.array(quant_margins)>0)),
        'quantized_local_neighbor_tie_fraction':float(np.mean(np.array(quant_margins)==0)),
        'fp32_selected_points_with_int8_selected_neighbor_within1canonical_px':float(np.mean(overlap)),
        'mean_repeated_score_fraction_among_int8_selected_points':float(np.mean(ties)),
        'scope':'84single-view DEV images, valid-content masks; detector diagnostic, not correspondence/pose evaluation. Local-winner subset excludes secondary peaks of iterated NMS. Absolute score change exceeding margin alone does not prove a ranking flip.',
        'no_training_or_threshold_changes':True,'code_sha256':sha(__file__)}
    put(dest/'peak_stability.json',result);print(json.dumps(result),flush=True)
