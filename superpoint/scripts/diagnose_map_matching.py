"""Explain fixed matching failures without changing thresholds or rerunning tuned PnP."""
import _bootstrap
import json
import numpy as np
from spk210.map_nms_gate import DEST,context,extract,normalize,cv2,qvec2rotmat
from spk210.io import sha,put

p=json.loads((DEST/'protocol.json').read_text())
results=json.loads((DEST/'results.json').read_text());assert results['complete']
spec={'purpose':'Post-hoc failure decomposition; no selection, calibration, training, or threshold change',
      'results_sha256':sha(DEST/'results.json'),'code_sha256':sha(__file__),
      'conditions':['fp32_original','int8_original','int8_greedy_corner'],
      'GT_residual':'known reconstruction pose, same SIMPLE_RADIAL camera; diagnostic only, not independent GT'}
path=DEST/'matching_diagnosis_protocol.json'
if path.exists():assert json.loads(path.read_text())==spec
else:put(path,spec)
images,cameras=context();report={}
for condition in spec['conditions']:
    backend,variant=condition.split('_',1);folder=DEST/f'map_{condition}'
    ids=np.load(folder/'point_ids.npy'); local=normalize(np.load(folder/'local.npy'));xyz=np.load(folder/'xyz.npy');rowof={int(v):j for j,v in enumerate(ids)}
    obs={}
    for i in p['split_ids']['train']:
        xy,d,logical,affine=extract(i,backend,variant);im=images[i];valid=im.point3D_ids>=0
        pid=im.point3D_ids[valid];uv=im.xys[valid]*affine[:2]+affine[2:];obs[i]=[]
        if not len(pid) or not len(xy):continue
        dist=np.linalg.norm(logical[:,None]-uv[None],axis=2)/1.25;nearest=dist.argmin(1);seen=set()
        for j in np.argsort(dist[np.arange(len(xy)),nearest],kind='stable'):
            v=int(pid[nearest[j]])
            if dist[j,nearest[j]]<=2 and v not in seen:seen.add(v);obs[i].append(v)
    rows=[];residuals=[];margins=[];bests=[]
    for frame in results['results'][condition]['frames']:
        i=frame['id'];active=[];seen=set()
        for ref in frame['references']:
            for v in sorted(obs[ref]):
                if v not in seen:seen.add(v);active.append(rowof[v])
        active=np.array(active[:p['matching_config']['max_active_points']])
        xy,d,_,_=extract(i,backend,variant);scores=normalize(d)@local[active].T
        ordered=np.argsort(-scores,axis=1,kind='stable')[:,:2]
        best=scores[np.arange(len(xy)),ordered[:,0]];second=scores[np.arange(len(xy)),ordered[:,1]]
        threshold=best>=.8;gap=best-second>=.05;accepted=threshold&gap
        assert int(accepted.sum())==frame['raw_matches']
        cam=cameras[images[i].camera_id];f,cx,cy,k=cam.params;R=qvec2rotmat(images[i].qvec)
        projected,_=cv2.projectPoints(xyz[active[ordered[:,0]]],cv2.Rodrigues(R)[0],images[i].tvec,
                                     np.array([[f,0,cx],[0,f,cy],[0,0,1.]],np.float64),np.array([k,0,0,0,0.]))
        residual=np.linalg.norm(projected[:,0]-xy,axis=1)
        residuals.extend(residual[accepted].tolist());margins.extend((best-second).tolist());bests.extend(best.tolist())
        rows.append({'id':i,'query_points':len(xy),'pass_similarity':int(threshold.sum()),'pass_margin':int(gap.sum()),
                     'pass_both_before_unique':int(accepted.sum()),'accepted_with_GT_residual_le4':int((accepted&(residual<=4)).sum())})
    summary={'median_query_points':float(np.median([r['query_points'] for r in rows])),
             'median_pass_similarity':float(np.median([r['pass_similarity'] for r in rows])),
             'median_pass_margin':float(np.median([r['pass_margin'] for r in rows])),
             'median_pass_both_before_unique':float(np.median([r['pass_both_before_unique'] for r in rows])),
             'median_best_similarity':float(np.median(bests)),'median_best_second_margin':float(np.median(margins)),
             'accepted_GT_residual_median_px':float(np.median(residuals)) if residuals else None,
             'accepted_GT_residual_le4_fraction':float(np.mean(np.array(residuals)<=4)) if residuals else None}
    report[condition]={'summary':summary,'frames':rows};print(condition,json.dumps(summary),flush=True)
put(DEST/'matching_diagnosis.json',{'protocol_sha256':sha(path),'results':report})
