"""Controlled exact-observation descriptor map rebuild; fixed query and PnP."""
import _bootstrap
import json
import argparse
from types import SimpleNamespace
import numpy as np
import torch.nn.functional as F
from spk210.map_nms_gate import DEST,MODEL,context,extract,normalize,match,cv2,qvec2rotmat,read_points3D_binary,torch,ops
from spk210.io import sha,put

parser=argparse.ArgumentParser();parser.add_argument('--matcher',choices=['strict','mnn'],default='mnn');args=parser.parse_args()
out=DEST/f'exact_observations_{args.matcher}';out.mkdir(exist_ok=True)
def selected_match(d,store,active,cfg):
    if args.matcher=='strict':return match(d,store,active,cfg)
    if not len(d) or not len(active):return [],0
    scores=normalize(d)@normalize(store.local[active]).T
    forward=scores.argmax(1);reverse=scores.argmax(0)
    ii=np.flatnonzero(reverse[forward]==np.arange(len(d)))
    result=[(int(j),int(active[forward[j]]),float(scores[j,forward[j]])) for j in ii]
    return sorted(result,key=lambda r:(-r[2],r[0]))[:cfg['max_matches']],len(result)
p=json.loads((DEST/'protocol.json').read_text());base=json.loads((DEST/'results.json').read_text())
oracle=json.loads((DEST/'oracle_references/results.json').read_text())
conditions=['fp32_original','int8_greedy_corner']
spec={'purpose':'Fix map descriptor association by sampling dense features at exact existing TRAIN COLMAP observations',
      'source_sha256':sha(__file__),'base_protocol_sha256':sha(DEST/'protocol.json'),'base_results_sha256':sha(DEST/'results.json'),
      'oracle_results_sha256':sha(DEST/'oracle_references/results.json'),'conditions':conditions,
      'association':'all valid TRAIN observations within same eroded input content mask; same stride8 descriptor interpolation/L2; mean normalized observations per Point3D then normalize',
      'matcher':args.matcher,'matcher_definition':'strict baseline OR cosine mutual-nearest-neighbor with no absolute cutoff/margin, unique Point3D, cap120; deliberately separate DEV intervention',
      'fixed':'query checkpoint/features; normal and oracle reference lists; active600 and PnP unchanged',
      'scope':'engineering DEV; oracle explicitly geometry-assisted; no training, GT1, or release promotion'}
path=out/'protocol.json'
if path.exists():assert json.loads(path.read_text())==spec
else:put(path,spec)
images,cameras=context();points=read_points3D_binary(MODEL/'points3D.bin');cfg=p['matching_config']
for name,digest in p['geometry_hashes'].items():assert sha(MODEL/name)==digest
inputs=json.loads((DEST/'inputs.json').read_text());sim=json.loads((DEST/'simulation.json').read_text())
assert sim['inputs_manifest_sha256']==sha(DEST/'inputs.json') and sim['kmodel_sha256']==p['kmodel_sha256']
for row in inputs['records']:
    i=row['id'];assert sha(DEST/'inputs'/f'{i}.npz')==row['input_sha256']
    assert sha(DEST/'fp32'/f'{i}.npz')==row['fp32_sha256'] and sha(DEST/'int8'/f'{i}.npz')==sim['outputs'][str(i)]
report={}
for condition in conditions:
    backend,variant=condition.split('_',1);sums={};obs={};nobs=0
    for i in p['split_ids']['train']:
        with np.load(DEST/'inputs'/f'{i}.npz') as z:mask,affine=z['mask'][0],z['affine']
        im=images[i];valid=im.point3D_ids>=0;pid=im.point3D_ids[valid];logical=im.xys[valid]*affine[:2]+affine[2:]
        rounded=np.rint(logical).astype(int);inside=(rounded[:,0]>=0)&(rounded[:,0]<184)&(rounded[:,1]>=0)&(rounded[:,1]<320)
        keep=np.zeros(len(pid),bool);keep[inside]=mask[rounded[inside,1],rounded[inside,0]]>0
        pid=pid[keep];logical=logical[keep];obs[i]=sorted(set(int(v) for v in pid))
        with np.load(DEST/backend/f'{i}.npz') as z:b=torch.from_numpy(z['desc'].copy())
        xy=torch.from_numpy(logical.astype(np.float32))[None]
        desc=ops.sample_descriptors(xy,F.normalize(b,dim=1),8)[0].T.numpy()
        assert np.isfinite(desc).all()
        for v,d in zip(pid,desc):v=int(v);sums[v]=sums.get(v,np.zeros(256,np.float32))+d;nobs+=1
    ids=np.array(sorted(sums),np.int64);local=normalize(np.stack([sums[int(v)] for v in ids]));xyz=np.array([points[int(v)].xyz for v in ids]);rowof={int(v):j for j,v in enumerate(ids)}
    store=SimpleNamespace(point_ids=ids,local=local,xyz=xyz)
    folder=out/condition;folder.mkdir(exist_ok=True)
    for n,a in [('point_ids',ids),('local',local),('xyz',xyz)]:np.save(folder/f'{n}.npy',a)
    put(folder/'identity.json',{'checkpoint_sha256':p['checkpoint_sha256'],'kmodel_sha256':p['kmodel_sha256'] if backend=='int8' else None,
        'query_condition':condition,'map_sampling':'exact TRAIN observations, same dense descriptor sampling as query','protocol_sha256':sha(path),
        'files':{n:sha(folder/f'{n}.npy') for n in ['point_ids','local','xyz']}})
    stages={}
    for stage in ['normal','oracle']:
        rows=[]
        source=base['results'][condition]['frames'] if stage=='normal' else oracle['results'][condition]['geometry_oracle']['frames']
        for old in source:
            i=old['id'];active=[];seen=set()
            for ref in old['references']:
                for v in obs[ref]:
                    if v not in seen:seen.add(v);active.append(rowof[v])
            active=np.array(active[:cfg['max_active_points']],np.int64);xy,d,_,_=extract(i,backend,variant)
            matches,raw=selected_match(d,store,active,cfg);im=images[i];cam=cameras[im.camera_id];assert cam.model=='SIMPLE_RADIAL'
            f,cx,cy,k=cam.params;K=np.array([[f,0,cx],[0,f,cy],[0,0,1.]]);dist=np.array([k,0,0,0,0.]);gtR=qvec2rotmat(im.qvec)
            r={'id':i,'references':old['references'],'status':'insufficient_matches','good_pose':False,'matches':len(matches),'inliers':0,
               'active_points':len(active),'shared_active_tracks':len(set(ids[active].tolist())&(set(im.point3D_ids.tolist())-{-1}))}
            projected,_=cv2.projectPoints(xyz[active],cv2.Rodrigues(gtR)[0],im.tvec,K,dist)
            distances=np.linalg.norm(xy[:,None]-projected[:,0][None],axis=2)
            distances[:,(xyz[active]@gtR.T+im.tvec)[:,2]<=0]=np.inf
            r['query_points_with_projected_active_within4px']=int((distances.min(axis=1)<=4).sum()) if len(xy) else 0
            if len(matches)>=cfg['min_inliers']:
                pts=np.array([xyz[v] for j,v,s in matches],np.float64);pix=np.array([xy[j] for j,v,s in matches],np.float64)
                cv2.setRNGSeed(17)
                ok,rv,tv,ins=cv2.solvePnPRansac(pts,pix,K,dist,iterationsCount=100,reprojectionError=cfg['pnp_error_px'],confidence=.99,flags=cv2.SOLVEPNP_EPNP)
                r['status']='pnp_failed';r['inliers']=0 if ins is None else len(ins)
                if ok and r['inliers']>=cfg['min_inliers']:
                    R=cv2.Rodrigues(rv)[0];center=(-R.T@tv).ravel();depth=(pts[ins.ravel()]@R.T+tv.ravel())[:,2]
                    angle=float(np.degrees(np.arccos(np.clip((np.trace(R@gtR.T)-1)/2,-1,1))));error=float(np.linalg.norm(center+gtR.T@im.tvec))
                    r.update(status='pose' if np.isfinite(center).all() and (depth>0).all() else 'invalid_pose',rotation_deg=angle,center_native=error)
                    r['good_pose']=r['status']=='pose' and angle<=5 and error<=.1
            rows.append(r)
        summary={'good_poses':sum(r['good_pose'] for r in rows),'poses':sum(r['status']=='pose' for r in rows),'queries':len(rows),'map_points':len(ids),'reference_observations':nobs}
        for key in ['matches','inliers','shared_active_tracks','query_points_with_projected_active_within4px']:summary['median_'+key]=float(np.median([r[key] for r in rows]))
        stages[stage]={'summary':summary,'frames':rows};print(condition,stage,json.dumps(summary),flush=True)
    report[condition]=stages;put(out/'results.json',{'protocol_sha256':sha(path),'complete':len(report)==len(conditions),'results':report})
