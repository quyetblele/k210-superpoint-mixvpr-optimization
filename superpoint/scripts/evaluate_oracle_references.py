"""DEV-only geometry-assisted reference diagnostic; never deployable retrieval."""
import _bootstrap
import json
from types import SimpleNamespace
import numpy as np
from spk210.map_nms_gate import DEST,context,extract,normalize,match,cv2,qvec2rotmat,MODEL
from spk210.io import sha,put

out=DEST/'oracle_references';out.mkdir(exist_ok=True)
p=json.loads((DEST/'protocol.json').read_text());base=json.loads((DEST/'results.json').read_text());assert base['complete']
conditions=['fp32_original','int8_original','int8_greedy_corner']
spec={'scope':'DEV geometry-assisted oracle diagnostic, NOT deployable performance; no training or GT1',
      'base_protocol_sha256':sha(DEST/'protocol.json'),'base_results_sha256':sha(DEST/'results.json'),'code_sha256':sha(__file__),
      'conditions':conditions,'intervention':'top3 TRAIN reference images by count of shared query COLMAP tracks, ties by image id; unchanged covisibility expansion to8 and active cap600',
      'fixed':'same existing per-condition map, query features, matcher, PnP and good-pose criteria; replay normal retrieval first',
      'diagnostics':'shared active tracks and nearest projected active-point distance; projection coverage is optimistic and does not establish correspondence'}
path=out/'protocol.json'
if path.exists():assert json.loads(path.read_text())==spec
else:put(path,spec)
for name,digest in p['geometry_hashes'].items():assert sha(MODEL/name)==digest
inputs=json.loads((DEST/'inputs.json').read_text());sim=json.loads((DEST/'simulation.json').read_text())
assert sim['inputs_manifest_sha256']==sha(DEST/'inputs.json') and sim['kmodel_sha256']==p['kmodel_sha256']
for row in inputs['records']:
    i=row['id'];assert sha(DEST/'inputs'/f'{i}.npz')==row['input_sha256']
    assert sha(DEST/'fp32'/f'{i}.npz')==row['fp32_sha256']
    assert sha(DEST/'int8'/f'{i}.npz')==sim['outputs'][str(i)]
images,cameras=context();train=p['split_ids']['train'];cfg=p['matching_config']
visible={i:set(int(v) for v in im.point3D_ids if v>=0) for i,im in images.items()}
oracle={}
for i in p['split_ids']['dev']:
    refs=sorted(train,key=lambda j:(-len(visible[i]&visible[j]),j))[:cfg['top_k']];expanded=refs.copy()
    for ref in refs:
        for j in sorted(train,key=lambda j:(-len(visible[ref]&visible[j]),j)):
            if len(expanded)>=cfg['max_active_images']:break
            if j not in expanded and visible[ref]&visible[j]:expanded.append(j)
    oracle[i]=expanded
report={}
for condition in conditions:
    backend,variant=condition.split('_',1);folder=DEST/f'map_{condition}';identity=json.loads((folder/'identity.json').read_text())
    expected=f"{p['checkpoint_sha256']}:{backend}:{variant}:"+sha(DEST/'protocol.json')
    assert identity['local_identity']==expected and identity['references']==train
    for name,digest in identity['files'].items():assert sha(folder/f'{name}.npy')==digest
    ids=np.load(folder/'point_ids.npy');local=np.load(folder/'local.npy');xyz=np.load(folder/'xyz.npy');rowof={int(v):j for j,v in enumerate(ids)}
    store=SimpleNamespace(local=local,xyz=xyz,point_ids=ids);obs={}
    for i in train:
        xy,d,logical,affine=extract(i,backend,variant);im=images[i];valid=im.point3D_ids>=0
        pid=im.point3D_ids[valid];uv=im.xys[valid]*affine[:2]+affine[2:];obs[i]=[]
        if not len(pid) or not len(xy):continue
        dist=np.linalg.norm(logical[:,None]-uv[None],axis=2)/1.25;nearest=dist.argmin(1);seen=set()
        for j in np.argsort(dist[np.arange(len(xy)),nearest],kind='stable'):
            v=int(pid[nearest[j]])
            if dist[j,nearest[j]]<=2 and v not in seen:seen.add(v);obs[i].append(v)
    stages={}
    for stage in ['normal_replay','geometry_oracle']:
      rows=[]
      for original in base['results'][condition]['frames']:
        i=original['id'];refs=original['references'] if stage=='normal_replay' else oracle[i]
        active=[];seen=set()
        for ref in refs:
            for v in sorted(obs[ref]):
                if v not in seen:seen.add(v);active.append(rowof[v])
        active=np.array(active[:cfg['max_active_points']],np.int64)
        xy,d,_,_=extract(i,backend,variant);matches,raw=match(d,store,active,cfg)
        cam=cameras[images[i].camera_id];assert cam.model=='SIMPLE_RADIAL';f,cx,cy,k=cam.params
        K=np.array([[f,0,cx],[0,f,cy],[0,0,1.]],np.float64);distortion=np.array([k,0,0,0,0.]);gtR=qvec2rotmat(images[i].qvec)
        r={'id':i,'references':refs,'status':'insufficient_matches','good_pose':False,'correspondences':len(matches),'raw_matches':raw,'inliers':0,
           'active_points':len(active),'shared_active_tracks':len(visible[i]&set(ids[active].tolist()))}
        if len(active) and len(xy):
            projected,_=cv2.projectPoints(xyz[active],cv2.Rodrigues(gtR)[0],images[i].tvec,K,distortion)
            front=(xyz[active]@gtR.T+images[i].tvec)[:,2]>0
            distance=np.linalg.norm(xy[:,None]-projected[:,0][None],axis=2);distance[:,~front]=np.inf
            r['query_points_with_projected_active_within4px']=int((distance.min(axis=1)<=4).sum())
        else:r['query_points_with_projected_active_within4px']=0
        if len(matches)>=cfg['min_inliers']:
            pts=np.array([xyz[v] for j,v,s in matches],np.float64);pix=np.array([xy[j] for j,v,s in matches],np.float64)
            cv2.setRNGSeed(17)
            ok,rv,tv,inliers=cv2.solvePnPRansac(pts,pix,K,distortion,iterationsCount=100,reprojectionError=cfg['pnp_error_px'],confidence=.99,flags=cv2.SOLVEPNP_EPNP)
            r['status']='pnp_failed';r['inliers']=0 if inliers is None else len(inliers)
            if ok and r['inliers']>=cfg['min_inliers']:
                R=cv2.Rodrigues(rv)[0];center=(-R.T@tv).ravel();depth=(pts[inliers.ravel()]@R.T+tv.ravel())[:,2]
                angle=float(np.degrees(np.arccos(np.clip((np.trace(R@gtR.T)-1)/2,-1,1))))
                error=float(np.linalg.norm(center+gtR.T@images[i].tvec))
                r.update(status='pose' if np.isfinite(center).all() and (depth>0).all() else 'invalid_pose',rotation_deg=angle,center_native=error)
                r['good_pose']=r['status']=='pose' and angle<=5 and error<=.1
        if stage=='normal_replay':
            for key in ['status','good_pose','correspondences','raw_matches','inliers','active_points']:assert r[key]==original[key],(condition,i,key)
        rows.append(r)
      summary={'good_poses':sum(r['good_pose'] for r in rows),'poses':sum(r['status']=='pose' for r in rows),'queries':len(rows)}
      for key in ['correspondences','inliers','shared_active_tracks','query_points_with_projected_active_within4px']:
        summary['median_'+key]=float(np.median([r[key] for r in rows]))
      stages[stage]={'summary':summary,'frames':rows};print(condition,stage,json.dumps(summary),flush=True)
    report[condition]=stages
    put(out/'results.json',{'protocol_sha256':sha(path),'complete':len(report)==len(conditions),'results':report})
