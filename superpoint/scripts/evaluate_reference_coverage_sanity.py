"""Adjacent-frame reference coverage sanity, explicitly not an independent test."""
import _bootstrap
import json
import argparse
from types import SimpleNamespace
import numpy as np
import torch.nn.functional as F
from spk210.map_nms_gate import DEST,MODEL,ASSETS,context,normalize,match,cv2,qvec2rotmat,read_points3D_binary,torch,ops
from spk210.runtime import w0,SuperPointKPUWrapper
from spk210.io import sha,put,guard

parser=argparse.ArgumentParser();parser.add_argument('--backend',choices=['official_high','official_low','student_fp32','student_int8'],required=True);args=parser.parse_args()
out=DEST/f'reference_coverage_{args.backend}';out.mkdir(exist_ok=True)
p=json.loads((DEST/'protocol.json').read_text());train=p['split_ids']['train'];dev=p['split_ids']['dev']
support=dev[::2];queries=dev[1::2];assert not set(support)&set(queries)
oldroot=DEST/('reference_observations_scale1' if args.backend=='official_low' else 'reference_observations_scale2')
spec={'scope':'nearby-frame engineering SANITY ONLY; DEV reassigned explicitly into support/query; NOT independent test, NOT improvement on the original41-query protocol',
      'source_sha256':sha(__file__),'base_protocol_sha256':sha(DEST/'protocol.json'),'teacher_protocol_sha256':sha(oldroot/'protocol.json'),
      'backend':args.backend,'student_checkpoint_sha256':p['checkpoint_sha256'] if args.backend.startswith('student') else None,
      'student_kmodel_sha256':p['kmodel_sha256'] if args.backend=='student_int8' else None,
      'support_ids':support,'query_ids':queries,'reference_conditions':['TRAIN_ONLY','TRAIN_PLUS_DEV_SUPPORT'],
      'representations':['mean','per_reference'],'fixed':'official_high logical368x640 OR official_low logical184x320; student logical184x320 greedy-corner NMS for BOTH FP32/INT8; strict similarity.8/margin.05, active600, top3 plus covisibility8, PnP100iter4px8inliers; frozen MixVPR retrieval',
      'no_self_match':True,'no_training':True,'GT1':False,'purpose':'test whether contiguous removal of all DEV reference views confounded previous0/41 conclusions'}
path=out/'protocol.json'
if path.exists():assert json.loads(path.read_text())==spec
else:put(path,spec)
images,cameras=context();points=read_points3D_binary(MODEL/'points3D.bin');cfg=p['matching_config']
for name,digest in p['geometry_hashes'].items():assert sha(MODEL/name)==digest
if args.backend.startswith('official'):
    teacher,_=w0.import_original_superpoint();model=SuperPointKPUWrapper(teacher).eval();torch.set_num_threads(2)
    observation={};g={};qfeatures={};records={}
    for i in train+dev:
        with np.load(DEST/'inputs'/f'{i}.npz') as z:g[i]=z['global_descriptor'].copy()
    for i in train:
        file=oldroot/'features'/f'{i}.npz'
        with np.load(file) as z:observation[i]=(z['point_ids'].copy(),z['desc'].copy())
        records[str(i)]=sha(file)
    for i in queries:
        file=oldroot/'features'/f'{i}.npz'
        with np.load(file) as z:qfeatures[i]=(z['xy'].copy(),z['desc'].copy())
        records[str(i)]=sha(file)
    (out/'support_features').mkdir(exist_ok=True)
    with torch.inference_mode():
     for index,i in enumerate(support):
        guard();gray=cv2.imread(str(ASSETS/images[i].name),0);ih,iw=gray.shape
        factor=min(288/iw,512/ih);nw,nh=round(iw*factor),round(ih*factor);x0,y0=(288-nw)//2,(512-nh)//2
        common=np.zeros((1024,576),np.uint8);common[y0*2:(y0+nh)*2,x0*2:(x0+nw)*2]=cv2.resize(gray,(nw*2,nh*2),interpolation=cv2.INTER_AREA)
        high=np.zeros((640,368),np.uint8);high[:,4:364]=cv2.resize(common,(360,640),interpolation=cv2.INTER_AREA)
        with np.load(DEST/'inputs'/f'{i}.npz') as z:mask=cv2.resize(z['mask'][0],(368,640),interpolation=cv2.INTER_NEAREST);affine=z['affine'].copy()
        affine=np.r_[affine[:2]*2,(affine[2:]+.5)*2-.5]
        if args.backend=='official_low':
            with np.load(DEST/'inputs'/f'{i}.npz') as z:
                network_input=z['image'].copy();mask=z['mask'][0].copy();affine=z['affine'].copy()
        else:network_input=high[None,None].astype(np.float32)/255
        _,b=model(torch.from_numpy(network_input))
        im=images[i];valid=im.point3D_ids>=0;pid=im.point3D_ids[valid];logical=im.xys[valid]*affine[:2]+affine[2:]
        r=np.rint(logical).astype(int);inside=(r[:,0]>=0)&(r[:,0]<mask.shape[1])&(r[:,1]>=0)&(r[:,1]<mask.shape[0])
        keep=np.zeros(len(pid),bool);keep[inside]=mask[r[inside,1],r[inside,0]]>0;pid=pid[keep];logical=logical[keep]
        d=ops.sample_descriptors(torch.from_numpy(logical.astype(np.float32))[None],F.normalize(b,dim=1),8)[0].T.numpy();observation[i]=(pid,d)
        file=out/'support_features'/f'{i}.npz';np.savez_compressed(file,point_ids=pid,desc=d);records[str(i)]=sha(file)
        if index%7==0:print('SUPPORT',index+1,len(support),flush=True)
else:
    from spk210.map_nms_gate import extract
    backend='fp32' if args.backend=='student_fp32' else 'int8'
    observation={};g={};qfeatures={};records={}
    manifest=json.loads((DEST/'inputs.json').read_text());sim=json.loads((DEST/'simulation.json').read_text())
    assert sim['kmodel_sha256']==p['kmodel_sha256'] and sim['inputs_manifest_sha256']==sha(DEST/'inputs.json')
    entries={r['id']:r for r in manifest['records']}
    for i in train+dev:
        file=DEST/'inputs'/f'{i}.npz';assert sha(file)==entries[i]['input_sha256']
        with np.load(file) as z:g[i]=z['global_descriptor'].copy();mask=z['mask'][0].copy();affine=z['affine'].copy()
        raw=DEST/backend/f'{i}.npz'
        assert sha(raw)==(entries[i]['fp32_sha256'] if backend=='fp32' else sim['outputs'][str(i)])
        records[str(i)]=sha(raw)
        if i in queries:
            xy,d,_,_=extract(i,backend,'greedy_corner');qfeatures[i]=(xy,d)
        else:
            with np.load(raw) as z:b=torch.from_numpy(z['desc'].copy())
            im=images[i];valid=im.point3D_ids>=0;pid=im.point3D_ids[valid];logical=im.xys[valid]*affine[:2]+affine[2:]
            r=np.rint(logical).astype(int);inside=(r[:,0]>=0)&(r[:,0]<184)&(r[:,1]>=0)&(r[:,1]<320)
            keep=np.zeros(len(pid),bool);keep[inside]=mask[r[inside,1],r[inside,0]]>0;pid=pid[keep];logical=logical[keep]
            d=ops.sample_descriptors(torch.from_numpy(logical.astype(np.float32))[None],F.normalize(b,dim=1),8)[0].T.numpy()
            observation[i]=(pid,d)
visible={i:set(int(v) for v in images[i].point3D_ids if v>=0) for i in train+dev}
report={}
for reference_condition in spec['reference_conditions']:
 refs_all=train if reference_condition=='TRAIN_ONLY' else train+support
 sums={}
 for i in refs_all:
    for v,d in zip(*observation[i]):v=int(v);sums[v]=sums.get(v,np.zeros(256,np.float32))+d
 ids=np.array(sorted(sums),np.int64);local=normalize(np.stack([sums[int(v)] for v in ids]));xyz=np.array([points[int(v)].xyz for v in ids]);rowof={int(v):j for j,v in enumerate(ids)}
 store=SimpleNamespace(point_ids=ids,local=local,xyz=xyz);global_db=np.stack([g[i] for i in refs_all])
 for representation in spec['representations']:
  rows=[]
  for i in queries:
    refs=[refs_all[j] for j in np.argsort(-(global_db@g[i]),kind='stable')[:cfg['top_k']]];expanded=refs.copy()
    for ref in refs:
        for j in sorted(refs_all,key=lambda j:(-len(visible[ref]&visible[j]),j)):
            if len(expanded)>=cfg['max_active_images']:break
            if j not in expanded and visible[ref]&visible[j]:expanded.append(j)
    assert i not in expanded and not set(expanded)&set(queries)
    active=[];seen=set()
    for ref in expanded:
        for v in sorted(set(observation[ref][0].tolist())):
            if v not in seen:seen.add(v);active.append(rowof[v])
    active=np.array(active[:cfg['max_active_points']],np.int64);xy,d=qfeatures[i]
    if representation=='mean':matches,_=match(d,store,active,cfg)
    else:
        proposals=[];active_ids=set(ids[active].tolist())
        for ref in expanded:
            pid,rd=observation[ref];keep=np.array([int(v) in active_ids for v in pid]);pid=pid[keep];rd=rd[keep]
            if len(rd)<2 or not len(d):continue
            score=normalize(d)@normalize(rd).T;forward=score.argmax(1);best=score[np.arange(len(d)),forward];second=np.partition(score,-2,axis=1)[:,-2]
            accepted=(best>=cfg['similarity_threshold'])&(best-second>=cfg['similarity_margin'])
            proposals.extend((int(j),int(pid[forward[j]]),float(best[j])) for j in np.flatnonzero(accepted))
        matches=[];used_q=set();used_p=set()
        for j,v,s in sorted(proposals,key=lambda a:(-a[2],a[0],a[1])):
            if j in used_q or v in used_p:continue
            used_q.add(j);used_p.add(v);matches.append((j,rowof[v],s))
            if len(matches)==cfg['max_matches']:break
    im=images[i];cam=cameras[im.camera_id];f,cx,cy,k=cam.params;assert cam.model=='SIMPLE_RADIAL'
    K=np.array([[f,0,cx],[0,f,cy],[0,0,1.]]);dist=np.array([k,0,0,0,0.]);gtR=qvec2rotmat(im.qvec)
    r={'id':i,'references':expanded,'matches':len(matches),'inliers':0,'status':'insufficient_matches','good_pose':False,'shared_active_tracks':len(visible[i]&set(ids[active].tolist()))}
    if len(matches)>=cfg['min_inliers']:
        pts=np.array([xyz[v] for j,v,s in matches],np.float64);pix=np.array([xy[j] for j,v,s in matches],np.float64);cv2.setRNGSeed(17)
        ok,rv,tv,ins=cv2.solvePnPRansac(pts,pix,K,dist,iterationsCount=100,reprojectionError=4,confidence=.99,flags=cv2.SOLVEPNP_EPNP)
        r['status']='pnp_failed';r['inliers']=0 if ins is None else len(ins)
        if ok and r['inliers']>=8:
            R=cv2.Rodrigues(rv)[0];center=(-R.T@tv).ravel();depth=(pts[ins.ravel()]@R.T+tv.ravel())[:,2]
            angle=float(np.degrees(np.arccos(np.clip((np.trace(R@gtR.T)-1)/2,-1,1))));error=float(np.linalg.norm(center+gtR.T@im.tvec))
            r.update(status='pose' if np.isfinite(center).all() and (depth>0).all() else 'invalid_pose',rotation_deg=angle,center_native=error)
            r['good_pose']=r['status']=='pose' and angle<=5 and error<=.1
    rows.append(r)
  summary={'good_poses':sum(r['good_pose'] for r in rows),'poses':sum(r['status']=='pose' for r in rows),'queries':len(rows),'median_matches':float(np.median([r['matches'] for r in rows])),'median_shared_active_tracks':float(np.median([r['shared_active_tracks'] for r in rows]))}
  key=reference_condition+'_'+representation;report[key]={'summary':summary,'frames':rows};print(key,json.dumps(summary),flush=True)
  put(out/'results.json',{'protocol_sha256':sha(path),'complete':len(report)==4,'results':report,'feature_hashes':records})
