"""Official SuperPoint FP32 anchor at the student's exact input and map protocol."""
import _bootstrap
import json
import argparse
from types import SimpleNamespace
import numpy as np
import torch.nn.functional as F
from spk210.map_nms_gate import DEST,MODEL,context,normalize,match,cv2,qvec2rotmat,read_points3D_binary,torch,ops
from spk210.runtime import w0,SuperPointKPUWrapper
from spk210.metrics import features
from spk210.io import sha,put,guard

parser=argparse.ArgumentParser();parser.add_argument('--scale',type=int,choices=[1,2],default=2);args=parser.parse_args();scale=args.scale
out=DEST/f'teacher_anchor_scale{scale}';out.mkdir(exist_ok=True)
p=json.loads((DEST/'protocol.json').read_text());base=json.loads((DEST/'results.json').read_text())
oracle=json.loads((DEST/'oracle_references/results.json').read_text())
checkpoint=w0.EDGE/'third_party/SuperGluePretrainedNetwork/models/weights/superpoint_v1.pth'
spec={'scope':'official teacher FP32 diagnostic, not student/K210 performance; DEV only, no training or GT1',
      'checkpoint_sha256':sha(checkpoint),'source_sha256':sha(__file__),'input_manifest_sha256':sha(DEST/'inputs.json'),
      'base_protocol_sha256':sha(DEST/'protocol.json'),'oracle_results_sha256':sha(DEST/'oracle_references/results.json'),
      'scale':scale,'resize':'reconstruct same source FOV letterbox directly from original image at multiplied sizes; not upsampling low-resolution input',
      'fixed':'same source FOV; NMS radius round(3*1.25*scale), cap160, threshold.005; TRAIN exact-observation map; mean/L2, normal/oracle refs, active600, matching and PnP unchanged'}
path=out/'protocol.json'
if path.exists():assert json.loads(path.read_text())==spec
else:put(path,spec)
images,cameras=context();points=read_points3D_binary(MODEL/'points3D.bin');cfg=p['matching_config']
for name,digest in p['geometry_hashes'].items():assert sha(MODEL/name)==digest
teacher,_=w0.import_original_superpoint();model=SuperPointKPUWrapper(teacher).eval();torch.set_num_threads(2)
sums={};obs={};queries={};records=[]
with torch.inference_mode():
 for index,row in enumerate(json.loads((DEST/'inputs.json').read_text())['records']):
    guard();i=row['id'];src=DEST/'inputs'/f'{i}.npz';assert sha(src)==row['input_sha256']
    with np.load(src) as z:x,mask,affine=(z[k].copy() for k in ('image','mask','affine'))
    if scale>1:
        from spk210.map_nms_gate import ASSETS
        gray=cv2.imread(str(ASSETS/images[i].name),0);ih,iw=gray.shape
        factor=min(288/iw,512/ih);nw,nh=round(iw*factor),round(ih*factor);x0,y0=(288-nw)//2,(512-nh)//2
        common=np.zeros((512*scale,288*scale),np.uint8)
        common[y0*scale:(y0+nh)*scale,x0*scale:(x0+nw)*scale]=cv2.resize(gray,(nw*scale,nh*scale),interpolation=cv2.INTER_AREA)
        high=np.zeros((320*scale,184*scale),np.uint8);high[:,2*scale:182*scale]=cv2.resize(common,(180*scale,320*scale),interpolation=cv2.INTER_AREA)
        x=high[None,None].astype(np.float32)/255
        mask=cv2.resize(mask[0],(184*scale,320*scale),interpolation=cv2.INTER_NEAREST)[None]
        affine=np.r_[affine[:2]*scale,(affine[2:]+.5)*scale-.5]
    a,b=model(torch.from_numpy(x));records.append({'id':i,'input_sha256':sha(src)})
    if i in p['split_ids']['dev']:
        if scale==1:
            xy,d=features(a,b,mask,'r2')[0];logical=xy*1.25+[2,0]
        else:
            prob=a.softmax(1)[:,:64];n,_,hh,ww=prob.shape
            prob=prob.permute(0,2,3,1).reshape(n,hh,ww,8,8).permute(0,1,3,2,4).reshape(n,320*scale,184*scale)
            prob=ops.simple_nms(prob*torch.from_numpy(mask),round(3*1.25*scale));yx=torch.nonzero(prob[0]>.005)
            yx,_=ops.top_k_keypoints(yx,prob[0][tuple(yx.t())],160);txy=yx.flip([1]).float()
            d=ops.sample_descriptors(txy[None],F.normalize(b,dim=1),8)[0].T.numpy();logical=txy.numpy()
        queries[i]=((logical-affine[2:])/affine[:2],d)
    else:
        im=images[i];valid=im.point3D_ids>=0;pid=im.point3D_ids[valid];logical=im.xys[valid]*affine[:2]+affine[2:]
        rounded=np.rint(logical).astype(int);inside=(rounded[:,0]>=0)&(rounded[:,0]<184*scale)&(rounded[:,1]>=0)&(rounded[:,1]<320*scale)
        keep=np.zeros(len(pid),bool);keep[inside]=mask[0,rounded[inside,1],rounded[inside,0]]>0
        pid=pid[keep];logical=logical[keep];obs[i]=sorted(set(int(v) for v in pid))
        desc=ops.sample_descriptors(torch.from_numpy(logical.astype(np.float32))[None],F.normalize(b,dim=1),8)[0].T.numpy()
        for v,d in zip(pid,desc):v=int(v);sums[v]=sums.get(v,np.zeros(256,np.float32))+d
    if index%40==0:print('TEACHER',index+1,189,flush=True)
ids=np.array(sorted(sums),np.int64);local=normalize(np.stack([sums[int(v)] for v in ids]));xyz=np.array([points[int(v)].xyz for v in ids]);rowof={int(v):j for j,v in enumerate(ids)}
for n,a in [('point_ids',ids),('local',local),('xyz',xyz)]:np.save(out/f'{n}.npy',a)
store=SimpleNamespace(point_ids=ids,local=local,xyz=xyz);report={}
for stage in ['normal','oracle']:
 rows=[];source=base['results']['fp32_original']['frames'] if stage=='normal' else oracle['results']['fp32_original']['geometry_oracle']['frames']
 for old in source:
    i=old['id'];active=[];seen=set()
    for ref in old['references']:
        for v in obs[ref]:
            if v not in seen:seen.add(v);active.append(rowof[v])
    active=np.array(active[:cfg['max_active_points']],np.int64);xy,d=queries[i];matches,raw=match(d,store,active,cfg)
    im=images[i];cam=cameras[im.camera_id];assert cam.model=='SIMPLE_RADIAL';f,cx,cy,k=cam.params
    K=np.array([[f,0,cx],[0,f,cy],[0,0,1.]]);dist=np.array([k,0,0,0,0.]);gtR=qvec2rotmat(im.qvec)
    r={'id':i,'status':'insufficient_matches','good_pose':False,'matches':len(matches),'inliers':0,'keypoints':len(xy)}
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
 summary={'good_poses':sum(r['good_pose'] for r in rows),'poses':sum(r['status']=='pose' for r in rows),'queries':len(rows),'map_points':len(ids),
          'median_matches':float(np.median([r['matches'] for r in rows])),'median_inliers':float(np.median([r['inliers'] for r in rows]))}
 report[stage]={'summary':summary,'frames':rows};print(stage,json.dumps(summary),flush=True)
put(out/'results.json',{'protocol_sha256':sha(path),'complete':True,'results':report,'inputs':records})
