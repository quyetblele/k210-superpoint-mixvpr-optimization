"""Descriptor-donor A/B/C with fixed teacher detections and geometry. DEV only."""
import _bootstrap
import argparse
import csv
import json
from pathlib import Path
import shutil
import numpy as np
import torch.nn.functional as F
from spk210.map_nms_gate import DEST, MODEL, GLOBAL, context, read_points3D_binary, qvec2rotmat, normalize
from spk210.runtime import torch, cv2, ops, w0, SuperPointKPUWrapper
from spk210.metrics import features
from spk210.io import sha, put, guard
from spk210.settings import SUPERPOINT


def read(p): return json.loads(p.read_text())


def check(p,digest):
    if sha(p)!=digest: raise RuntimeError(f'Identity mismatch: {p}')


def camera(cam):
    assert cam.model=='SIMPLE_RADIAL'
    f,cx,cy,k=cam.params
    return np.array([[f,0,cx],[0,f,cy],[0,0,1.]]),np.array([k,0,0,0,0.])


def pose(xyz,xy,cam,im):
    result={'matches':len(xy),'inliers':0,'inlier_ratio':0.,'solver_ok':False,
            'status':'insufficient_matches','good_pose':False,'rotation_deg':None,'center_native':None}
    if len(xy)<8:return result
    K,dist=camera(cam);cv2.setRNGSeed(17)
    ok,rv,tv,ins=cv2.solvePnPRansac(np.asarray(xyz,np.float64),np.asarray(xy,np.float64),K,dist,
                                  iterationsCount=100,reprojectionError=4.,confidence=.99,flags=cv2.SOLVEPNP_EPNP)
    result.update(solver_ok=bool(ok),status='pnp_failed',inliers=0 if ins is None else len(ins))
    result['inlier_ratio']=result['inliers']/len(xy)
    if ok:
        R=cv2.Rodrigues(rv)[0];gt=qvec2rotmat(im.qvec);center=(-R.T@tv).ravel()
        angle=float(np.degrees(np.arccos(np.clip((np.trace(R@gt.T)-1)/2,-1,1))))
        error=float(np.linalg.norm(center+gt.T@im.tvec))
        finite=np.isfinite(center).all() and np.isfinite(angle) and np.isfinite(error)
        depth=(xyz[ins.ravel()]@R.T+tv.ravel())[:,2] if ins is not None else np.array([])
        accepted=finite and len(depth)>=8 and (depth>0).all()
        result.update(status='pose' if accepted else 'invalid_pose',rotation_deg=angle if np.isfinite(angle) else None,
                      center_native=error if np.isfinite(error) else None,
                      good_pose=bool(accepted and angle<=5 and error<=.1))
    return result


def match(query,banks,references,active):
    proposals=[]; allowed=set(active)
    for ref in references:
        ids,desc=banks[ref];keep=np.array([int(v) in allowed for v in ids],bool)
        ids,desc=ids[keep],desc[keep]
        if len(desc)<2 or not len(query):continue
        scores=normalize(query)@normalize(desc).T
        forward=scores.argmax(1);best=scores[np.arange(len(query)),forward]
        second=np.partition(scores,-2,axis=1)[:,-2]
        valid=(best>=.8)&(best-second>=.05)
        proposals.extend((int(j),int(ids[forward[j]]),float(best[j])) for j in np.flatnonzero(valid))
    result=[];used_q=set();used_p=set()
    for j,v,s in sorted(proposals,key=lambda t:(-t[2],t[0],t[1])):
        if j in used_q or v in used_p:continue
        used_q.add(j);used_p.add(v);result.append((j,v,s))
        if len(result)==120:break
    return result


@torch.inference_mode()
def run(out):
    out.mkdir(parents=True,exist_ok=False)
    base=read(DEST/'protocol.json');inputs=read(DEST/'inputs.json');sim=read(DEST/'simulation.json')
    entries={r['id']:r for r in inputs['records']}
    prior=DEST/'reference_coverage_official_low'
    old=read(prior/'results.json');check(prior/'protocol.json',old['protocol_sha256'])
    frames=old['results']['TRAIN_PLUS_DEV_SUPPORT_per_reference']['frames']
    queries=[r['id'] for r in frames];references={r['id']:r['references'] for r in frames}
    assert queries==base['split_ids']['dev'][1::2]
    refids=sorted(set(r for refs in references.values() for r in refs));assert not set(queries)&set(refids)
    gate=SUPERPOINT/'artifacts/measurement_gate/run_002'
    manifest=read(gate/'manifest.json')
    check(Path(manifest['checkpoint']),manifest['checkpoint_sha256'])
    assert sim['kmodel_sha256']==base['kmodel_sha256']==manifest['compile']['kmodel_sha256']
    check(SUPERPOINT/'artifacts/deployment/wider_r2_cw/model.kmodel',sim['kmodel_sha256'])
    assert base['checkpoint_sha256']==manifest['checkpoint_sha256']
    check(DEST/'inputs.json',sim['inputs_manifest_sha256'])
    check(GLOBAL/'candidate.pt',base['global_checkpoint_sha256'])
    images,cameras=context();points=read_points3D_binary(MODEL/'points3D.bin')
    for file,digest in base['geometry_hashes'].items():check(MODEL/file,digest)
    hloc=Path('/home/quyet/edge_ai_project/third_party/hloc/hloc/localize_sfm.py')
    assert 'kpq += 0.5' in hloc.read_text()
    spec={'scope':'nearby-frame DEV descriptor-donor diagnostic, not independent TEST or deployment result',
          'arms':['teacher_fp32','student_fp32','student_int8'], 'checkpoint_sha256':manifest['checkpoint_sha256'],
          'teacher_sha256':manifest['teacher_sha256'],'kmodel_sha256':sim['kmodel_sha256'],
          'input_manifest_sha256':sha(DEST/'inputs.json'),'simulation_manifest_sha256':sha(DEST/'simulation.json'),
          'geometry_hashes':base['geometry_hashes'],'frozen_global_checkpoint_sha256':base['global_checkpoint_sha256'],
          'historical_reference_results_sha256':sha(prior/'results.json'),
          'query_ids':queries,'references':references,
          'keypoints':'original teacher low-resolution W184 H320; original NMS4 threshold.005 top160 mask; identical all arms',
          'reference_observations':'existing COLMAP tracks, exclude every Point3D duplicated in one image; common corrected coordinates/masks, keep image identity; not feature-specific triangulation',
          'coordinates':'COLMAP observation minus0.5 -> zero-based source pixel -> stored pixel-center affine -> network; query inverse affine plus0.5 -> COLMAP K for OpenCV PnP',
          'hloc_coordinate_reference_sha256':sha(hloc),
          'matching':'per-image cosine>=.8, best-second>=.05; same fixed600 active unique IDs; unique query/Point3D merge, max120; no MNN',
          'pnp':{'iterations':100,'error_px':4.,'confidence':.99,'method':'EPNP','rng_seed':17,'min_inliers':8},
          'good_pose':'finite pose, all inlier depths positive, >=8 inliers, <=5degrees, <=.1 native map units',
          'correct_match':'positive GT depth and reprojection residual<=4 original COLMAP pixels; geometric proxy, not verified query feature track identity/visibility',
          'selection':'all20 queries retained; no thresholds tuned; no training; no production changes',
          'source_sha256':sha(__file__)}
    put(out/'protocol.json',spec);shutil.copyfile(__file__,out/'evaluated_source.py')
    teacher,_=w0.import_original_superpoint();check(w0.CHECKPOINT,manifest['teacher_sha256'])
    teacher=SuperPointKPUWrapper(teacher).eval()
    banks={name:{} for name in spec['arms']};qdesc={name:{} for name in spec['arms']};qxy={};obs={};raw_hashes={};excluded={}
    for name in spec['arms']:(out/name).mkdir()
    for index,i in enumerate(sorted(refids+queries)):
        guard();src=DEST/'inputs'/f'{i}.npz';check(src,entries[i]['input_sha256']);raw_hashes[str(src)]=sha(src)
        with np.load(src) as z:x=z['image'].copy();mask=z['mask'].copy();aff=z['affine'].copy()
        a,b=teacher(torch.from_numpy(x));heads={'teacher_fp32':b}
        for arm,backend in [('student_fp32','fp32'),('student_int8','int8')]:
            file=DEST/backend/f'{i}.npz';check(file,entries[i]['fp32_sha256'] if backend=='fp32' else sim['outputs'][str(i)])
            raw_hashes[str(file)]=sha(file)
            with np.load(file) as z:heads[arm]=torch.from_numpy(z['desc'].copy())
        if i in queries:
            canonical,_=features(a,b,mask,'r2')[0];xy=canonical*1.25+[2,0]
            qxy[i]=(xy-aff[2:])/aff[:2]+.5
            np.testing.assert_allclose((qxy[i]-.5)*aff[:2]+aff[2:],xy,atol=1e-10,rtol=0)
            ids=np.arange(len(xy),dtype=np.int64)
        else:
            im=images[i];valid=im.point3D_ids>=0;ids=im.point3D_ids[valid];source=im.xys[valid]
            u,c=np.unique(ids,return_counts=True);duplicates=u[c>1];excluded[str(i)]=duplicates.tolist()
            keep=~np.isin(ids,duplicates);ids,source=ids[keep],source[keep]
            xy=(source-.5)*aff[:2]+aff[2:]
            ij=np.rint(xy).astype(int);inside=(ij[:,0]>=0)&(ij[:,0]<184)&(ij[:,1]>=0)&(ij[:,1]<320)
            keep=np.zeros(len(ids),bool);keep[inside]=mask[0,ij[inside,1],ij[inside,0]]>0
            ids,xy=ids[keep],xy[keep];order=np.argsort(ids,kind='stable');ids,xy=ids[order],xy[order]
            obs[i]=ids.tolist()
        for arm,head in heads.items():
            desc=ops.sample_descriptors(torch.from_numpy(xy.astype(np.float32))[None],F.normalize(head,dim=1),8)[0].T.numpy()
            assert np.isfinite(desc).all()
            np.savez_compressed(out/arm/f'{i}.npz',ids=ids,network_xy=xy,desc=desc)
            if i in queries:qdesc[arm][i]=desc
            else:banks[arm][i]=(ids,desc)
        if index%20==0:print('EXTRACT',index+1,len(refids)+len(queries),flush=True)
    active={}
    for q,refs in references.items():
        seen=set();ids=[]
        for r in refs:
            for v in obs[r]:
                if v not in seen:seen.add(v);ids.append(v)
        active[q]=ids[:600]
    put(out/'common_inputs.json',{'query_colmap_xy':{str(q):xy.tolist() for q,xy in qxy.items()},
                                'active_ids':active,'excluded_duplicate_ids':excluded,'source_hashes':raw_hashes})
    # Same solver tested independently of all descriptor donors, in COLMAP coordinates.
    sanity=[]
    for q in queries:
        im=images[q];valid=im.point3D_ids>=0;ids=im.point3D_ids[valid];xy=im.xys[valid]
        unique,c=np.unique(ids,return_counts=True);keep=~np.isin(ids,unique[c>1]);ids,xy=ids[keep],xy[keep]
        xyz=np.array([points[int(v)].xyz for v in ids]);r=pose(xyz,xy,cameras[im.camera_id],im);r['query']=q;sanity.append(r)
    put(out/'geometry_sanity.json',{'scope':'internal reconstruction oracle; no network quality claim','rows':sanity})
    if not all(r['good_pose'] for r in sanity):raise RuntimeError('Geometry sanity failed; do not interpret descriptor results')
    allrows=[];details={}
    for arm in spec['arms']:
        rows=[]
        for q in queries:
            im=images[q];cam=cameras[im.camera_id];matches=match(qdesc[arm][q],banks[arm],references[q],active[q])
            ix=np.array([j for j,v,s in matches],int);pid=np.array([v for j,v,s in matches],np.int64)
            xyz=np.array([points[int(v)].xyz for v in pid],np.float64).reshape(-1,3);xy=qxy[q][ix]
            r=pose(xyz,xy,cam,im)
            if len(xyz):
                K,dist=camera(cam);R=qvec2rotmat(im.qvec)
                uv,_=cv2.projectPoints(xyz,cv2.Rodrigues(R)[0],im.tvec,K,dist)
                residual=np.linalg.norm(uv[:,0]-xy,axis=1);depth=(xyz@R.T+im.tvec)[:,2]
                correct=(residual<=4)&(depth>0)
            else:residual=np.array([]);correct=np.array([],bool)
            r.update(arm=arm,query=q,keypoints=len(qxy[q]),active_points=len(active[q]),
                     correct_2d3d_geometric=int(correct.sum()),geometric_precision=float(correct.mean()) if len(correct) else 0.)
            rows.append(r);allrows.append(r)
            details[f'{arm}_{q}']={'query_feature_indices':ix.tolist(),'point3d_ids':pid.tolist(),'gt_reprojection_px':residual.tolist()}
        summary={'queries':len(rows),'good_poses':sum(r['good_pose'] for r in rows),
                 'solver_successes':sum(r['solver_ok'] for r in rows),'accepted_poses':sum(r['status']=='pose' for r in rows),
                 'failures':sum(not r['good_pose'] for r in rows),
                 'median_rotation_deg_returned':float(np.median([r['rotation_deg'] for r in rows if r['rotation_deg'] is not None])) if any(r['rotation_deg'] is not None for r in rows) else None,
                 'median_center_native_returned':float(np.median([r['center_native'] for r in rows if r['center_native'] is not None])) if any(r['center_native'] is not None for r in rows) else None}
        for key in ('matches','correct_2d3d_geometric','inliers','inlier_ratio','geometric_precision'):
            summary['mean_'+key]=float(np.mean([r[key] for r in rows]));summary['median_'+key]=float(np.median([r[key] for r in rows]))
        put(out/f'{arm}_results.json',{'summary':summary,'rows':rows});print(arm,json.dumps(summary),flush=True)
    with (out/'per_query.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(allrows[0]));writer.writeheader();writer.writerows(allrows)
    put(out/'match_details.json',details)
    check(Path(__file__),spec['source_sha256']);check(Path(manifest['checkpoint']),manifest['checkpoint_sha256'])
    for file,digest in base['geometry_hashes'].items():check(MODEL/file,digest)
    put(out/'completion.json',{'status':'COMPLETE','geometry_sanity_good':sum(r['good_pose'] for r in sanity),
                              'queries':20,'training':False,'production_map_changed':False,'board':'NOT_MEASURED'})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True,type=Path);run(p.parse_args().output)
