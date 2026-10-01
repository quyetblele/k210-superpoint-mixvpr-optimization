"""Bounded score/rank/K/coverage diagnostics; same B INT8 descriptor field/bank."""
import _bootstrap
import argparse,shutil
from pathlib import Path
import numpy as np
import torch.nn.functional as F
from spk210.runtime import torch,ops
from spk210.settings import SUPERPOINT
from spk210.io import sha,put
from spk210.metrics import features
from spk210.postprocess import greedy_keypoints
from spk210.map_nms_gate import context,MODEL,read_points3D_binary,qvec2rotmat
from spk210.map_geometry import pose,project
from fixed_keypoint_pnp import read,check,match
from keypoint_cause_audit import scoremap

@torch.inference_mode()
def run(out):
    out.mkdir(parents=True,exist_ok=False);raw=SUPERPOINT/'artifacts/student_point_supervision_ab/point_geometry_r2_cw/floor6';base=SUPERPOINT/'artifacts/floor6_teacher_baseline/run_001';reg=SUPERPOINT/'artifacts/greedy_nms_regression/run_001'
    be=read(base/'evidence_hashes.json');re=read(raw/'evidence_hashes.json');ge=read(reg/'evidence_hashes.json')
    for f in ['protocol.json','common_inputs.json']:check(base/f,be[f])
    check(reg/'results.json',ge['results.json']);prior=read(reg/'results.json');rp=read(raw/'protocol.json');check(raw/'protocol.json',re['protocol.json']);check(SUPERPOINT/'artifacts/deployment/point_geometry_r2_cw/model.kmodel',rp['compile']['kmodel_sha256'])
    p=read(base/'protocol.json');common=read(base/'common_inputs.json');qs=p['query_ids'];refs={int(q):r for q,r in common['references'].items()};ids=sorted(set(qs+[i for rr in refs.values() for i in rr]))
    variants=['baseline','top320','threshold001','grid10','teacher_rerank','fp32_keypoints']
    spec={'scope':'one bounded exploratory audit,20 canonical floor6 DEV, no training/promotion','candidate':'point_geometry_r2_cw','variants':variants,'fixed':'B INT8 descriptor field/reference bank, map/retrieval/activeIDs/matcher/PnP','baseline':'greedy NMS4 threshold.005 top160','top320':'only cap320; same .005 score/ordering','threshold001':'only threshold.001; cap160 unchanged','grid10':'full .005 greedy survivors, up to10 per4x4cell in original score order, max160; count may decrease','teacher_rerank':'diagnostic teacher-score ordering of unchanged full .005 greedy survivor pool, cap160; not deployable teacher-free','fp32_keypoints':'FP32 greedy .005 top160 with same INT8 descriptor donor, isolates detector from descriptor quantization','geometry':'positive-depth query-observed active Point3D projections; original-image pixel thresholds4/8; no externalGT','ranking_probe':'labels on DEV for diagnosis only; no thresholds selected posthoc','source_sha256':sha(__file__)}
    put(out/'protocol.json',spec);shutil.copyfile(__file__,out/'evaluated_source.py');images,cameras=context();points=read_points3D_binary(MODEL/'points3D.bin');bank={};heads={};inputs={};hashes={}
    for i in ids:
        f=raw/'int8'/f'{i}.npz';check(f,re[str(f.relative_to(raw))]);hashes[str(f)]=sha(f)
        with np.load(f) as z:ai=torch.from_numpy(z['logits'].copy());b=torch.from_numpy(z['desc'].copy())
        if i in qs:
            f=base/'inputs'/f'{i}.npz';check(f,be[str(f.relative_to(base))]);hashes[str(f)]=sha(f)
            with np.load(f) as z:inputs[i]=(z['mask'].copy(),z['affine'].copy())
            f=raw/'fp32'/f'{i}.npz';check(f,re[str(f.relative_to(raw))]);hashes[str(f)]=sha(f)
            with np.load(f) as z:af=torch.from_numpy(z['logits'].copy())
            f=base/'teacher'/f'{i}.npz';check(f,be[str(f.relative_to(base))]);hashes[str(f)]=sha(f)
            with np.load(f) as z:at=torch.from_numpy(z['logits'].copy())
            heads[i]=(ai,af,at,b)
        else:
            f=base/'features'/f'{i}.npz';check(f,be[str(f.relative_to(base))]);hashes[str(f)]=sha(f)
            with np.load(f) as z:pid=z['ids'].copy();xy=z['network_xy'].copy()
            bank[i]=(pid,ops.sample_descriptors(torch.from_numpy(xy.astype(np.float32))[None],F.normalize(b,dim=1),8)[0].T.numpy())
    rows=[];audit=[];details={}
    for q in qs:
        mask,aff=inputs[q];ai,af,at,b=heads[q];si=scoremap(ai)[0]*torch.from_numpy(mask[0]);sf=scoremap(af)[0]*torch.from_numpy(mask[0]);st=scoremap(at)[0].numpy()
        full=greedy_keypoints(si,4,.005,10000);low=greedy_keypoints(si,4,.001,10000);fp=greedy_keypoints(sf,4,.005,160);baseline=full[:160]
        grid=[];counts=np.zeros(16,int)
        for xy in full:
            gx,gy=np.minimum((xy/[184,320]*4).astype(int),3);cell=gy*4+gx
            if counts[cell]<10:grid.append(xy);counts[cell]+=1
        grid=np.asarray(grid,np.float32).reshape(-1,2);ij=full.astype(int);ts=st[ij[:,1],ij[:,0]];order=np.lexsort((full[:,0],full[:,1],-ts))
        variants_xy={'baseline':baseline,'top320':full[:320],'threshold001':low[:160],'grid10':grid,'teacher_rerank':full[order[:160]],'fp32_keypoints':fp}
        im=images[q];cam=cameras[im.camera_id];visible=set(int(v) for v in im.point3D_ids if v>=0);pids=sorted(visible&set(common['active_ids'][str(q)]));xyzobs=np.array([points[v].xyz for v in pids]);uv=project(xyzobs,im,cam);depth=(xyzobs@qvec2rotmat(im.qvec).T+im.tvec)[:,2];uv=uv[np.isfinite(uv).all(1)&(depth>0)]
        def spatial(xy):
            source=(xy-aff[2:])/aff[:2]+.5;dist=np.linalg.norm(source[:,None]-uv[None],axis=2);nearest=dist.min(1) if len(xy) else np.empty(0)
            return source,nearest
        _,fd=spatial(full);_,ld=spatial(low);scores=si.numpy()[ij[:,1],ij[:,0]]
        positions=np.rint((uv-.5)*aff[:2]+aff[2:]).astype(int);valid=(positions[:,0]>=0)&(positions[:,0]<184)&(positions[:,1]>=0)&(positions[:,1]<320);positions=positions[valid]
        audit.append({'query':q,'candidate_count005':len(full),'candidate_count001':len(low),'topk_binding':len(full)>160,'baseline_count':len(baseline),'useful4_in_top160':int((fd[:160]<=4).sum()),'useful4_beyond160':int((fd[160:]<=4).sum()),'useful4_full005':int((fd<=4).sum()),'useful4_full001':int((ld<=4).sum()),'useful8_full005':int((fd<=8).sum()),'useful_candidate_ranks005':(np.flatnonzero(fd<=4)+1).tolist(),'score_quantiles005':np.quantile(scores,[0,.1,.5,.9,1]).tolist() if len(scores) else [],'teacher_points':read(base/'results.json')['rows'][qs.index(q)]['keypoints']})
        for mode,xy in variants_xy.items():
            source,near=spatial(xy);d=ops.sample_descriptors(torch.from_numpy(xy)[None],F.normalize(b,dim=1),8)[0].T.numpy();mm=match(d,bank,refs[q],common['active_ids'][str(q)]);ix=np.array([j for j,v,s in mm],int);pid=[v for j,v,s in mm];xyz=np.array([points[v].xyz for v in pid]).reshape(-1,3);row=pose(xyz,source[ix],cam,im);assert row==pose(xyz,source[ix],cam,im)
            if mode=='baseline':
                old=next(r for r in prior['floor6']['int8/greedy']['rows'] if r['query']==q)
                for k,v in row.items():assert old[k]==v
            err=np.linalg.norm(project(xyz,im,cam)-source[ix],axis=1) if mm else np.empty(0);dep=(xyz@qvec2rotmat(im.qvec).T+im.tvec)[:,2];bins=np.clip((xy/[184,320]*4).astype(int),0,3);hist=np.bincount(bins[:,1]*4+bins[:,0],minlength=16)
            row.update(query=q,mode=mode,keypoints=len(xy),matches=len(mm),correct_matches=int(((err<=4)&(dep>0)).sum()),inlier_ratio=row['inliers']/len(mm) if mm else 0.,useful4=int((near<=4).sum()),useful8=int((near<=8).sum()),occupied_cells=int((hist>0).sum()))
            rows.append(row);details[f'{q}_{mode}']={'network_xy':xy.tolist(),'query_indices':ix.tolist(),'point3d_ids':pid,'grid':hist.tolist()}
    summaries={}
    for mode in variants:
        rr=[r for r in rows if r['mode']==mode];s={'good_poses':sum(r['good_pose'] for r in rr),'success_ids':[r['query'] for r in rr if r['good_pose']]}
        for k in ['keypoints','matches','correct_matches','inliers','inlier_ratio','useful4','useful8','occupied_cells']:s['mean_'+k]=float(np.mean([r[k] for r in rr]))
        summaries[mode]=s;print(mode,s,flush=True)
    put(out/'results.json',{'summary':summaries,'rows':rows,'ranking_audit':audit});put(out/'details.json',details)
    put(out/'checks.json',{'status':'PASS','baseline_exact_replays':20,'deterministic_pnp_replays':120,'input_hashes':hashes,'default_changed':False})
    put(out/'evidence_hashes.json',{str(f.relative_to(out)):sha(f) for f in sorted(out.rglob('*')) if f.is_file()})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True,type=Path);run(p.parse_args().output)
