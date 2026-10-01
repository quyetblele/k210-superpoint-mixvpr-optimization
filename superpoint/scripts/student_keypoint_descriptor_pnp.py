"""Frozen current-student FP32 keypoints, teacher vs student descriptor donor. Internal DEV only."""
import _bootstrap
import argparse, csv, shutil
import numpy as np
import torch.nn.functional as F
from keypoint_cause_audit import FIX, RAW, geometry_stats, scoremap, select
from fixed_keypoint_pnp import read, check, camera, pose, match, context, MODEL, DEST, read_points3D_binary, qvec2rotmat
from spk210.runtime import torch, cv2, ops, w0, SuperPointKPUWrapper
from spk210.settings import SUPERPOINT
from spk210.io import sha, put, guard
from pathlib import Path

@torch.inference_mode()
def run(out):
    out.mkdir(parents=True, exist_ok=False)
    prior=SUPERPOINT/'artifacts/keypoint_cause_audit/run_003'
    pe=read(prior/'evidence_hashes.json'); fe=read(FIX/'evidence_hashes.json')
    for f in ['keypoints_and_matches.json','results.json','protocol.json']: check(prior/f,pe[f])
    for f in ['protocol.json','common_inputs.json']: check(FIX/f,fe[f])
    p=read(FIX/'protocol.json'); common=read(FIX/'common_inputs.json'); old=read(prior/'results.json'); loc=read(prior/'keypoints_and_matches.json')
    qids=p['query_ids']; refs={int(k):v for k,v in p['references'].items()}; rids=sorted(set(r for rr in refs.values() for r in rr))
    assert len(qids)==20 and not set(qids)&set(rids)
    rawresult=read(RAW/'results.json'); sim=read(RAW/'simulation.json'); rs=read(RAW/'protocol.json')
    check(RAW/'protocol.json',rawresult['protocol_sha256'])
    check(SUPERPOINT/'artifacts/descriptor_kd_ab/desc_control_r2_cw/best.pt',rs['checkpoint_sha256'])
    check(SUPERPOINT/'artifacts/deployment/desc_control_r2_cw/model.kmodel',sim['kmodel_sha256'])
    default=read(SUPERPOINT/'artifacts/quality_pilot/summary.json')['selected_for_next_gate'];assert default=='wider_r2_cw'
    original=torch.load(SUPERPOINT/'artifacts/quality_pilot/wider_r2_cw/best.pt',weights_only=False)['model']
    control=torch.load(SUPERPOINT/'artifacts/descriptor_kd_ab/desc_control_r2_cw/best.pt',weights_only=False)['model']
    assert original.keys()==control.keys() and all(torch.equal(original[k],control[k]) for k in original)
    images,cameras=context(); points=read_points3D_binary(MODEL/'points3D.bin')
    for f,d in p['geometry_hashes'].items(): check(MODEL/f,d)
    spec={'scope':'20 nearby DEV queries; fixed default student FP32 keypoints, descriptor donor only; no training or promotion',
          'arms':['teacher_fp32','student_fp32'], 'base_protocol':p, 'default_candidate':default, 'control_weights_exact_default':True,
          'keypoint_source_sha256':sha(prior/'keypoints_and_matches.json'), 'keypoint_variant':'fp32_baseline',
          'student_protocol':rs,'kmodel_sha256':sim['kmodel_sha256'],
          'sampler':'unchanged upstream sample_descriptors, normalized dense head, stride8; donor on both query/reference',
          'source_sha256':sha(__file__), 'prior_results_sha256':sha(prior/'results.json')}
    put(out/'protocol.json',spec); shutil.copyfile(__file__,out/'evaluated_source.py')
    check(w0.CHECKPOINT,p['teacher_sha256']); teacher,_=w0.import_original_superpoint(); teacher=SuperPointKPUWrapper(teacher).eval()
    banks={a:{} for a in spec['arms']}; queries={a:{} for a in spec['arms']}; xy={}; hashes={}
    def sample(head,coords):
        d=ops.sample_descriptors(torch.from_numpy(coords.astype(np.float32))[None], F.normalize(head,dim=1),8)[0].T.numpy()
        assert np.isfinite(d).all(); return d
    for a in spec['arms']: (out/a).mkdir()
    for n,i in enumerate(sorted(qids+rids)):
        guard(); f=DEST/'inputs'/f'{i}.npz'; check(f,sim['records'][str(i)]['input_sha256']); hashes[str(f)]=sha(f)
        with np.load(f) as z: image=z['image'].copy(); aff=z['affine'].copy(); mask=z['mask'][0].copy()
        if i in qids:
            saved=loc[f'fp32_baseline_{i}']; coords=np.asarray(saved['network_xy']); xy[i]=np.asarray(saved['pnp_colmap_xy'])
            f=RAW/'fp32'/f'{i}.npz';check(f,rawresult['results']['fp32']['raw_hashes'][str(i)])
            with np.load(f) as z: fresh=select(scoremap(torch.from_numpy(z['logits'].copy())),mask,'baseline',image[0,0])
            np.testing.assert_array_equal(coords,fresh)
            np.testing.assert_allclose((coords-aff[2:])/aff[:2]+.5,xy[i],atol=1e-10,rtol=0)
            ij=np.rint(coords).astype(int); assert (mask[ij[:,1],ij[:,0]]>0).all()
            ids=np.arange(len(coords)); td=sample(teacher(torch.from_numpy(image))[1],coords)
        else:
            f=FIX/'teacher_fp32'/f'{i}.npz'; check(f,fe[str(f.relative_to(FIX))]); hashes[str(f)]=sha(f)
            with np.load(f) as z: coords=z['network_xy'].copy(); ids=z['ids'].copy(); td=z['desc'].copy()
        donors={'teacher_fp32':td}
        for backend in ['fp32']:
            f=RAW/backend/f'{i}.npz'; check(f,rawresult['results'][backend]['raw_hashes'][str(i)]); hashes[str(f)]=sha(f)
            with np.load(f) as z: donors['student_'+backend]=sample(torch.from_numpy(z['desc'].copy()),coords)
        for a,d in donors.items():
            np.savez_compressed(out/a/f'{i}.npz',ids=ids,network_xy=coords,desc=d)
            if i in qids: queries[a][i]=d
            else: banks[a][i]=(ids,d)
        if n%15==0: print('EXTRACT',n+1,54,flush=True)
    put(out/'common_inputs.json',{'query_colmap_xy':{str(q):v.tolist() for q,v in xy.items()},'active_ids':common['active_ids'],'references':refs,'input_hashes':hashes})
    rows=[]; details={}; summary={}; replay=0
    for a in spec['arms']:
        ar=[]
        for q in qids:
            im=images[q]; cam=cameras[im.camera_id]; active=common['active_ids'][str(q)]
            mm=match(queries[a][q],banks[a],refs[q],active)
            ix=np.array([j for j,v,s in mm],int); xyz=np.array([points[v].xyz for j,v,s in mm]).reshape(-1,3); pix=xy[q][ix]
            row=pose(xyz,pix,cam,im); assert row==pose(xyz,pix,cam,im); replay+=1
            residual=np.empty(0); correct=np.empty(0,bool)
            if len(mm):
                K,dist=camera(cam); R=qvec2rotmat(im.qvec); uv,_=cv2.projectPoints(xyz,cv2.Rodrigues(R)[0],im.tvec,K,dist)
                residual=np.linalg.norm(uv[:,0]-pix,axis=1); correct=(residual<=4)&((xyz@R.T+im.tvec)[:,2]>0)
            if a!='teacher_fp32':
                backend=a.replace('student_',''); prev=next(r for r in old['rows'] if r['query']==q and r['backend']==backend and r['variant']=='baseline')
                for k,v in row.items(): assert v==prev[k],(a,q,k,v,prev[k])
                dd=loc[f'{backend}_baseline_{q}']; assert ix.tolist()==dd['matched_query_indices'] and [v for j,v,s in mm]==dd['matched_point3d_ids']
            geo,_,_=geometry_stats(xy[q],im,cam,points,active)
            row.update(arm=a,query=q,keypoints=len(xy[q]),correct_matches=int(correct.sum()),oracle_good_pose=geo['oracle_good_pose'])
            ar.append(row); rows.append(row)
            details[f'{a}_{q}']={'query_indices':ix.tolist(),'point3d_ids':[v for j,v,s in mm],'scores':[s for j,v,s in mm],'gt_reprojection_px':residual.tolist()}
        s={'queries':len(ar),'good_poses':sum(r['good_pose'] for r in ar),'accepted_poses':sum(r['status']=='pose' for r in ar),'oracle_good_poses':sum(r['oracle_good_pose'] for r in ar)}
        for k in ['matches','correct_matches','inliers','inlier_ratio']: s['mean_'+k]=float(np.mean([r[k] for r in ar]))
        for k in ['rotation_deg','center_native']:
            vals=[r[k] for r in ar if r[k] is not None]; s['median_'+k+'_returned']=float(np.median(vals)) if vals else None
        summary[a]=s; print(a,s,flush=True)
    assert summary['student_fp32']['good_poses']==0
    paired={}
    for other in ['student_fp32']:
        t={r['query'] for r in rows if r['arm']=='teacher_fp32' and r['good_pose']}; s={r['query'] for r in rows if r['arm']==other and r['good_pose']}
        paired[other]={'teacher_only':sorted(t-s),'student_only':sorted(s-t),'both':sorted(t&s),'neither':sorted(set(qids)-(t|s))}
    put(out/'results.json',{'summary':summary,'rows':rows,'paired':paired})
    put(out/'match_details.json',details)
    with (out/'per_query.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    put(out/'checks.json',{'status':'PASS','pnp_deterministic_replays':replay,'student_prior_exact_replays':20,'same_keypoints_and_mask':True,'hash_checks':True})
    put(out/'evidence_hashes.json',{str(f.relative_to(out)):sha(f) for f in sorted(out.rglob('*')) if f.is_file()})

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--output',type=Path,required=True); run(p.parse_args().output)
