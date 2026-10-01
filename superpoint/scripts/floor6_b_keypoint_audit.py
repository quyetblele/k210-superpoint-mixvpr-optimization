"""B FP32/INT8 keypoint audit with a frozen INT8 descriptor field and bank."""
import _bootstrap
import argparse,csv,shutil
from pathlib import Path
import numpy as np
import torch.nn.functional as F
from spk210.runtime import torch,ops
from spk210.io import sha,put
from spk210.settings import SUPERPOINT
from spk210.map_nms_gate import context,MODEL,read_points3D_binary,qvec2rotmat
from spk210.map_geometry import pose,project
from spk210.metrics import features
from fixed_keypoint_pnp import read,check,match
from keypoint_cause_audit import scoremap,select

@torch.inference_mode()
def run(out):
    out.mkdir(parents=True,exist_ok=False)
    experiment=SUPERPOINT/'artifacts/student_point_supervision_ab';raw=experiment/'point_geometry_r2_cw/floor6';base=SUPERPOINT/'artifacts/floor6_teacher_baseline/run_001'
    be=read(base/'evidence_hashes.json');re=read(raw/'evidence_hashes.json');ee=read(experiment/'evidence_hashes.json')
    for f in ['protocol.json','common_inputs.json']:check(base/f,be[f])
    for f in ['protocol.json','results.json','simulation.json']:check(raw/f,re[f])
    diagnostic=experiment/'point_geometry_r2_cw/descriptor_diagnostics.json';check(diagnostic,ee[str(diagnostic.relative_to(experiment))])
    p=read(base/'protocol.json');rp=read(raw/'protocol.json');common=read(base/'common_inputs.json');prior=read(raw/'results.json');donor=read(diagnostic)
    check(SUPERPOINT/'artifacts/deployment/point_geometry_r2_cw/model.kmodel',rp['compile']['kmodel_sha256']);check(SUPERPOINT/'src/spk210/map_geometry.py',rp['geometry_source_sha256'])
    queries=p['query_ids'];refs={int(q):r for q,r in common['references'].items()};ids=sorted(set(queries+[i for rr in refs.values() for i in rr]))
    spec={'scope':'canonical floor6 DEV, recipe B; no training or automatic promotion','candidate':'point_geometry_r2_cw','compile':rp['compile'],'fixed':'INT8 dense descriptor field and INT8 reference bank; same query set, map, references/active IDs, matcher/PnP','arms':['fp32_keypoints','int8_keypoints','int8_greedy_nms'],'greedy':'single exploratory NMS-only intervention, score descending ties y/x, strict square radius4 exclusion, unchanged threshold.005 andcap160; no parameter search','source_sha256':sha(__file__),'baseline_protocol_sha256':sha(base/'protocol.json')}
    put(out/'protocol.json',spec);shutil.copyfile(__file__,out/'evaluated_source.py')
    images,cameras=context();points=read_points3D_binary(MODEL/'points3D.bin');bank={};head={};inputs={};hashes={}
    for i in ids:
        f=raw/'int8'/f'{i}.npz';check(f,re[str(f.relative_to(raw))]);hashes[str(f)]=sha(f)
        with np.load(f) as z:a=torch.from_numpy(z['logits'].copy());b=torch.from_numpy(z['desc'].copy())
        if i in queries:
            f=base/'inputs'/f'{i}.npz';check(f,be[str(f.relative_to(base))]);hashes[str(f)]=sha(f)
            with np.load(f) as z:inputs[i]=(z['image'].copy(),z['mask'].copy(),z['affine'].copy())
            f=raw/'fp32'/f'{i}.npz';check(f,re[str(f.relative_to(raw))]);hashes[str(f)]=sha(f)
            with np.load(f) as z:af=torch.from_numpy(z['logits'].copy())
            head[i]=(af,a,b)
        else:
            f=base/'features'/f'{i}.npz';check(f,be[str(f.relative_to(base))]);hashes[str(f)]=sha(f)
            with np.load(f) as z:pid=z['ids'].copy();xy=z['network_xy'].copy()
            d=ops.sample_descriptors(torch.from_numpy(xy.astype(np.float32))[None],F.normalize(b,dim=1),8)[0].T.numpy();bank[i]=(pid,d)
    rows=[];details={}
    for mode in spec['arms']:
        for q in queries:
            af,ai,b=head[q];x,mask,aff=inputs[q];a=af if mode=='fp32_keypoints' else ai;s=scoremap(a);score=s[0].numpy()
            xy=select(s,mask[0],'greedy_nms' if mode=='int8_greedy_nms' else 'baseline',x[0,0]);source=(xy-aff[2:])/aff[:2]+.5
            d=ops.sample_descriptors(torch.from_numpy(xy.astype(np.float32))[None],F.normalize(b,dim=1),8)[0].T.numpy()
            if mode!='int8_greedy_nms':
                canonical,_=features(a,b,mask,'r2')[0];np.testing.assert_allclose(canonical*1.25+[2,0],xy,atol=1e-10,rtol=0)
                f=raw/('fp32' if mode=='fp32_keypoints' else 'int8')/f'query_features_{q}.npz';check(f,re[str(f.relative_to(raw))])
                with np.load(f) as z:np.testing.assert_allclose(source,z['own_colmap_xy'],atol=1e-10,rtol=0)
            mm=match(d,bank,refs[q],common['active_ids'][str(q)]);ix=np.array([j for j,v,s in mm],int);pid=[v for j,v,s in mm];xyz=np.array([points[v].xyz for v in pid]).reshape(-1,3);im=images[q];cam=cameras[im.camera_id];row=pose(xyz,source[ix],cam,im);assert row==pose(xyz,source[ix],cam,im)
            if mode!='int8_greedy_nms':
                old=next(r for r in (donor['rows'] if mode=='fp32_keypoints' else prior['rows']) if r['query']==q and ((r.get('mode')=='own' and r.get('donor')=='int8') if mode=='fp32_keypoints' else (r.get('mode')=='own' and r.get('backend')=='int8')))
                for k,v in row.items():assert old[k]==v,(q,mode,k)
            err=np.linalg.norm(project(xyz,im,cam)-source[ix],axis=1) if len(mm) else np.empty(0);depth=(xyz@qvec2rotmat(im.qvec).T+im.tvec)[:,2]
            delta=np.max(np.abs(xy[:,None]-xy[None]),axis=2);np.fill_diagonal(delta,np.inf);cluster=float((delta.min(1)<=4).mean()) if len(xy)>1 else 0.
            if mode=='int8_greedy_nms':assert cluster==0.
            bins=np.clip((xy/[184,320]*4).astype(int),0,3);grid=np.bincount(bins[:,1]*4+bins[:,0],minlength=16);ij=xy.astype(int);scores=score[ij[:,1],ij[:,0]]
            observed=set(int(v) for v in im.point3D_ids if v>=0);active=sorted(observed&set(common['active_ids'][str(q)]));uv=project(np.array([points[v].xyz for v in active]),im,cam)
            finite=np.isfinite(uv).all(1);uv=uv[finite];nearest=np.linalg.norm(source[:,None]-uv[None],axis=2).min(1) if len(uv) else np.full(len(source),np.inf)
            row.update(query=q,mode=mode,keypoints=len(xy),matches=len(mm),correct_matches=int(((err<=4)&(depth>0)).sum()),inlier_ratio=row['inliers']/len(mm) if mm else 0.,cluster_fraction=cluster,occupied_grid_cells=int((grid>0).sum()),useful_keypoints_4px=int((nearest<=4).sum()),median_score=float(np.median(scores)) if len(scores) else None,unique_selected_scores=len(np.unique(scores)))
            rows.append(row);details[f'{mode}_{q}']={'network_xy':xy.tolist(),'colmap_xy':source.tolist(),'query_indices':ix.tolist(),'point3d_ids':pid,'grid':grid.tolist()}
    summary={}
    for mode in spec['arms']:
        rr=[r for r in rows if r['mode']==mode];s={'good_poses':sum(r['good_pose'] for r in rr),'queries':len(rr)}
        for k in ['keypoints','matches','correct_matches','inliers','inlier_ratio','cluster_fraction','occupied_grid_cells','useful_keypoints_4px','median_score','unique_selected_scores']:s['mean_'+k]=float(np.mean([r[k] for r in rr]))
        summary[mode]=s;print(mode,s,flush=True)
    put(out/'results.json',{'summary':summary,'rows':rows});put(out/'details.json',details)
    with (out/'per_query.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    put(out/'checks.json',{'status':'PASS','fixed_descriptor_bank':True,'baseline_replays':40,'deterministic_pnp_replays':60,'greedy_exclusion_pass':True,'input_hashes':hashes})
    put(out/'evidence_hashes.json',{str(f.relative_to(out)):sha(f) for f in sorted(out.rglob('*')) if f.is_file()})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);run(p.parse_args().output)
