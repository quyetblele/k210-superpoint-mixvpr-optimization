"""Three descriptor donors at identical current-student FP32 keypoints on floor6."""
import _bootstrap
import argparse,csv,shutil
from pathlib import Path
import numpy as np
import torch.nn.functional as F
from floor6_student_evaluate import context,read,check
from spk210.runtime import torch,ops
from spk210.settings import SUPERPOINT
from spk210.io import sha,put
from spk210.map_nms_gate import context as map_context,MODEL,read_points3D_binary,qvec2rotmat
from spk210.map_geometry import pose,project
from fixed_keypoint_pnp import match

@torch.inference_mode()
def run(out):
    out.mkdir(parents=True,exist_ok=False)
    source=SUPERPOINT/'artifacts/floor6_student_evaluation/run_001';se=read(source/'evidence_hashes.json')
    base,p,be,common,queries,refs,ids,candidate,dep,compile=context(out)
    for f in ['protocol.json','results.json','simulation.json','match_details.json']:check(source/f,se[f])
    sp=read(source/'protocol.json');check(SUPERPOINT/'src/spk210/map_geometry.py',sp['geometry_source_sha256']);assert sp['compile']==compile
    sim=read(source/'simulation.json');old=read(source/'results.json');oldmatches=read(source/'match_details.json')
    spec={'scope':'canonical floor6 DEV; three donors, same student FP32 query keypoints; no training','arms':{'A':'teacher FP32','B':'student FP32','C':'student faithful INT8'},'teacher_protocol_sha256':sha(base/'protocol.json'),'student_protocol_sha256':sha(source/'protocol.json'),'compile':compile,'keypoints':'frozen student FP32 own_colmap_xy for ALL arms including INT8 donor; no INT8 detector here','fixed':'same20query IDs, map, active IDs, reference observation positions, matcher/thresholds/native fisheye PnP','descriptor_donor':'on BOTH query and reference, original stride8 sampler','source_sha256':sha(__file__)}
    put(out/'protocol.json',spec);shutil.copyfile(__file__,out/'evaluated_source.py')
    images,cameras=map_context();points=read_points3D_binary(MODEL/'points3D.bin');banks={a:{} for a in spec['arms']};qd={a:{} for a in spec['arms']};qxy={};locations={};hashes={}
    for i in ids:
        f=base/'inputs'/f'{i}.npz';check(f,be[str(f.relative_to(base))]);hashes[str(f)]=sha(f)
        with np.load(f) as z:aff=z['affine'].copy();mask=z['mask'][0].copy()
        if i in queries:
            f=source/'fp32'/f'query_features_{i}.npz';check(f,se[str(f.relative_to(source))]);hashes[str(f)]=sha(f)
            with np.load(f) as z:qxy[i]=z['own_colmap_xy'].copy();bd=z['own_desc'].copy()
            xy=(qxy[i]-.5)*aff[:2]+aff[2:];pid=np.arange(len(xy));ij=np.rint(xy).astype(int);assert (mask[ij[:,1],ij[:,0]]>0).all()
        else:
            f=base/'features'/f'{i}.npz';check(f,be[str(f.relative_to(base))]);hashes[str(f)]=sha(f)
            with np.load(f) as z:xy=z['network_xy'].copy();pid=z['ids'].copy()
        locations[i]={'ids':pid.tolist(),'network_xy':xy.tolist()}
        for arm,folder in [('A',base/'teacher'),('B',source/'fp32'),('C',source/'int8')]:
            f=folder/f'{i}.npz';check(f,(be if arm=='A' else se)[str(f.relative_to(base if arm=='A' else source))]);hashes[str(f)]=sha(f)
            if arm=='C':check(f,sim['records'][str(i)]['output_sha256'])
            with np.load(f) as z:b=torch.from_numpy(z['desc'].copy())
            d=ops.sample_descriptors(torch.from_numpy(xy.astype(np.float32))[None],F.normalize(b,dim=1),8)[0].T.numpy();assert np.isfinite(d).all()
            if i in queries:
                qd[arm][i]=d
                if arm=='B':np.testing.assert_array_equal(d,bd)
            else:banks[arm][i]=(pid,d)
    rows=[];details={};summary={}
    for arm in spec['arms']:
        rr=[]
        for q in queries:
            mm=match(qd[arm][q],banks[arm],refs[q],common['active_ids'][str(q)]);ix=np.array([j for j,v,s in mm],int);pid=[v for j,v,s in mm];xyz=np.array([points[v].xyz for v in pid]).reshape(-1,3);xy=qxy[q][ix];im=images[q];cam=cameras[im.camera_id]
            row=pose(xyz,xy,cam,im);assert row==pose(xyz,xy,cam,im)
            err=np.linalg.norm(project(xyz,im,cam)-xy,axis=1) if len(mm) else np.empty(0);depth=(xyz@qvec2rotmat(im.qvec).T+im.tvec)[:,2]
            row.update(query=q,arm=arm,keypoints=len(qxy[q]),matches=len(mm),correct_matches=int(((err<=4)&(depth>0)).sum()),inlier_ratio=row['inliers']/len(mm) if mm else 0.)
            if arm=='B':
                prev=next(r for r in old['rows'] if r['query']==q and r['backend']=='fp32' and r['mode']=='own')
                for k in ['good_pose','inliers','rotation_deg','center_native','matches','inlier_ratio']:assert row[k]==prev[k]
                assert row['correct_matches']==prev['correct_geometric_matches'];prev=oldmatches[f'fp32_own_{q}'];assert ix.tolist()==prev['query_indices'] and pid==prev['point3d_ids']
            rr.append(row);rows.append(row);details[f'{arm}_{q}']={'query_indices':ix.tolist(),'point3d_ids':pid,'scores':[s for j,v,s in mm]}
        s={'good_poses':sum(r['good_pose'] for r in rr),'queries':len(rr)}
        for k in ['matches','correct_matches','inliers','inlier_ratio']:s['mean_'+k]=float(np.mean([r[k] for r in rr]))
        summary[arm]=s;print(arm,s,flush=True)
    assert summary['B']['good_poses']==5
    put(out/'results.json',{'summary':summary,'rows':rows});put(out/'match_details.json',details)
    put(out/'common_inputs.json',{'query_colmap_xy':{q:xy.tolist() for q,xy in qxy.items()},'locations_all_donors':locations,'references':refs,'active_ids':common['active_ids'],'input_hashes':hashes})
    with (out/'per_query.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    put(out/'checks.json',{'status':'PASS','pnp_replays':60,'exact_B_prior_replays':20,'B_query_descriptors_bitwise_equal':True,'same_keypoint_and_reference_locations_all_arms':True,'real_INT8_raw_hashes_verified':len(ids),'no_training':True})
    put(out/'evidence_hashes.json',{str(f.relative_to(out)):sha(f) for f in sorted(out.rglob('*')) if f.is_file()})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);run(p.parse_args().output)
