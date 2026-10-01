"""Frozen floor6 teacher-keypoint and own-keypoint student FP32/real nncase evaluation."""
import _bootstrap
import argparse,json,shutil,csv,os
from pathlib import Path
import numpy as np
from spk210.settings import SUPERPOINT
from spk210.io import sha,put,guard

def read(p):return json.loads(Path(p).read_text())
def check(p,h):assert sha(p)==h,str(p)

def context(out):
    config=read(SUPERPOINT/'configs/localization_map.json');base=Path(config['teacher_baseline']);check(base/'protocol.json',config['teacher_baseline_protocol_sha256'])
    p=read(base/'protocol.json');e=read(base/'evidence_hashes.json')
    for f in ['common_inputs.json','results.json']:check(base/f,e[f])
    common=read(base/'common_inputs.json');queries=p['query_ids'];refs={int(q):r for q,r in common['references'].items()};ids=sorted(set(queries+[i for rr in refs.values() for i in rr]))
    candidate=os.environ.get('SP_STUDENT_CANDIDATE') or read(SUPERPOINT/'artifacts/quality_pilot/summary.json')['selected_for_next_gate']
    dep=SUPERPOINT/'artifacts/deployment'/candidate;compile=read(dep/'compile.json')
    check(dep/'model.kmodel',compile['kmodel_sha256'])
    registry=read(SUPERPOINT/'configs/candidates.json');checkpoint=SUPERPOINT/registry[candidate]['checkpoint'] if candidate in registry else SUPERPOINT/'artifacts/quality_pilot'/candidate/'best.pt'
    check(checkpoint,compile['checkpoint_sha256'])
    return base,p,e,common,queries,refs,ids,candidate,dep,compile

def simulate(out):
    import nncase
    out.mkdir(parents=True,exist_ok=False);base,p,e,common,queries,refs,ids,candidate,dep,compile=context(out)
    spec={'scope':'same frozen floor6 DEV, student FP32 vs actual nncase INT8, fixed-teacher and own-keypoint modes; no training','teacher_protocol_sha256':sha(base/'protocol.json'),'common_inputs_sha256':sha(base/'common_inputs.json'),'candidate':candidate,'compile':compile,'source_sha256':sha(__file__),'geometry_source_sha256':sha(SUPERPOINT/'src/spk210/map_geometry.py'),'map_hashes':p['map_config']['geometry_hashes'],'calibration':'reuse original frozen TRAIN calibration and kmodel; no calibration on floor6 DEV','no_threshold_changes':True}
    put(out/'protocol.json',spec);shutil.copyfile(__file__,out/'evaluated_source.py')
    sim=nncase.Simulator();sim.load_model((dep/'model.kmodel').read_bytes());assert sim.get_input_tensor(0).dtype==np.dtype('float32');assert all(sim.get_output_tensor(j).dtype==np.dtype('float32') for j in range(2))
    perm=np.r_[np.rot90(np.arange(64).reshape(8,8),-1).ravel(),64];(out/'int8').mkdir();records={}
    for k,i in enumerate(ids):
        guard();f=base/'inputs'/f'{i}.npz';check(f,e[str(f.relative_to(base))])
        with np.load(f) as z:x=np.ascontiguousarray(np.rot90(z['image'],-1,(2,3)))
        sim.set_input_tensor(0,nncase.RuntimeTensor.from_numpy(x));sim.run()
        a,b=[np.rot90(sim.get_output_tensor(j).to_numpy().copy(),1,(2,3)) for j in range(2)];a=a[:,np.argsort(perm)]
        assert a.shape==(1,65,40,23) and b.shape==(1,256,40,23) and np.isfinite(a).all() and np.isfinite(b).all()
        dest=out/'int8'/f'{i}.npz';np.savez_compressed(dest,logits=a,desc=b);records[i]={'input_sha256':sha(f),'output_sha256':sha(dest)}
        if (k+1)%40==0:print('SIM',k+1,len(ids),flush=True)
    put(out/'simulation.json',{'status':'PASS','backend':'actual nncase K210 Simulator','kmodel_sha256':compile['kmodel_sha256'],'records':records})

def evaluate(out):
    import torch
    import torch.nn.functional as F
    from spk210.runtime import ops
    from spk210.onnx_export import load_model
    from spk210.metrics import features
    from spk210.map_nms_gate import context as map_context,MODEL,read_points3D_binary,qvec2rotmat
    from spk210.map_geometry import pose,project
    from fixed_keypoint_pnp import match
    base,p,e,common,queries,refs,ids,candidate,dep,compile=context(out);spec=read(out/'protocol.json');check(__file__,spec['source_sha256']);check(SUPERPOINT/'src/spk210/map_geometry.py',spec['geometry_source_sha256'])
    sim=read(out/'simulation.json');assert sim['kmodel_sha256']==compile['kmodel_sha256']
    model,res,checkpoint=load_model(candidate);assert res=='r2';check(checkpoint,compile['checkpoint_sha256'])
    images,cameras=map_context();points=read_points3D_binary(MODEL/'points3D.bin');allrows=[];summary={};details={};replays=0
    for backend in ['fp32','int8']:
        (out/backend).mkdir(exist_ok=True);banks={};qd={};own={};qxy={};rawhashes={}
        with torch.inference_mode():
            for i in ids:
                f=base/'inputs'/f'{i}.npz';check(f,sim['records'][str(i)]['input_sha256'])
                with np.load(f) as z:x=z['image'].copy();mask=z['mask'].copy();aff=z['affine'].copy()
                raw=out/backend/f'{i}.npz'
                if backend=='fp32':
                    a,b=model(torch.from_numpy(x));np.savez_compressed(raw,logits=a.numpy(),desc=b.numpy())
                else:
                    check(raw,sim['records'][str(i)]['output_sha256'])
                    with np.load(raw) as z:a=torch.from_numpy(z['logits'].copy());b=torch.from_numpy(z['desc'].copy())
                rawhashes[i]=sha(raw);f=base/'features'/f'{i}.npz';check(f,e[str(f.relative_to(base))])
                with np.load(f) as z:pid=z['ids'].copy();network=z['network_xy'].copy()
                d=ops.sample_descriptors(torch.from_numpy(network.astype(np.float32))[None],F.normalize(b,dim=1),8)[0].T.numpy()
                assert np.isfinite(d).all()
                if i in queries:
                    qd[i]=d;canonical,desc=features(a,b,mask,'r2')[0];xy=(canonical*1.25+[2,0]-aff[2:])/aff[:2]+.5;own[i]=desc;qxy[i]=xy
                    np.savez_compressed(out/backend/f'query_features_{i}.npz',fixed_network_xy=network,fixed_desc=d,own_colmap_xy=xy,own_desc=desc)
                else:banks[i]=(pid,d)
        for mode in ['fixed_teacher','own']:
            rows=[]
            for q in queries:
                xy=np.array(common['query_colmap_xy'][str(q)]) if mode=='fixed_teacher' else qxy[q];d=qd[q] if mode=='fixed_teacher' else own[q]
                mm=match(d,banks,refs[q],common['active_ids'][str(q)]);ix=np.array([j for j,v,s in mm],int);pid=[v for j,v,s in mm];xyz=np.array([points[v].xyz for v in pid]).reshape(-1,3);im=images[q];cam=cameras[im.camera_id]
                row=pose(xyz,xy[ix],cam,im);assert row==pose(xyz,xy[ix],cam,im);replays+=1
                err=np.linalg.norm(project(xyz,im,cam)-xy[ix],axis=1) if len(mm) else np.empty(0);depth=(xyz@qvec2rotmat(im.qvec).T+im.tvec)[:,2]
                row.update(query=q,backend=backend,mode=mode,keypoints=len(xy),matches=len(mm),correct_geometric_matches=int(((err<=4)&(depth>0)).sum()),inlier_ratio=row['inliers']/len(mm) if mm else 0.)
                rows.append(row);allrows.append(row);details[f'{backend}_{mode}_{q}']={'query_indices':ix.tolist(),'point3d_ids':pid,'scores':[s for j,v,s in mm]}
            s={'good_poses':sum(r['good_pose'] for r in rows),'queries':len(rows)}
            for k in ['keypoints','matches','correct_geometric_matches','inliers','inlier_ratio']:s['mean_'+k]=float(np.mean([r[k] for r in rows]))
            summary[backend+'/'+mode]=s;print(backend,mode,s,flush=True)
        put(out/f'{backend}_raw_hashes.json',rawhashes)
    put(out/'results.json',{'status':'COMPLETE','summary':summary,'rows':allrows});put(out/'match_details.json',details)
    with (out/'per_query.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(allrows[0]));w.writeheader();w.writerows(allrows)
    put(out/'checks.json',{'status':'PASS','pnp_replays':replays,'queries':len(queries),'fresh_images_each_backend':len(ids),'kmodel_verified':True,'teacher_reference_identities_verified':True,'no_training':True})
    put(out/'evidence_hashes.json',{str(f.relative_to(out)):sha(f) for f in sorted(out.rglob('*')) if f.is_file()})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['simulate','evaluate']);p.add_argument('--output',type=Path,required=True);a=p.parse_args();(simulate if a.stage=='simulate' else evaluate)(a.output)
