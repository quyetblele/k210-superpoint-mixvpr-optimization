"""Canonical floor6 FP32-only diagnostic, independent of the ONNX/INT8 gate."""
import _bootstrap
import argparse,json
from pathlib import Path
import numpy as np
from spk210.runtime import torch,ops
import torch.nn.functional as F
from spk210.settings import SUPERPOINT
from spk210.io import put,sha
from spk210.onnx_export import load_model
from spk210.metrics import features
from spk210.map_nms_gate import context as map_context,MODEL,read_points3D_binary,qvec2rotmat
from spk210.map_geometry import pose,project
from floor6_student_evaluate import read,check
from fixed_keypoint_pnp import match

@torch.inference_mode()
def run(candidate,out):
    cfg=read(SUPERPOINT/'configs/localization_map.json');base=Path(cfg['teacher_baseline']);check(base/'protocol.json',cfg['teacher_baseline_protocol_sha256']);p=read(base/'protocol.json');e=read(base/'evidence_hashes.json');check(base/'common_inputs.json',e['common_inputs.json']);common=read(base/'common_inputs.json');queries=p['query_ids'];refs={int(q):r for q,r in common['references'].items()};ids=sorted(set(queries+[i for rr in refs.values() for i in rr]))
    model,res,checkpoint=load_model(candidate);assert res=='r2';images,cameras=map_context();points=read_points3D_binary(MODEL/'points3D.bin');banks={};queries_data={};inputs={};rows=[];summaries={};details={};out.mkdir(parents=True,exist_ok=True)
    for i in ids:
        f=base/'inputs'/f'{i}.npz';check(f,e[str(f.relative_to(base))]);inputs[str(f)]=sha(f)
        with np.load(f) as z:x=z['image'].copy();mask=z['mask'].copy();aff=z['affine'].copy()
        a,b=model(torch.from_numpy(x));f=base/'features'/f'{i}.npz';check(f,e[str(f.relative_to(base))])
        with np.load(f) as z:pid=z['ids'].copy();network=z['network_xy'].copy()
        d=ops.sample_descriptors(torch.from_numpy(network.astype(np.float32))[None],F.normalize(b,dim=1),8)[0].T.numpy()
        if i not in queries:banks[i]=(pid,d);continue
        own={}
        for nms in ['original','greedy']:
            xy,desc=features(a,b,mask,'r2',nms=nms)[0];own[nms]=((xy*1.25+[2,0]-aff[2:])/aff[:2]+.5,desc)
        own['fixed_teacher']=(np.array(common['query_colmap_xy'][str(i)]),d);queries_data[i]=own
    for mode in ['fixed_teacher','original','greedy']:
        rr=[]
        for q in queries:
            xy,d=queries_data[q][mode];mm=match(d,banks,refs[q],common['active_ids'][str(q)]);ix=np.array([j for j,v,s in mm],int);pid=[v for j,v,s in mm];xyz=np.array([points[v].xyz for v in pid]).reshape(-1,3);im=images[q];cam=cameras[im.camera_id];r=pose(xyz,xy[ix],cam,im);assert r==pose(xyz,xy[ix],cam,im)
            err=np.linalg.norm(project(xyz,im,cam)-xy[ix],axis=1) if mm else np.empty(0);depth=(xyz@qvec2rotmat(im.qvec).T+im.tvec)[:,2];r.update(query=q,mode=mode,keypoints=len(xy),matches=len(mm),correct_matches=int(((err<=4)&(depth>0)).sum()),inlier_ratio=r['inliers']/len(mm) if mm else 0.);rr.append(r);rows.append(r);details[f'{mode}_{q}']={'query_indices':ix.tolist(),'point3d_ids':pid,'colmap_xy':xy.tolist()}
        summaries[mode]={'good_poses':sum(r['good_pose'] for r in rr),**{f'mean_{k}':float(np.mean([r[k] for r in rr])) for k in ['keypoints','matches','correct_matches','inliers','inlier_ratio']}}
    put(out/'results.json',{'candidate':candidate,'checkpoint_sha256':sha(checkpoint),'source_sha256':sha(__file__),'teacher_protocol_sha256':sha(base/'protocol.json'),'map_hashes':cfg['geometry_hashes'],'summary':summaries,'rows':rows,'match_details':details,'input_hashes':inputs,'pnp_replays':60,'scope':'FP32-only internal DEV diagnostic; no INT8/board/TEST claim'});print(json.dumps(summaries),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--candidate',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();run(a.candidate,a.output)
