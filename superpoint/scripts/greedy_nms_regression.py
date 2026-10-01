"""Frozen42-pair FP32/INT8 NMS regression with floor6 downstream checks."""
import _bootstrap
import argparse,shutil
from pathlib import Path
import numpy as np
import torch.nn.functional as F
from spk210.runtime import torch,ops
from spk210.settings import SUPERPOINT,ROOT
from spk210.io import sha,put
from spk210.metrics import features,evaluate
from spk210.protocol import protocol
from spk210.onnx_export import load_model
from fixed_keypoint_pnp import read,check,match
from spk210.map_nms_gate import context,MODEL,read_points3D_binary,qvec2rotmat
from spk210.map_geometry import pose,project

CANDIDATE='point_geometry_r2_cw'

@torch.inference_mode()
def run(out, candidate=CANDIDATE, raw=None):
    out.mkdir(parents=True,exist_ok=False);dep=SUPERPOINT/'artifacts/deployment'/candidate;raw=raw or SUPERPOINT/'artifacts/student_point_supervision_ab'/candidate/'floor6';base=SUPERPOINT/'artifacts/floor6_teacher_baseline/run_001'
    compile=read(dep/'compile.json');model,res,checkpoint=load_model(candidate);check(checkpoint,compile['checkpoint_sha256']);check(dep/'model.kmodel',compile['kmodel_sha256'])
    sim=read(dep/'simulation.json');quality=read(dep/'int8_quality.json')
    spec={'scope':'NMS-only regression, same frozen model,42 synthetic DEV pairs and20 canonical floor6 queries; no training','candidate':candidate,'checkpoint_sha256':sha(checkpoint),'kmodel_sha256':compile['kmodel_sha256'],'arms':['original','greedy'],'unchanged':'threshold.005,top160,square NMS radius4,descriptor sampler/data/evaluator/matcher/native fisheye PnP','nearby_rate':'fraction of keypoints with another within Chebyshev4network pixels, self excluded','gate_before_results':{'FP32':'macro correctMNN >=95% of original, precision drop<=2percentage points; repeatability/coverage >=95%; each domain correctMNN >=90%; floor6 own pose count >=original-1','INT8':'macro correctMNN strictly improves, precision drop<=2percentage points, no repeatability/coverage regression; nearby rate falls and <=1%; floor6 own pose count >=original','note':'engineering DEV gate, not statistical significance; no posthoc relaxation'},'source_sha256':sha(__file__)}
    put(out/'protocol.json',spec);shutil.copyfile(__file__,out/'evaluated_source.py')
    outputs={};stats={};cached={'fp32':[],'int8':[]}
    for i,row in enumerate(sim['rows']):
        assert row['index']==i;src=ROOT/'r2/dev'/f'{i:03d}.npz';check(src,row['input_sha256'])
        with np.load(src) as z:cached['fp32'].append(tuple(v.detach() for v in model(torch.from_numpy(z['pair'].copy()))))
        f=dep/'sim_outputs'/f'{i:03d}.npz';check(f,row['output_sha256'])
        with np.load(f) as z:cached['int8'].append(tuple(torch.from_numpy(z[k].copy()) for k in ['logits','desc']))
    class Cached(torch.nn.Module):
        def __init__(self,b):super().__init__();self.b=b;self.i=0
        def forward(self,x):v=cached[self.b][self.i];self.i+=1;return v
    for backend in ['fp32','int8']:
        for mode in ['original','greedy']:
            details=[]
            def extractor(a,b,masks,res,pair):
                fs=features(a,b,masks,res,nms=mode)
                for xy,d in fs:
                    xy=xy*1.25+[2,0];distance=np.max(np.abs(xy[:,None]-xy[None]),axis=2);np.fill_diagonal(distance,np.inf)
                    duplicate=float((distance.min(1)<1e-6).mean()) if len(xy)>1 else 0.;near=float((distance.min(1)<=4+1e-10).mean()) if len(xy)>1 else 0.
                    details.append({'keypoints':len(xy),'duplicate_rate':duplicate,'nearby_rate':near})
                return fs
            r=evaluate(Cached(backend),'r2',protocol()['rows']['dev'],feature_extractor=extractor)
            if mode=='original':assert r==quality[backend], 'Original evaluation no longer reproduces recorded metrics'
            outputs[backend+'/'+mode]=r;stats[backend+'/'+mode]={'image_mean_keypoints':float(np.mean([d['keypoints'] for d in details])),'duplicate_rate':float(np.mean([d['duplicate_rate'] for d in details])),'nearby_rate':float(np.mean([d['nearby_rate'] for d in details])),'images':details}
            print(backend,mode,r['macro'],stats[backend+'/'+mode]['nearby_rate'],flush=True)
    be=read(base/'evidence_hashes.json');re=read(raw/'evidence_hashes.json');check(base/'common_inputs.json',be['common_inputs.json']);p=read(base/'protocol.json');common=read(base/'common_inputs.json');queries=p['query_ids'];refs={int(q):v for q,v in common['references'].items()};ids=sorted(set(queries+[i for rr in refs.values() for i in rr]));images,cameras=context();points=read_points3D_binary(MODEL/'points3D.bin')
    downstream={};allposes=[];old=read(raw/'results.json')
    for backend in ['fp32','int8']:
        bank={};head={};inputs={}
        for i in ids:
            f=raw/backend/f'{i}.npz';check(f,re[str(f.relative_to(raw))])
            with np.load(f) as z:a=torch.from_numpy(z['logits'].copy());b=torch.from_numpy(z['desc'].copy())
            if i in queries:
                f=base/'inputs'/f'{i}.npz';check(f,be[str(f.relative_to(base))])
                with np.load(f) as z:inputs[i]=(z['mask'].copy(),z['affine'].copy())
                head[i]=(a,b)
            else:
                f=base/'features'/f'{i}.npz';check(f,be[str(f.relative_to(base))])
                with np.load(f) as z:pid=z['ids'].copy();xy=z['network_xy'].copy()
                bank[i]=(pid,ops.sample_descriptors(torch.from_numpy(xy.astype(np.float32))[None],F.normalize(b,dim=1),8)[0].T.numpy())
        for mode in ['original','greedy']:
            rr=[]
            for q in queries:
                mask,aff=inputs[q];canonical,d=features(*head[q],mask,'r2',nms=mode)[0];xy=(canonical*1.25+[2,0]-aff[2:])/aff[:2]+.5;mm=match(d,bank,refs[q],common['active_ids'][str(q)]);ix=np.array([j for j,v,s in mm],int);xyz=np.array([points[v].xyz for j,v,s in mm]).reshape(-1,3);im=images[q];cam=cameras[im.camera_id];r=pose(xyz,xy[ix],cam,im);assert r==pose(xyz,xy[ix],cam,im)
                if mode=='original':
                    prev=next(v for v in old['rows'] if v['query']==q and v['backend']==backend and v['mode']=='own')
                    for k,v in r.items():assert prev[k]==v
                err=np.linalg.norm(project(xyz,im,cam)-xy[ix],axis=1) if len(mm) else np.empty(0);depth=(xyz@qvec2rotmat(im.qvec).T+im.tvec)[:,2]
                r.update(query=q,backend=backend,nms=mode,keypoints=len(xy),matches=len(mm),correct_matches=int(((err<=4)&(depth>0)).sum()),inlier_ratio=r['inliers']/len(mm) if mm else 0.);rr.append(r);allposes.append(r)
            downstream[backend+'/'+mode]={'good_poses':sum(r['good_pose'] for r in rr),'rows':rr};print('FLOOR6',backend,mode,downstream[backend+'/'+mode]['good_poses'],flush=True)
    fa,fb=[outputs['fp32/'+m]['macro'] for m in ['original','greedy']];ia,ib=[outputs['int8/'+m]['macro'] for m in ['original','greedy']]
    checks={'fp32_correct':fb['correct_mnn_count']>=.95*fa['correct_mnn_count'],'fp32_precision':fb['mnn_precision']>=fa['mnn_precision']-.02,'fp32_repeatability_coverage':all(fb[k]>=.95*fa[k] for k in ['repeatability','coverage']),'fp32_domain_correct':all(outputs['fp32/greedy']['per_domain'][d]['correct_mnn_count']>=.9*outputs['fp32/original']['per_domain'][d]['correct_mnn_count'] for d in ['indoor','project']),'fp32_floor6':downstream['fp32/greedy']['good_poses']>=downstream['fp32/original']['good_poses']-1,'int8_correct':ib['correct_mnn_count']>ia['correct_mnn_count'],'int8_precision':ib['mnn_precision']>=ia['mnn_precision']-.02,'int8_repeatability_coverage':all(ib[k]>=ia[k] for k in ['repeatability','coverage']),'int8_nearby':stats['int8/greedy']['nearby_rate']<stats['int8/original']['nearby_rate'] and stats['int8/greedy']['nearby_rate']<=.01,'int8_floor6':downstream['int8/greedy']['good_poses']>=downstream['int8/original']['good_poses']}
    put(out/'results.json',{'metrics':outputs,'keypoints':stats,'floor6':downstream,'gate_checks':checks,'gate_pass':all(checks.values())});print('GATE',checks,flush=True)
    put(out/'evidence_hashes.json',{str(f.relative_to(out)):sha(f) for f in sorted(out.rglob('*')) if f.is_file()})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--candidate',default=CANDIDATE);p.add_argument('--raw',type=Path);a=p.parse_args();run(a.output,a.candidate,a.raw)
