"""Post-training floor6 descriptor ranking and own/fixed-location donor evidence."""
import _bootstrap
import os,csv
from pathlib import Path
import numpy as np
import torch.nn.functional as F
from student_point_supervision_ab import DEST,BRANCHES
from floor6_student_evaluate import context,read,check
from spk210.runtime import torch,ops
from spk210.settings import SUPERPOINT
from spk210.io import sha,put
from spk210.map_nms_gate import context as map_context,MODEL,read_points3D_binary,qvec2rotmat
from spk210.map_geometry import pose,project
from fixed_keypoint_pnp import match

@torch.inference_mode()
def main():
    original=SUPERPOINT/'artifacts/floor6_student_evaluation/run_001';oe=read(original/'evidence_hashes.json');records=[];calibrations=[];initial=[];allhashes={}
    images,cameras=map_context();points=read_points3D_binary(MODEL/'points3D.bin');spec=read(DEST/'protocol.json')
    for name in BRANCHES:
        folder=DEST/name;raw=folder/'floor6';os.environ['SP_STUDENT_CANDIDATE']=name
        base,p,be,common,queries,refs,ids,candidate,dep,compile=context(raw)
        check(raw/'protocol.json',read(raw/'evidence_hashes.json')['protocol.json'])
        assert compile['CPU_Conv2D_count']==0 and compile['KPU_Conv2D_count']==12
        calibrations.append(compile['calibration_sha256']);check(dep/'calibration_train.npy',calibrations[-1])
        progress=read(folder/'progress.json');assert progress['step']==400 and len(progress['losses'])==400
        assert all(np.isfinite([r['detector'],r['geometry'],r['student_anchor_geometry']]).all() for r in progress['losses'])
        initial.append(progress['evaluations'][0]['metrics'])
        report=read(folder/'report.json');assert report['initial_sha256']==spec['initial_sha256'] and report['plan_sha256']==spec['plan_sha256'];check(folder/'best.pt',report['checkpoint_sha256'])
        quality=read(dep/'int8_quality.json');assert quality['status']=='MEASURED'
        sim=read(raw/'simulation.json');heads={};coords={};banks={b:{} for b in ['teacher','fp32','int8']};locations={};qsource={};trackdesc={b:{} for b in ['teacher','fp32','int8']}
        rh={b:read(raw/f'{b}_raw_hashes.json') for b in ['fp32','int8']}
        for i in ids:
            f=base/'inputs'/f'{i}.npz';check(f,be[str(f.relative_to(base))]);allhashes[str(f)]=sha(f)
            with np.load(f) as z:aff=z['affine'].copy();mask=z['mask'][0].copy()
            if i in queries:
                f=raw/'fp32'/f'query_features_{i}.npz'
                with np.load(f) as z:ownxy=z['own_colmap_xy'].copy()
                f=original/'fp32'/f'query_features_{i}.npz';check(f,oe[str(f.relative_to(original))])
                with np.load(f) as z:frozenxy=z['own_colmap_xy'].copy()
                locations[i]={'own':(ownxy-.5)*aff[:2]+aff[2:],'frozen':(frozenxy-.5)*aff[:2]+aff[2:]};qsource[i]={'own':ownxy,'frozen':frozenxy}
            else:
                f=base/'features'/f'{i}.npz';check(f,be[str(f.relative_to(base))])
                with np.load(f) as z:pid=z['ids'].copy();network=z['network_xy'].copy()
            im=images[i];valid=im.point3D_ids>=0;ti=im.point3D_ids[valid];uv=im.xys[valid];u,n=np.unique(ti,return_counts=True);keep=~np.isin(ti,u[n>1]);ti,uv=ti[keep],uv[keep];tx=(uv-.5)*aff[:2]+aff[2:];ij=np.rint(tx).astype(int)
            inside=(ij[:,0]>=0)&(ij[:,0]<184)&(ij[:,1]>=0)&(ij[:,1]<320);keep=np.zeros(len(ti),bool);keep[inside]=mask[ij[inside,1],ij[inside,0]]>0;ti,tx=ti[keep],tx[keep];order=np.argsort(ti,kind='stable');ti,tx=ti[order],tx[order]
            for b in ['teacher','fp32','int8']:
                f=(base/'teacher' if b=='teacher' else raw/b)/f'{i}.npz';check(f,be[str(f.relative_to(base))] if b=='teacher' else rh[b][str(i)])
                if b=='int8':check(f,sim['records'][str(i)]['output_sha256'])
                allhashes[str(f)]=sha(f)
                with np.load(f) as z:h=torch.from_numpy(z['desc'].copy())
                def sample(xy):return ops.sample_descriptors(torch.from_numpy(xy.astype(np.float32))[None],F.normalize(h,dim=1),8)[0].T.numpy()
                trackdesc[b][i]=(ti,sample(tx))
                if i in queries:
                    for mode in ['own','frozen']:coords[i,b,mode]=sample(locations[i][mode])
                else:banks[b][i]=(pid,sample(network))
        ranking={}
        for b in ['teacher','fp32','int8']:
            rr=[]
            for q in queries:
                ref=refs[q][0];qi,qd=trackdesc[b][q];ri,rd=trackdesc[b][ref];ri,rd=ri[:512],rd[:512];shared=np.intersect1d(qi,ri)[:256]
                if not len(shared):rr.append({'query':q,'reference':ref,'tracks':0,'top1':0.,'top5':0.});continue
                scores=qd[np.searchsorted(qi,shared)]@rd.T;target=np.searchsorted(ri,shared);ranks=np.argmax(np.argsort(-scores,axis=1,kind='stable')==target[:,None],axis=1)+1
                rr.append({'query':q,'reference':ref,'tracks':len(shared),'top1':float((ranks==1).mean()),'top5':float((ranks<=5).mean())})
            ranking[b]={'top1':float(np.mean([r['top1'] for r in rr])),'top5':float(np.mean([r['top5'] for r in rr])),'rows':rr}
        rows=[];details={};summaries={}
        for mode in ['own','frozen']:
            for b in (['teacher','fp32','int8'] if mode=='own' else ['fp32','int8']):
                rr=[]
                for q in queries:
                    d=coords[q,b,mode];xy=qsource[q][mode];mm=match(d,banks[b],refs[q],common['active_ids'][str(q)]);ix=np.array([j for j,v,s in mm],int);pid=[v for j,v,s in mm];xyz=np.array([points[v].xyz for v in pid]).reshape(-1,3);im=images[q];cam=cameras[im.camera_id]
                    row=pose(xyz,xy[ix],cam,im);assert row==pose(xyz,xy[ix],cam,im)
                    err=np.linalg.norm(project(xyz,im,cam)-xy[ix],axis=1) if len(mm) else np.empty(0);depth=(xyz@qvec2rotmat(im.qvec).T+im.tvec)[:,2]
                    row.update(query=q,mode=mode,donor=b,keypoints=len(xy),matches=len(mm),correct_matches=int(((err<=4)&(depth>0)).sum()),inlier_ratio=row['inliers']/len(mm) if mm else 0.)
                    rr.append(row);rows.append(row);details[f'{mode}_{b}_{q}']={'query_indices':ix.tolist(),'point3d_ids':pid}
                v={'good_poses':sum(r['good_pose'] for r in rr)}
                for k in ['correct_matches','inliers','inlier_ratio']:v['mean_'+k]=float(np.mean([r[k] for r in rr]))
                summaries[mode+'/'+b]=v
        put(folder/'descriptor_diagnostics.json',{'ranking':ranking,'summary':summaries,'rows':rows,'match_details':details,'scope':'same floor6 evaluator; own uses candidate FP32 locations for every donor; frozen uses original default FP32 locations shared across A/B'})
        core=read(raw/'results.json')
        if not BRANCHES[name]:
            assert core['summary']==read(original/'results.json')['summary'] and core['rows']==read(original/'results.json')['rows'], 'Control floor6 replay mismatch'
            assert summaries['own/teacher']['good_poses']==11 and summaries['own/fp32']['good_poses']==5 and summaries['own/int8']['good_poses']==5
        records.append({'arm':'A' if not BRANCHES[name] else 'B','candidate':name,'step':report['best_step'],'descriptor_top1':{b:ranking[b]['top1'] for b in ranking},'donors':summaries,'own_full_pipeline':{b:core['summary'][b+'/own'] for b in ['fp32','int8']},'synthetic':{b:quality[b]['macro'] for b in ['fp32','int8']}})
        print(name,records[-1],flush=True)
    assert initial[0]==initial[1] and len(set(calibrations))==1
    check(SUPERPOINT/'artifacts/deployment/wider_r2_cw/calibration_train.npy',calibrations[0])
    A,B=records
    keep=(B['descriptor_top1']['fp32']>A['descriptor_top1']['fp32'] and B['own_full_pipeline']['fp32']['good_poses']>A['own_full_pipeline']['fp32']['good_poses'] and B['own_full_pipeline']['int8']['good_poses']>=A['own_full_pipeline']['int8']['good_poses'] and all(B['donors']['frozen/'+b]['good_poses']>=A['donors']['frozen/'+b]['good_poses'] for b in ['fp32','int8']))
    put(DEST/'summary.json',{'records':records,'keep_B_provisionally':keep,'criterion':'predeclared protocol.json','source_sha256':sha(__file__)})
    put(DEST/'verification.json',{'status':'PASS','same_initial_metrics':True,'same_calibration_as_default':True,'extra_pnp_replays':200,'source_hashes':allhashes})

if __name__=='__main__':main()
