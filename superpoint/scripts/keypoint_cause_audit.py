"""Bounded six-hypothesis keypoint audit and non-training recovery trials on DEV."""
import _bootstrap
import argparse,json,csv,shutil
from pathlib import Path
import numpy as np
import torch.nn.functional as F
from fixed_keypoint_pnp import (read,check,camera,pose,match,context,MODEL,DEST,read_points3D_binary,qvec2rotmat)
from spk210.runtime import torch,cv2,ops,w0,SuperPointKPUWrapper
from spk210.settings import SUPERPOINT
from spk210.map_nms_gate import ASSETS
from spk210.io import sha,put,guard
from hloc.extractors.superpoint import sample_descriptors_fix_sampling

FIX=SUPERPOINT/'artifacts/fixed_keypoint_pnp/run_001'
RAW=SUPERPOINT/'artifacts/descriptor_kd_ab/desc_control_r2_cw/natural'
VARIANTS=['baseline','greedy_nms','threshold_001','content_border','spatial_grid',
          'subpixel_geometry','subpixel_resample','sampling_center','mapping_legacy','gftt']


def scoremap(a):
    s=a.softmax(1)[:,:64];n,_,h,w=s.shape
    return s.permute(0,2,3,1).reshape(n,h,w,8,8).permute(0,1,3,2,4).reshape(n,h*8,w*8)


def select(score,mask,variant,gray):
    threshold=.001 if variant=='threshold_001' else .005
    if variant=='gftt':
        corners=cv2.goodFeaturesToTrack(gray,160,.01,4,mask=mask.astype(np.uint8),blockSize=3)
        return np.empty((0,2),np.float32) if corners is None else corners[:,0]
    if variant=='greedy_nms':
        s=score[0].numpy();yy,xx=np.nonzero((s>threshold)&(mask>0));order=np.lexsort((xx,yy,-s[yy,xx]));blocked=np.zeros_like(mask,bool);result=[]
        for k in order:
            y,x=yy[k],xx[k]
            if blocked[y,x]:continue
            result.append((x,y));blocked[max(0,y-4):y+5,max(0,x-4):x+5]=True
            if len(result)==160:break
        return np.asarray(result,np.float32).reshape(-1,2)
    s=ops.simple_nms(score*torch.from_numpy(mask[None]),4)[0]
    yx=torch.nonzero(s>threshold);conf=s[tuple(yx.t())]
    if variant=='spatial_grid':
        xy=yx.flip([1]).numpy();ss=conf.numpy();order=np.lexsort((xy[:,0],xy[:,1],-ss));counts=np.zeros((4,4),int);result=[]
        for k in order:
            gx=min(3,int(xy[k,0]*4/184));gy=min(3,int(xy[k,1]*4/320))
            if counts[gy,gx]>=10:continue
            counts[gy,gx]+=1;result.append(xy[k])
        return np.asarray(result,np.float32).reshape(-1,2)
    yx,_=ops.top_k_keypoints(yx,conf,160)
    return yx.flip([1]).float().numpy()


def geometry_stats(xy,im,cam,points,active):
    visible=set(int(v) for v in im.point3D_ids if v>=0);ids=np.array(sorted(visible&set(active)),np.int64)
    xyz=np.array([points[int(v)].xyz for v in ids]).reshape(-1,3);K,dist=camera(cam);R=qvec2rotmat(im.qvec)
    uv,_=cv2.projectPoints(xyz,cv2.Rodrigues(R)[0],im.tvec,K,dist);uv=uv[:,0]
    depth=(xyz@R.T+im.tvec)[:,2];good=depth>0;xyz,uv,ids=xyz[good],uv[good],ids[good]
    distance=np.linalg.norm(xy[:,None]-uv[None],axis=2) if len(xy) else np.empty((0,len(uv)))
    nearest=distance.min(1) if len(xy) and len(uv) else np.full(len(xy),np.inf)
    # Geometry oracle: greedy unique associations, no descriptor; not deployable or a strict optimum.
    candidates=np.argwhere(distance<=4);ordered=sorted(candidates.tolist(),key=lambda t:(distance[t[0],t[1]],t[0],t[1]))
    iq=set();ir=set();chosen=[]
    for j,k in ordered:
        if j in iq or k in ir:continue
        iq.add(j);ir.add(k);chosen.append((j,k))
        if len(chosen)==120:break
    oracle=pose(np.array([xyz[k] for j,k in chosen]).reshape(-1,3),np.array([xy[j] for j,k in chosen]).reshape(-1,2),cam,im)
    return {'visible_active_tracks':len(ids),'geometric_usable_keypoints_4px':int((nearest<=4).sum()),
            'geometric_usable_keypoints_8px':int((nearest<=8).sum()),'geometric_usable_keypoints_12px':int((nearest<=12).sum()),
            'unique_oracle_matches':len(chosen),'oracle_good_pose':oracle['good_pose'],
            'nearest_median_px':float(np.median(nearest)) if len(nearest) else None},nearest,uv


@torch.inference_mode()
def run(out, followup=False):
    variants_to_run=['baseline','gftt_common','gftt_common_subpixel'] if followup else VARIANTS
    out.mkdir(parents=True,exist_ok=False)
    p=read(FIX/'protocol.json');common=read(FIX/'common_inputs.json');qids=p['query_ids'];refs={int(k):v for k,v in p['references'].items()}
    rids=sorted(set(r for rr in refs.values() for r in rr));images,cameras=context();points=read_points3D_binary(MODEL/'points3D.bin')
    for f,d in p['geometry_hashes'].items():check(MODEL/f,d)
    evidence=read(FIX/'evidence_hashes.json')
    for f in ['protocol.json','common_inputs.json']:check(FIX/f,evidence[f])
    sim=read(RAW/'simulation.json');rawresult=read(RAW/'results.json');rawspec=read(RAW/'protocol.json')
    check(RAW/'protocol.json',rawresult['protocol_sha256'])
    check(SUPERPOINT/'artifacts/descriptor_kd_ab/desc_control_r2_cw/best.pt',rawspec['checkpoint_sha256'])
    check(SUPERPOINT/'artifacts/deployment/desc_control_r2_cw/model.kmodel',sim['kmodel_sha256'])
    spec={'scope':'six-hypothesis bounded exploratory DEV audit; no independent TEST or automatic promotion',
          'candidate':'current recipe A, exact original wider weights','candidate_protocol':rawspec,
          'base_pnp_protocol_sha256':sha(FIX/'protocol.json'),'common_inputs_sha256':sha(FIX/'common_inputs.json'),
          'variants':variants_to_run,'source_sha256':sha(__file__),
          'interventions':{'baseline':'unchanged original NMS4,score.005,cap160,eroded mask,legacy upstream sampling,corrected COLMAP coordinates',
            'greedy_nms':'strict greedy square radius4; same score/cap/mask',
            'threshold_001':'only score threshold .001 instead of .005; same cap160',
            'content_border':'only mask uses full valid-content region, without5-pixel erosion; padding excluded',
            'spatial_grid':'up to10 original-NMS points per4x4 grid cell; max160; no post-hoc parameter search',
            'subpixel_geometry':'OpenCV cornerSubPix on original source image,win5,30iter,eps.01; reject shifts>6original pixels; refine PnP position only',
            'subpixel_resample':'same subpixel coordinates but also resample query descriptors there',
            'sampling_center':'HLoc fix_sampling function, applied to BOTH query/reference descriptors',
            'mapping_legacy':'separate negative/control intervention removing COLMAP -.5/+ .5 conversion on BOTH sides',
            'gftt':'Shi-Tomasi on network gray,160,.01,minDistance4,block3,original mask; candidate descriptors',
            'gftt_common':'Shi-Tomasi at existing intermediate288x512,160,.01,minDistance6,block3; same original network-valid mask projected back; same low-res descriptors',
            'gftt_common_subpixel':'same intermediate corners plus cornerSubPix win3,30iter,eps.01, reject shift>3 intermediate pixels or outside unchanged mask; resample descriptor at refined location'},
          'fixed':'same20 queries, reference list/active IDs, per-image descriptor matcher and PnP; every failure retained; all variants reported',
          'limitations':'old-track resampling, nearby DEV, map-derived ground truth; score/coverage statistics alone are not causality; original-image subpixel is not board-qualified',
          'no_training':True,'production_map_changed':False}
    put(out/'protocol.json',spec);shutil.copyfile(__file__,out/'evaluated_source.py')
    teacher,_=w0.import_original_superpoint();check(w0.CHECKPOINT,p['teacher_sha256']);teacher=SuperPointKPUWrapper(teacher).eval()
    inputs={};heads={};affine_checks=[];hashes={}
    for i in sorted(qids+rids):
        src=DEST/'inputs'/f'{i}.npz';check(src,sim['records'][str(i)]['input_sha256']);hashes[str(src)]=sha(src)
        with np.load(src) as z:x=z['image'].copy();mask=z['mask'][0].copy();aff=z['affine'].copy()
        cam=cameras[images[i].camera_id];w,h=cam.width,cam.height
        factor=min(288/w,512/h);nw,nh=round(w*factor),round(h*factor);x0,y0=(288-nw)//2,(512-nh)//2
        expected=np.array([nw/w*180/288,nh/h*320/512,(.5*nw/w+x0)*180/288-.5+2,(.5*nh/h+y0)*320/512-.5])
        assert np.allclose(expected,aff,atol=1e-12,rtol=0)
        # Independently compose two pixel-center resizes and the pads.
        test=np.array([[0.,0.],[w-1.,h-1.],[w*.37,h*.61]])
        two=((test+.5)*[nw/w,nh/h]+[x0,y0])*[180/288,320/512]-.5+[2,0]
        assert np.allclose(test*aff[:2]+aff[2:],two,atol=1e-10,rtol=0)
        rawmask=np.zeros((512,288),np.uint8);rawmask[y0:y0+nh,x0:x0+nw]=1
        content=np.zeros((320,184),np.uint8);content[:,2:182]=cv2.resize(rawmask,(180,320),interpolation=cv2.INTER_NEAREST)
        inputs[i]=(x,mask,aff,content);affine_checks.append({'image':i,'max_error':float(np.max(np.abs(two-(test*aff[:2]+aff[2:]))))})
        for name in ['fp32','int8']:
            f=RAW/name/f'{i}.npz';digest=rawresult['results'][name]['raw_hashes'][str(i)];check(f,digest);hashes[str(f)]=sha(f)
            with np.load(f) as z:heads[i,name]=(torch.from_numpy(z['logits'].copy()),torch.from_numpy(z['desc'].copy()))
        if i in qids:heads[i,'teacher']=teacher(torch.from_numpy(x))
    put(out/'coordinate_checks.json',{'status':'PASS_ALGEBRA','rows':affine_checks,'note':'tests stored affine against independently composed pixel-center resizes; not proof of external map ground truth or all distortions'})
    def sample(b,xy,center=False):
        fn=sample_descriptors_fix_sampling if center else ops.sample_descriptors
        return fn(torch.from_numpy(np.asarray(xy,np.float32))[None],F.normalize(b,dim=1),8)[0].T.numpy()
    banks={}
    for backend in ['fp32','int8']:
        for sampler in ['baseline','sampling_center','mapping_legacy']:
            bank={}
            for i in rids:
                f=FIX/'teacher_fp32'/f'{i}.npz';check(f,evidence[str(f.relative_to(FIX))])
                with np.load(f) as z:ids=z['ids'].copy();xy=z['network_xy'].copy()
                if sampler=='mapping_legacy':xy+=inputs[i][2][:2]*.5
                bank[i]=(ids,sample(heads[i,backend][1],xy,sampler=='sampling_center'))
            banks[backend,sampler]=bank
    teacher_bank={}
    for i in rids:
        f=FIX/'teacher_fp32'/f'{i}.npz'
        with np.load(f) as z:teacher_bank[i]=(z['ids'].copy(),z['desc'].copy())
    rows=[];details={};baseline_scores={};baseline_distance={}
    for backend in ['teacher','fp32','int8']:
        variants=['baseline'] if backend=='teacher' else variants_to_run
        for variant in variants:
            vr=[]
            for q in qids:
                im=images[q];cam=cameras[im.camera_id];x,mask,aff,content=inputs[q];a,b=heads[q,backend]
                score=scoremap(a);chosenmask=content if variant=='content_border' else mask
                if variant.startswith('gftt_common'):
                    gray=cv2.imread(str(ASSETS/im.name),0);assert gray.shape==(cam.height,cam.width)
                    factor=min(288/cam.width,512/cam.height);nw,nh=round(cam.width*factor),round(cam.height*factor)
                    x0,y0=(288-nw)//2,(512-nh)//2;intermediate=np.zeros((512,288),np.uint8)
                    intermediate[y0:y0+nh,x0:x0+nw]=cv2.resize(gray,(nw,nh),interpolation=cv2.INTER_AREA)
                    # Exact second-resize coordinate map; do not resize the already low-resolution image upward.
                    yy,xx=np.mgrid[:512,:288];tx=np.clip(np.rint((xx+.5)*.625-.5+2).astype(int),0,183);ty=np.clip(np.rint((yy+.5)*.625-.5).astype(int),0,319)
                    cmask=mask[ty,tx].astype(np.uint8)
                    corners=cv2.goodFeaturesToTrack(intermediate,160,.01,6,mask=cmask,blockSize=3)
                    cxy=np.empty((0,2),np.float32) if corners is None else corners[:,0]
                    if variant=='gftt_common_subpixel' and len(cxy):
                        refined=cxy.copy().reshape(-1,1,2);cv2.cornerSubPix(intermediate,refined,(3,3),(-1,-1),(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,30,.01))
                        refined=refined[:,0];valid=np.isfinite(refined).all(1)&(np.linalg.norm(refined-cxy,axis=1)<=3)
                        ij=np.rint((np.nan_to_num(refined)+.5)*.625-.5+[2,0]).astype(int)
                        inside=(ij[:,0]>=0)&(ij[:,0]<184)&(ij[:,1]>=0)&(ij[:,1]<320)
                        maskvalid=np.zeros(len(ij),bool);maskvalid[inside]=mask[ij[inside,1],ij[inside,0]]>0
                        valid &= maskvalid
                        cxy[valid]=refined[valid]
                    logical=(cxy+.5)*.625-.5+[2,0]
                else:logical=select(score,chosenmask,variant,x[0,0])
                descxy=logical.copy();xy=(logical-aff[2:])/aff[:2]+.5
                moved=0
                if variant.startswith('subpixel') and len(xy):
                    gray=cv2.imread(str(ASSETS/im.name),0);assert gray.shape==(cam.height,cam.width)
                    initial=(xy-.5).astype(np.float32);refined=initial.copy().reshape(-1,1,2)
                    cv2.cornerSubPix(gray,refined,(5,5),(-1,-1),(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,30,.01));refined=refined[:,0]
                    shift=np.linalg.norm(refined-initial,axis=1);keep=np.isfinite(refined).all(1)&(shift<=6)
                    result=initial.copy();result[keep]=refined[keep];xy=result+.5;moved=int((keep&(shift>.01)).sum())
                    if variant=='subpixel_resample':descxy=(xy-.5)*aff[:2]+aff[2:]
                if variant=='mapping_legacy':xy-=.5
                d=sample(b,descxy,variant=='sampling_center')
                sampler=variant if variant in ['sampling_center','mapping_legacy'] else 'baseline'
                bank=teacher_bank if backend=='teacher' else banks[backend,sampler]
                active=common['active_ids'][str(q)];matched=match(d,bank,refs[q],active)
                xyz=np.array([points[v].xyz for j,v,s in matched]).reshape(-1,3);pix=xy[np.array([j for j,v,s in matched],int)]
                row=pose(xyz,pix,cam,im)
                correct=0
                if len(xyz):
                    K,dist=camera(cam);R=qvec2rotmat(im.qvec);uv,_=cv2.projectPoints(xyz,cv2.Rodrigues(R)[0],im.tvec,K,dist)
                    correct=int(((np.linalg.norm(uv[:,0]-pix,axis=1)<=4)&((xyz@R.T+im.tvec)[:,2]>0)).sum())
                geometry,distances,projected=geometry_stats(xy,im,cam,points,active)
                row.update(query=q,backend=backend,variant=variant,keypoints=len(xy),correct_matches=correct,refined_points=moved,**geometry)
                yy=np.clip(np.rint(logical[:,1]).astype(int),0,319);xx=np.clip(np.rint(logical[:,0]).astype(int),0,183)
                scores=score[0].numpy()[yy,xx]
                if variant=='baseline':
                    s=score[0].numpy();nms=ops.simple_nms(score*torch.from_numpy(mask[None]),4)[0].numpy()
                    allxy=np.argwhere(nms>.005)[:,::-1].copy();allsource=(allxy-aff[2:])/aff[:2]+.5
                    useful_all=0
                    for chunk in np.array_split(allsource,max(1,int(np.ceil(len(allsource)/256)))):
                        if len(chunk):useful_all+=int((np.linalg.norm(chunk[:,None]-projected[None],axis=2).min(1)<=4).sum())
                    logical_gt=(projected-.5)*aff[:2]+aff[2:];ij=np.rint(logical_gt).astype(int)
                    inside=(ij[:,0]>=0)&(ij[:,0]<184)&(ij[:,1]>=0)&(ij[:,1]<320)
                    maskok=np.zeros(len(ij),bool);maskok[inside]=mask[ij[inside,1],ij[inside,0]]>0
                    delta=np.max(np.abs(logical[:,None]-logical[None]),axis=2);np.fill_diagonal(delta,np.inf)
                    bins=np.minimum((logical/[184,320]*4).astype(int),3)
                    row.update(pre_nms_above_threshold=int(((s>.005)&(mask>0)).sum()),post_nms_candidates=len(allxy),
                        usable_before_topk=useful_all,projected_tracks_mask_excluded=int((~maskok).sum()),
                        clustered_fraction=float((delta.min(1)<=4).mean()) if len(logical)>1 else 0.,
                        spatial_cells=len(np.unique(bins,axis=0)),score_median=float(np.median(scores)) if len(scores) else None,
                        score_p95=float(np.percentile(scores,95)) if len(scores) else None)
                    baseline_scores.setdefault(backend,[]).extend(scores.tolist());baseline_distance.setdefault(backend,[]).extend(distances.tolist())
                vr.append(row);rows.append(row)
                details[f'{backend}_{variant}_{q}']={'network_xy':logical.tolist(),'pnp_colmap_xy':xy.tolist(),'matched_query_indices':[j for j,v,s in matched],'matched_point3d_ids':[v for j,v,s in matched]}
            print(backend,variant,'poses',sum(r['good_pose'] for r in vr),'oracle',sum(r['oracle_good_pose'] for r in vr),
                  'useful',np.mean([r['geometric_usable_keypoints_4px'] for r in vr]),flush=True)
    summaries={}
    for backend in ['teacher','fp32','int8']:
        for variant in (['baseline'] if backend=='teacher' else variants_to_run):
            rr=[r for r in rows if r['backend']==backend and r['variant']==variant]
            summary={'good_poses':sum(r['good_pose'] for r in rr),'queries':len(rr),'oracle_good_poses':sum(r['oracle_good_pose'] for r in rr)}
            for k in ['keypoints','correct_matches','inliers','inlier_ratio','geometric_usable_keypoints_4px','geometric_usable_keypoints_8px','geometric_usable_keypoints_12px','unique_oracle_matches','nearest_median_px']:
                summary['mean_'+k]=float(np.mean([r[k] for r in rr]))
            if variant=='baseline':
                for k in ['post_nms_candidates','usable_before_topk','projected_tracks_mask_excluded','clustered_fraction','spatial_cells','score_median','score_p95']:
                    summary['mean_'+k]=float(np.mean([r[k] for r in rr]))
            summaries[backend+'_'+variant]=summary
    assert summaries['teacher_baseline']['good_poses']==16
    assert summaries['fp32_baseline']['good_poses']==summaries['int8_baseline']['good_poses']==0
    put(out/'results.json',{'status':'COMPLETE','summary':summaries,'rows':rows,'input_hashes':hashes})
    put(out/'keypoints_and_matches.json',details)
    fields=sorted(set(k for r in rows for k in r))
    with (out/'per_query.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for backend in ['teacher','fp32','int8']:
        d=np.sort(baseline_distance[backend]);axes[0].plot(d,np.arange(1,len(d)+1)/len(d),label=backend)
        axes[1].hist(baseline_scores[backend],bins=40,histtype='step',density=True,label=backend)
    axes[0].set(xlim=(0,40),xlabel='Nearest visible active Point3D projection (original pixels)',ylabel='Fraction of selected keypoints')
    axes[0].axvline(4,color='black',linestyle='--');axes[1].set(xlabel='Selected detector score',ylabel='Density')
    for ax in axes:ax.legend();ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(out/'diagnostic_curves.png',dpi=160);plt.close(fig)
    check(Path(__file__),spec['source_sha256'])


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True,type=Path);p.add_argument('--followup',action='store_true');args=p.parse_args();run(args.output,args.followup)
