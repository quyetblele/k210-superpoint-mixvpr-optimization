"""Spatial detector audit on frozen DEV: raw peaks, equal K, map coverage and coordinates."""
import _bootstrap
import argparse,json,csv,shutil
from pathlib import Path
import numpy as np
import torch.nn.functional as F
from keypoint_cause_audit import FIX,RAW,scoremap,select
from fixed_keypoint_pnp import read,check,camera,context,MODEL,DEST,read_points3D_binary,qvec2rotmat
from spk210.runtime import torch,cv2,ops,w0,SuperPointKPUWrapper
from spk210.settings import SUPERPOINT
from spk210.metrics import features
from spk210.io import sha,put

RADII=[1,2,3,4]

def nearest(a,b):
    if not len(b):return np.full(len(a),np.inf)
    return np.concatenate([np.linalg.norm(x[:,None]-b[None],axis=2).min(1) for x in np.array_split(a,max(1,int(np.ceil(len(a)/256))))]) if len(a) else np.empty(0)

def top(score,valid,k=160):
    yy,xx=np.nonzero(valid);v=score[yy,xx];order=np.lexsort((xx,yy,-v))[:k]
    return np.c_[xx[order],yy[order]].astype(float)

def stats(xy,uv,aff):
    source=(xy-aff[2:])/aff[:2]+.5;d=nearest(source,uv);back=nearest(uv,source)
    out={'keypoints':len(xy),'nearest_projection_median_px':float(np.median(d)) if len(d) else None}
    for r in RADII:
        out[f'keypoints_near_{r}px']=int((d<=r).sum());out[f'fraction_near_{r}px']=float((d<=r).mean()) if len(d) else 0.
        out[f'projection_recall_{r}px']=float((back<=r).mean()) if len(back) else 0.
    bins=np.clip((xy/[184,320]*4).astype(int),0,3);counts=np.bincount(bins[:,1]*4+bins[:,0],minlength=16)
    out.update(grid_counts=counts.tolist(),occupied_cells=int((counts>0).sum()),largest_cell_fraction=float(counts.max()/len(xy)) if len(xy) else 0.)
    return out,d

@torch.inference_mode()
def run(out):
    out.mkdir(parents=True,exist_ok=False)
    prior=SUPERPOINT/'artifacts/keypoint_cause_audit/run_003';pe=read(prior/'evidence_hashes.json');fe=read(FIX/'evidence_hashes.json')
    for f in ['results.json','keypoints_and_matches.json']:check(prior/f,pe[f])
    for f in ['protocol.json','common_inputs.json']:check(FIX/f,fe[f])
    p=read(FIX/'protocol.json');common=read(FIX/'common_inputs.json');loc=read(prior/'keypoints_and_matches.json')
    nr=read(RAW/'results.json');ns=read(RAW/'simulation.json');rp=read(RAW/'protocol.json');check(RAW/'protocol.json',nr['protocol_sha256'])
    checkpoint=SUPERPOINT/'artifacts/descriptor_kd_ab/desc_control_r2_cw/best.pt';check(checkpoint,rp['checkpoint_sha256'])
    original=torch.load(SUPERPOINT/'artifacts/quality_pilot/wider_r2_cw/best.pt',weights_only=False)['model'];control=torch.load(checkpoint,weights_only=False)['model']
    assert all(torch.equal(original[k],control[k]) for k in original)
    images,cameras=context();pts=read_points3D_binary(MODEL/'points3D.bin')
    for f,h in p['geometry_hashes'].items():check(MODEL/f,h)
    spec={'scope':'same20 nearby DEV queries, spatial audit only; no training or tuning', 'default':'wider_r2_cw FP32, exact control weights',
          'checkpoint_sha256':sha(checkpoint),'teacher_sha256':p['teacher_sha256'],'geometry_hashes':p['geometry_hashes'],
          'base_protocol_sha256':sha(FIX/'protocol.json'),'common_inputs_sha256':sha(FIX/'common_inputs.json'),'source_sha256':sha(__file__),
          'radii':'1/2/3/4 ORIGINAL COLMAP image pixels, not network pixels',
          'stages':['baseline','nms_equal160_no_threshold','raw_localmax_equal160_no_threshold','raw_top160_no_nms','all_nms_above005'],
          'equalization':'K160 each teacher/student, threshold disabled; deterministic score descending ties y then x. Original baseline uses unchanged upstream topK.',
          'map_sets':'positive-depth query-observed Point3Ds in current full map versus intersection with existing active IDs; no claim about unobserved visibility or preoptimization map',
          'projection_filter':'compare full/active tracks within same eroded query-valid mask; separately record mask exclusions',
          'raw_localmax':'3x3 max-pool of unmasked score, before mask; no radius4 NMS; ties retained before deterministic top160',
          'limitations':'internal reconstruction and DEV; raw spatial weakness cannot identify training cause; repeated NMS/border sweeps intentionally not run'}
    put(out/'protocol.json',spec);shutil.copyfile(__file__,out/'evaluated_source.py')
    check(w0.CHECKPOINT,p['teacher_sha256']);teacher,_=w0.import_original_superpoint();teacher=SuperPointKPUWrapper(teacher).eval()
    # All64 channel-to-pixel offsets, independent pixel_shuffle implementation.
    logits=torch.full((64,65,4,5),-30.);logits[torch.arange(64),torch.arange(64),2,3]=30.
    decoded=scoremap(logits);reference=F.pixel_shuffle(logits.softmax(1)[:,:64],8)[:,0]
    np.testing.assert_array_equal(decoded.numpy(),reference.numpy())
    for c in range(64):assert np.unravel_index(decoded[c].argmax(),decoded[c].shape)==(16+c//8,24+c%8)
    rows=[]; maps=[]; rawrows=[]; coord=[];distrows=[];hashes={};pictures={}
    for q in p['query_ids']:
        im=images[q];cam=cameras[im.camera_id];f=DEST/'inputs'/f'{q}.npz';check(f,ns['records'][str(q)]['input_sha256']);hashes[str(f)]=sha(f)
        with np.load(f) as z:x=z['image'].copy();mask=z['mask'][0].copy();aff=z['affine'].copy()
        f=RAW/'fp32'/f'{q}.npz';check(f,nr['results']['fp32']['raw_hashes'][str(q)]);hashes[str(f)]=sha(f)
        with np.load(f) as z:student=(torch.from_numpy(z['logits'].copy()),torch.from_numpy(z['desc'].copy()))
        heads={'teacher':teacher(torch.from_numpy(x)),'student':student}; ss={k:scoremap(v[0]) for k,v in heads.items()}
        factor=min(288/cam.width,512/cam.height);nw,nh=round(cam.width*factor),round(cam.height*factor);x0,y0=(288-nw)//2,(512-nh)//2
        expected=np.array([nw/cam.width*.625,nh/cam.height*.625,(.5*nw/cam.width+x0)*.625-.5+2,(.5*nh/cam.height+y0)*.625-.5])
        np.testing.assert_allclose(aff,expected,atol=1e-12,rtol=0)
        ids=np.unique(im.point3D_ids[im.point3D_ids>=0]);xyz=np.array([pts[int(i)].xyz for i in ids]);K,dist=camera(cam);R=qvec2rotmat(im.qvec)
        uv=cv2.projectPoints(xyz,cv2.Rodrigues(R)[0],im.tvec,K,dist)[0][:,0];pos=(xyz@R.T+im.tvec)[:,2]>0
        ids,uv=ids[pos],uv[pos];network=(uv-.5)*aff[:2]+aff[2:];ij=np.rint(network).astype(int)
        inside=(ij[:,0]>=0)&(ij[:,0]<184)&(ij[:,1]>=0)&(ij[:,1]<320);valid=np.zeros(len(ids),bool);valid[inside]=mask[ij[inside,1],ij[inside,0]]>0
        active=np.isin(ids,common['active_ids'][str(q)]);sets={'full_observed':uv[valid],'active_observed':uv[valid&active]}
        floor=nearest(uv[valid],(np.rint(network[valid])-aff[2:])/aff[:2]+.5) # nearest lattice representative among projected tracks
        maps.append({'query':q,'positive_observed_tracks':len(ids),'full_valid':int(valid.sum()),'active_valid':int((active&valid).sum()),'mask_excluded_full':int((~valid).sum()),
                     'active_fraction':float((active&valid).sum()/valid.sum()),'rounded_lattice_within_px':{str(r):float((floor<=r).mean()) for r in RADII}})
        baseline={k:select(v,mask,'baseline',x[0,0]) for k,v in ss.items()}
        ts=(baseline['teacher']-aff[2:])/aff[:2]+.5;st=(baseline['student']-aff[2:])/aff[:2]+.5;d=nearest(st,ts)
        distrows.append({'query':q,'median_student_to_teacher_px':float(np.median(d)),'fractions_within_px':{str(r):float((d<=r).mean()) for r in [1,2,3,4,8,12]},'distances_px':d.tolist()})
        for name,score in ss.items():
            s=score[0].numpy();nms=ops.simple_nms(score*torch.from_numpy(mask[None]),4)[0].numpy();xy=baseline[name]
            np.testing.assert_array_equal(xy,loc[f"{'teacher' if name=='teacher' else 'fp32'}_baseline_{q}"]['network_xy'])
            canonical,_=features(*heads[name],mask[None],'r2')[0]
            np.testing.assert_allclose(canonical*1.25+[2,0],xy,atol=1e-12,rtol=0)
            np.testing.assert_allclose((xy-aff[2:])/aff[:2]+.5,loc[f"{'teacher' if name=='teacher' else 'fp32'}_baseline_{q}"]['pnp_colmap_xy'],atol=1e-10,rtol=0)
            np.testing.assert_array_equal(nms[nms>0],s[nms>0]);assert not (nms[mask==0]>0).any()
            local=s==F.max_pool2d(score[:,None],3,1,1)[0,0].numpy()
            variants={'baseline':xy,'nms_equal160_no_threshold':top(nms,(nms>0)&(mask>0)),
                      'raw_localmax_equal160_no_threshold':top(s,local&(mask>0)),
                      'raw_top160_no_nms':top(s,mask>0),'all_nms_above005':np.argwhere(nms>.005)[:,::-1].astype(float)}
            assert len(variants['nms_equal160_no_threshold'])==len(variants['raw_localmax_equal160_no_threshold'])==160
            for stage,selected in variants.items():
                for mapname,projections in sets.items():
                    v,_=stats(selected,projections,aff);rows.append(dict(query=q,model=name,stage=stage,map_set=mapname,**v))
            ijxy=xy.astype(int);selected=s[ijxy[:,1],ijxy[:,0]];allvalid=s[mask>0]
            # Rank scores at nearest raster cell of known projections vs all valid image pixels.
            near=ij[valid&active];gt_scores=s[near[:,1],near[:,0]];ordered=np.sort(allvalid)
            rawrows.append({'query':q,'model':name,'score_quantiles_selected':np.quantile(selected,[0,.1,.5,.9,1]).tolist(),
                            'score_quantiles_valid':np.quantile(allvalid,[0,.1,.5,.9,1]).tolist(),
                            'median_projection_score_percentile':float(np.median(np.searchsorted(ordered,gt_scores,side='right')/len(ordered))),
                            'post_nms_above005':int((nms>.005).sum()),'raw_localmax_count':int((local&(mask>0)).sum()),
                            'projection_raster_is_localmax_fraction':float(local[near[:,1],near[:,0]].mean()),
                            'projection_raster_survives_nms_fraction':float((nms[near[:,1],near[:,0]]>0).mean())})
            coord.append({'query':q,'model':name,'decoder_pixel_shuffle_max_error':float(torch.max(torch.abs(score-F.pixel_shuffle(heads[name][0].softmax(1)[:,:64],8)[:,0]))),
                          'nms_preserves_score_and_location':True,'features_to_source_matches_prior':True,'affine_matches_two_resize_pixel_centers':True})
        if q in [84,96,108,122]:pictures[q]=(x[0,0],{k:v[0].numpy() for k,v in ss.items()},baseline,network[valid&active])
    summary={}
    for model in ['teacher','student']:
        for stage in spec['stages']:
            for mapset in ['full_observed','active_observed']:
                rr=[r for r in rows if r['model']==model and r['stage']==stage and r['map_set']==mapset]
                summary[f'{model}/{stage}/{mapset}']={k:float(np.mean([r[k] for r in rr])) for k in rr[0] if k not in ['query','model','stage','map_set','grid_counts']}
                summary[f'{model}/{stage}/{mapset}']['mean_grid_counts']=np.mean([r['grid_counts'] for r in rr],axis=0).tolist()
    assert all(r['decoder_pixel_shuffle_max_error']==0 for r in coord)
    put(out/'results.json',{'summary':summary,'rows':rows,'map_coverage':maps,'student_to_teacher':distrows,'raw_scores':rawrows})
    put(out/'checks.json',{'status':'PASS','channel_offset_impulses':64,'coordinate_paths':coord,'source_hashes':hashes})
    with (out/'per_query.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(4,4,figsize=(12,18))
    for row,(q,(gray,scores,kps,obs)) in enumerate(pictures.items()):
        for j,name in enumerate(['teacher','student']):
            ax=axes[row,2*j];ax.imshow(np.log10(np.maximum(scores[name],1e-6)),vmin=-5,vmax=0,cmap='magma');ax.set_title(f'{q} {name} log10 score')
            ax=axes[row,2*j+1];ax.imshow(gray,cmap='gray');ax.scatter(obs[:,0],obs[:,1],s=2,c='cyan',alpha=.35);xy=kps[name];ax.scatter(xy[:,0],xy[:,1],s=9,c='red',marker='+');ax.set_title(f'{q} {name}: red KP / cyan map')
        for ax in axes[row]:ax.set_xlim(0,184);ax.set_ylim(320,0);ax.set_xticks([0,46,92,138,184]);ax.set_yticks([0,80,160,240,320]);ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(out/'spatial_examples.png',dpi=130);plt.close(fig)
    put(out/'evidence_hashes.json',{str(f.relative_to(out)):sha(f) for f in sorted(out.rglob('*')) if f.is_file()})
    for k,v in summary.items():
        if k.endswith('active_observed'):print(k,'N',v['keypoints'],'near4',v['keypoints_near_4px'],'fraction',v['fraction_near_4px'],'median',v['nearest_projection_median_px'],flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True,type=Path);run(p.parse_args().output)
