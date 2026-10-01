"""Fresh floor6 teacher baseline, frozen retrieval, native fisheye, internal DEV."""
import _bootstrap
import argparse,csv,json,shutil
from pathlib import Path
import numpy as np
import torch.nn.functional as F
from PIL import Image
from spk210.map_nms_gate import context,MODEL,ASSETS,GLOBAL,MAP_CONFIG,preprocess,read_points3D_binary,normalize,TorchStudent
from spk210.map_geometry import pose,project,camera
from spk210.runtime import torch,cv2,ops,w0,SuperPointKPUWrapper
from spk210.metrics import features
from spk210.settings import SUPERPOINT
from spk210.io import sha,put,guard
from fixed_keypoint_pnp import match

@torch.inference_mode()
def run(out):
    out.mkdir(parents=True,exist_ok=False)
    images,cameras=context();points=read_points3D_binary(MODEL/'points3D.bin')
    data=json.loads(Path(MAP_CONFIG['data']).read_text());lookup={im.name:i for i,im in images.items()}
    queries=[lookup[r['image_name']] for r in data['rows'] if r['split']=='dev'];refs=[lookup[r['image_name']] for r in data['rows'] if r['split']=='train']
    assert len(queries)==20 and len(refs)==1124 and not set(queries)&set(refs)
    spec={'scope':'fresh floor6 teacher FP32 low-resolution baseline;20 internal DEV queries, not independent TEST',
          'map_config':MAP_CONFIG,'query_ids':queries,'reference_ids':refs,'data_sha256':sha(MAP_CONFIG['data']),
          'teacher_sha256':sha(w0.CHECKPOINT),'global_checkpoint_sha256':sha(GLOBAL/'candidate.pt'),
          'source_sha256':sha(__file__),'geometry_source_sha256':sha(SUPERPOINT/'src/spk210/map_geometry.py'),
          'preprocessing':'same logical184x320: common288x512 then180x320 with2px left/right pad, validmask erosion5, gray/255; exact pixel-center affine',
          'teacher_keypoints':'unchanged original NMS4,threshold.005,cap160; original normalized descriptor sampling stride8',
          'reference_descriptors':'fresh teacher descriptors at COLMAP observations-.5 mapped to network; all duplicated IDs per image excluded; mask checked; sorted IDs',
          'retrieval':'unchanged frozen MixVPR student RGB240bicubic normalize; new global descriptors for all1144 images; top3 reference cosine, covisibility expansion to8 reference images; no query in reference pool',
          'active_points':'union of selected reference valid IDs in reference order, sorted within reference, first600unique',
          'matcher':'unchanged per-reference cosine>=.8,best-second>=.05; unique query/Point3D,cap120',
          'pnp':'native pycolmap3.13 camera RADIAL_FISHEYE; LO-RANSAC pixelerror4,100..1000trials,confidence.99,seed17; minimum8matches',
          'good_pose':'at least8inliers, positive inlierdepths,finite, rotation<=5degrees, center<=.1native map units',
          'limitations':'query observations exist in source map; old-track descriptor resampling not teacher-specific reconstruction; no independent metric-scale GT; not comparable to retired stair scores',
          'no_training':True,'no_threshold_search':True,'no_runtime_promotion':True}
    put(out/'protocol.json',spec);shutil.copyfile(__file__,out/'evaluated_source.py')
    global_model=TorchStudent(GLOBAL/'candidate.pt');torch.set_num_threads(2)
    vectors=[];source_records=[];ordered=refs+queries
    for k,i in enumerate(ordered):
        guard();f=ASSETS/images[i].name
        with Image.open(f) as image:
            assert image.size==(cameras[images[i].camera_id].width,cameras[images[i].camera_id].height)
            rgb=np.asarray(image.convert('RGB').resize((240,240),Image.Resampling.BICUBIC))
        vectors.append(normalize(global_model.infer(rgb)));source_records.append({'id':i,'name':images[i].name,'sha256':sha(f)})
        if (k+1)%200==0:print('GLOBAL',k+1,len(ordered),flush=True)
    vectors=np.stack(vectors);assert np.isfinite(vectors).all()
    np.savez_compressed(out/'global_descriptors.npz',ids=np.array(ordered),descriptors=vectors)
    put(out/'source_images.json',source_records)
    visible={i:set(int(v) for v in images[i].point3D_ids if v>=0) for i in refs}
    reference_lists={};retrieval={}
    for j,q in enumerate(queries):
        scores=vectors[:len(refs)]@vectors[len(refs)+j];order=sorted(range(len(refs)),key=lambda k:(-float(scores[k]),refs[k]));seeds=[refs[k] for k in order[:3]]
        counts={r:sum(len(visible[r]&visible[s]) for s in seeds) for r in refs if r not in seeds}
        expanded=sorted(counts,key=lambda r:(-counts[r],r));chosen=seeds+[r for r in expanded if counts[r]>0][:5]
        assert q not in chosen;reference_lists[q]=chosen;retrieval[q]={'seeds':seeds,'scores':[float(scores[refs.index(s)]) for s in seeds],'references':chosen}
    put(out/'retrieval.json',retrieval)
    ids=sorted(set(queries+[r for rr in reference_lists.values() for r in rr]));teacher,_=w0.import_original_superpoint();teacher=SuperPointKPUWrapper(teacher).eval()
    del global_model
    for folder in ['inputs','teacher','features']:(out/folder).mkdir()
    banks={};qfeatures={};allobs={};affines={};duplicate_ids={};source_hashes={r['id']:r['sha256'] for r in source_records}
    for k,i in enumerate(ids):
        guard();im=images[i];f=ASSETS/im.name;assert sha(f)==source_hashes[i];gray=cv2.imread(str(f),0);x,mask,aff=preprocess(gray);affines[i]=aff
        a,b=teacher(torch.from_numpy(x));assert np.isfinite(a.numpy()).all() and np.isfinite(b.numpy()).all()
        np.savez_compressed(out/'inputs'/f'{i}.npz',image=x,mask=mask,affine=aff)
        np.savez_compressed(out/'teacher'/f'{i}.npz',logits=a.numpy(),desc=b.numpy())
        if i in queries:
            canonical,d=features(a,b,mask,'r2')[0];network=canonical*1.25+[2,0];xy=(network-aff[2:])/aff[:2]+.5;pid=np.arange(len(xy));qfeatures[i]=(xy,d)
            np.testing.assert_allclose((xy-.5)*aff[:2]+aff[2:],network,atol=1e-10,rtol=0)
        else:
            valid=im.point3D_ids>=0;pid=im.point3D_ids[valid];uv=im.xys[valid];u,n=np.unique(pid,return_counts=True);dup=u[n>1];duplicate_ids[i]=dup.tolist();keep=~np.isin(pid,dup);pid,uv=pid[keep],uv[keep]
            network=(uv-.5)*aff[:2]+aff[2:];ij=np.rint(network).astype(int);inside=(ij[:,0]>=0)&(ij[:,0]<184)&(ij[:,1]>=0)&(ij[:,1]<320)
            keep=np.zeros(len(pid),bool);keep[inside]=mask[0,ij[inside,1],ij[inside,0]]>0;pid,network=pid[keep],network[keep];sort=np.argsort(pid,kind='stable');pid,network=pid[sort],network[sort]
            d=ops.sample_descriptors(torch.from_numpy(network.astype(np.float32))[None],F.normalize(b,dim=1),8)[0].T.numpy();banks[i]=(pid,d);allobs[i]=pid.tolist()
        np.savez_compressed(out/'features'/f'{i}.npz',ids=pid,network_xy=network,desc=d)
        if (k+1)%25==0:print('TEACHER',k+1,len(ids),flush=True)
    rows=[];sanity=[];details={};active_by_query={};coverage=[]
    for q in queries:
        im=images[q];cam=cameras[im.camera_id];ref=reference_lists[q];seen=set();pool=[]
        for r in ref:
            for pid in allobs[r]:
                if pid not in seen:seen.add(pid);pool.append(pid)
        active=pool[:600];active_by_query[q]=active;xy,d=qfeatures[q];mm=match(d,banks,ref,active)
        ix=np.array([j for j,v,s in mm],int);xyz=np.array([points[v].xyz for j,v,s in mm]).reshape(-1,3);pixels=xy[ix]
        result=pose(xyz,pixels,cam,im);assert result==pose(xyz,pixels,cam,im)
        errors=np.linalg.norm(project(xyz,im,cam)-pixels,axis=1) if len(xyz) else np.empty(0)
        # Native camera projection returns NaNs for unsupported/behind-camera rays.
        result.update(query=q,image=im.name,keypoints=len(xy),matches=len(mm),correct_geometric_matches=int((np.isfinite(errors)&(errors<=4)).sum()),inlier_ratio=result['inliers']/len(mm) if mm else 0.)
        rows.append(result);details[q]={'query_indices':ix.tolist(),'point3d_ids':[v for j,v,s in mm],'scores':[s for j,v,s in mm],'reprojection_px':[float(v) if np.isfinite(v) else None for v in errors]}
        valid=im.point3D_ids>=0;pid=im.point3D_ids[valid];obs=im.xys[valid];u,n=np.unique(pid,return_counts=True);keep=~np.isin(pid,u[n>1]);pid,obs=pid[keep],obs[keep]
        gt=pose(np.array([points[int(v)].xyz for v in pid]),obs,cam,im);assert gt['good_pose'];sanity.append(dict(query=q,**gt))
        coverage.append({'query':q,'observed_unique':len(pid),'in_selected_references':len(set(pid)&set(pool)),'in_active600':len(set(pid)&set(active)),'reference_pool_points':len(pool),'active_points':len(active)})
    summary={'queries':len(rows),'good_poses':sum(r['good_pose'] for r in rows),'geometry_sanity_good':sum(r['good_pose'] for r in sanity)}
    for k in ['keypoints','matches','correct_geometric_matches','inliers','inlier_ratio']:summary['mean_'+k]=float(np.mean([r[k] for r in rows]))
    put(out/'common_inputs.json',{'query_colmap_xy':{q:v[0].tolist() for q,v in qfeatures.items()},'active_ids':active_by_query,'references':reference_lists,'excluded_duplicate_ids':duplicate_ids})
    put(out/'results.json',{'status':'COMPLETE','summary':summary,'rows':rows,'geometry_sanity':sanity,'coverage':coverage})
    put(out/'match_details.json',details)
    with (out/'per_query.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    assert sha(GLOBAL/'candidate.pt')==spec['global_checkpoint_sha256'] and sha(w0.CHECKPOINT)==spec['teacher_sha256']
    put(out/'checks.json',{'status':'PASS','fresh_source_images':len(source_records),'fresh_local_images':len(ids),'pnp_replays':20,'self_matches':0,'geometry_sanity':20,'training':False})
    put(out/'evidence_hashes.json',{str(f.relative_to(out)):sha(f) for f in sorted(out.rglob('*')) if f.is_file()})
    print('RESULT',json.dumps(summary),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);run(p.parse_args().output)
