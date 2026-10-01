"""TRAIN-reference / DEV-query real-map diagnostic; never alters the online map."""
import json, sys, time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from .runtime import torch, cv2, ops
from .settings import SUPERPOINT, WORKSPACE
from .io import sha, put, guard
from .onnx_export import load_model, deployment_dir
from .metrics import features
from .nms_diagnosis import greedy
sys.path.insert(0,str(WORKSPACE/'online'))
sys.path.insert(0,'/home/quyet/edge_ai_project/third_party/hloc')
from pipeline import match, normalize
from backends import TorchStudent
from hloc.utils.read_write_model import read_images_binary, read_cameras_binary, read_points3D_binary, qvec2rotmat
from PIL import Image

MAP_CONFIG=json.loads((SUPERPOINT/'configs/localization_map.json').read_text())
DEST=SUPERPOINT/MAP_CONFIG['artifact_dir']
MODEL=Path(MAP_CONFIG['model'])
ASSETS=Path(MAP_CONFIG['assets'])
GLOBAL=Path('/home/quyet/training_recovery/final_gate2')
DATA=Path(MAP_CONFIG['data'])
VARIANTS=('original','greedy_score','greedy_corner')

def context():
    for filename,digest in MAP_CONFIG['geometry_hashes'].items():
        assert sha(MODEL/filename)==digest, 'Active map geometry changed; revalidate before use'
    images=read_images_binary(MODEL/'images.bin')
    cameras=read_cameras_binary(MODEL/'cameras.bin')
    return images,cameras

def preprocess(gray):
    ih,iw=gray.shape; factor=min(288/iw,512/ih)
    nw,nh=round(iw*factor),round(ih*factor); x0,y0=(288-nw)//2,(512-nh)//2
    common=np.zeros((512,288),np.uint8); valid=np.zeros_like(common)
    common[y0:y0+nh,x0:x0+nw]=cv2.resize(gray,(nw,nh),interpolation=cv2.INTER_AREA)
    valid[y0:y0+nh,x0:x0+nw]=1
    im=np.zeros((320,184),np.uint8); mask=np.zeros_like(im)
    im[:,2:182]=cv2.resize(common,(180,320),interpolation=cv2.INTER_AREA)
    mask[:,2:182]=cv2.resize(valid,(180,320),interpolation=cv2.INTER_NEAREST)
    mask=cv2.erode(mask,np.ones((11,11),np.uint8),borderType=cv2.BORDER_CONSTANT,borderValue=0)
    # Exact pixel-center transform through both resize operations and padding.
    sx,sy=nw/iw*180/288,nh/ih*320/512
    tx=(.5*nw/iw+x0)*180/288-.5+2
    ty=(.5*nh/ih+y0)*320/512-.5
    return im[None,None].astype(np.float32)/255,mask[None],np.array([sx,sy,tx,ty])

def prepare():
    guard();DEST.mkdir(parents=True,exist_ok=True)
    images,_=context(); lookup={im.name:i for i,im in images.items()}
    rows=[r for r in json.loads(DATA.read_text())['rows'] if r['domain']=='project']
    ids={s:sorted(lookup[r.get('image_name','/'.join(Path(r['path']).parts[-2:]))] for r in rows if r['split']==s) for s in ('train','dev')}
    assert not set(ids['train'])&set(ids['dev'])
    model,res,checkpoint=load_model('wider_r2_cw'); assert res=='r2'
    dep=deployment_dir('wider_r2_cw')
    cfg=json.loads((WORKSPACE/'online/config.json').read_text())
    spec={'purpose':'Real COLMAP DEV PnP diagnostic of original/greedy NMS, same frozen FP32 and actual INT8 student',
          'split_ids':ids,'variants':list(VARIANTS),'checkpoint_sha256':sha(checkpoint),'kmodel_sha256':sha(dep/'model.kmodel'),
          'code_sha256':sha(__file__),'nms_code_sha256':sha(Path(__file__).with_name('nms_diagnosis.py')),
          'data_sha256':sha(DATA),'geometry_hashes':{n:sha(MODEL/n) for n in ('images.bin','cameras.bin','points3D.bin')},
          'global_checkpoint_sha256':sha(GLOBAL/'candidate.pt'),
          'reference_images':'TRAIN only; all DEV excluded from local and global reference database',
          'reference_association':'nearest COLMAP observation within 2 canonical pixels, one selected feature per Point3D per reference; mean normalized descriptors then normalize',
          'preprocessing':'pilot R2 two-stage letterbox gray/255; border5; pixel-center affine back to original camera',
          'matching_config':cfg,'retrieval':'frozen MixVPR top3 TRAIN refs; add strongest geometric covisibility to max8; max600 valid map points',
          'good_pose':'PnP >=8 inliers, rotation<=5deg, center error<=0.1 native COLMAP units',
          'selection':'diagnostic only, all variants reported; no automatic promotion or threshold tuning',
          'limitations':'DEV engineering; source reconstruction contains query observations. No independent generalization claim; no GT1; no board latency.'}
    path=DEST/'protocol.json'
    if path.exists(): assert json.loads(path.read_text())==spec,'Protocol changed'
    else: put(path,spec)
    global_model=TorchStudent(GLOBAL/'candidate.pt'); torch.set_num_threads(2)
    (DEST/'inputs').mkdir(exist_ok=True);(DEST/'fp32').mkdir(exist_ok=True)
    records=[]
    with torch.inference_mode():
        for k,i in enumerate(ids['train']+ids['dev']):
            guard(); source=ASSETS/images[i].name; bgr=cv2.imread(str(source)); assert bgr is not None
            x,mask,affine=preprocess(cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY))
            input_path=DEST/'inputs'/f'{i}.npz'; output_path=DEST/'fp32'/f'{i}.npz'
            rgb=np.asarray(Image.fromarray(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)).resize((240,240),Image.Resampling.BICUBIC))
            g=normalize(global_model.infer(rgb))
            a,b=model(torch.from_numpy(x))
            np.savez_compressed(input_path,image=x,mask=mask,affine=affine,global_descriptor=g)
            np.savez_compressed(output_path,logits=a.numpy(),desc=b.numpy())
            records.append({'id':i,'source_sha256':sha(source),'input_sha256':sha(input_path),'fp32_sha256':sha(output_path)})
            if k%30==0: print('PREPARE',k+1,len(ids['train'])+len(ids['dev']),flush=True)
    put(DEST/'inputs.json',{'protocol_sha256':sha(path),'records':records})

def extract(i,backend,variant):
    with np.load(DEST/'inputs'/f'{i}.npz') as z: x,mask,affine=(z[n].copy() for n in ('image','mask','affine'))
    with np.load(DEST/backend/f'{i}.npz') as z: a,b=(torch.from_numpy(z[n].copy()) for n in ('logits','desc'))
    if variant=='original': xy,d=features(a,b,mask,'r2')[0]
    else: xy,d=greedy(a,b,mask,'r2',x,variant=='greedy_corner')[0]
    logical=xy*1.25+[2,0]
    original=(logical-affine[2:])/affine[:2]
    return original,d,logical,affine

def evaluate_gate():
    if MAP_CONFIG['map_id']!='stair_6_7_v1':
        raise RuntimeError('Legacy SIMPLE_RADIAL evaluator retired. Use the active map camera backend and a new teacher baseline; never reuse stair caches.')
    p=json.loads((DEST/'protocol.json').read_text()); manifest=json.loads((DEST/'inputs.json').read_text())
    assert manifest['protocol_sha256']==sha(DEST/'protocol.json') and p['code_sha256']==sha(__file__)
    sim=json.loads((DEST/'simulation.json').read_text())
    assert sim['kmodel_sha256']==p['kmodel_sha256'] and sim['inputs_manifest_sha256']==sha(DEST/'inputs.json')
    for row in manifest['records']:
        i=row['id'];assert sha(DEST/'inputs'/f'{i}.npz')==row['input_sha256']
        assert sha(DEST/'fp32'/f'{i}.npz')==row['fp32_sha256']
        assert sha(DEST/'int8'/f'{i}.npz')==sim['outputs'][str(i)]
    images,cameras=context(); points=read_points3D_binary(MODEL/'points3D.bin')
    train,dev=p['split_ids']['train'],p['split_ids']['dev'];cfg=p['matching_config']
    g=np.stack([np.load(DEST/'inputs'/f'{i}.npz')['global_descriptor'] for i in train])
    visible={i:set(int(v) for v in images[i].point3D_ids if v>=0) for i in train}
    retrieval={}
    for i in dev:
        q=np.load(DEST/'inputs'/f'{i}.npz')['global_descriptor']; order=np.argsort(-(g@q),kind='stable')[:cfg['top_k']]
        refs=[train[j] for j in order]; expanded=refs.copy()
        for ref in refs:
            for other in sorted(train,key=lambda j:(-len(visible[ref]&visible[j]),j)):
                if len(expanded)>=cfg['max_active_images']: break
                if other not in expanded and visible[ref]&visible[other]: expanded.append(other)
        retrieval[i]=expanded
    all_results={}
    for backend in ('fp32','int8'):
      for variant in VARIANTS:
        sums={};counts={};obs={};total_selected=0
        for i in train:
            xy,d,logical,affine=extract(i,backend,variant);total_selected+=len(xy)
            im=images[i]; valid=im.point3D_ids>=0; pid=im.point3D_ids[valid]; uv=im.xys[valid]*affine[:2]+affine[2:]
            if len(uv)==0 or len(xy)==0:obs[i]=[];continue
            dist=np.linalg.norm(logical[:,None]-uv[None],axis=2)/1.25
            nearest=dist.argmin(1); seen=set(); obs[i]=[]
            for j in np.argsort(dist[np.arange(len(xy)),nearest],kind='stable'):
                v=int(pid[nearest[j]])
                if dist[j,nearest[j]]>2 or v in seen:continue
                seen.add(v);obs[i].append(v);sums[v]=sums.get(v,np.zeros(256,np.float32))+d[j];counts[v]=counts.get(v,0)+1
        point_ids=np.array(sorted(sums),np.int64);rowof={v:j for j,v in enumerate(point_ids)}
        local=normalize(np.stack([sums[v] for v in point_ids])) if len(point_ids) else np.empty((0,256),np.float32)
        xyz=np.array([points[v].xyz for v in point_ids]).reshape(-1,3)
        identity=f"{p['checkpoint_sha256']}:{backend}:{variant}:"+sha(DEST/'protocol.json')
        folder=DEST/f'map_{backend}_{variant}';folder.mkdir(exist_ok=True)
        for name,value in [('point_ids',point_ids),('local',local),('xyz',xyz)]:np.save(folder/f'{name}.npy',value)
        put(folder/'identity.json',{'local_identity':identity,'references':train,'point_count':len(point_ids),'source_protocol_sha256':sha(DEST/'protocol.json'),
            'files':{n:sha(folder/f'{n}.npy') for n in ('point_ids','local','xyz')}})
        store=SimpleNamespace(local=local,xyz=xyz,point_ids=point_ids,identity=identity); results=[]
        for i in dev:
            start=time.perf_counter();xy,d,_,_=extract(i,backend,variant)
            assert store.identity==identity
            active=[];seen=set()
            for ref in retrieval[i]:
                for v in sorted(obs[ref]):
                    if v not in seen:seen.add(v);active.append(rowof[v])
            active=np.array(active[:cfg['max_active_points']],np.int64)
            matches,raw=match(d,store,active,cfg)
            r={'id':i,'status':'insufficient_matches','good_pose':False,'keypoints':len(xy),'active_points':len(active),'correspondences':len(matches),'raw_matches':raw,'inliers':0,'references':retrieval[i]}
            if len(matches)>=cfg['min_inliers']:
                cam=cameras[images[i].camera_id]; assert cam.model=='SIMPLE_RADIAL';f,cx,cy,k=cam.params
                K=np.array([[f,0,cx],[0,f,cy],[0,0,1]],np.float64);distortion=np.array([k,0,0,0,0],np.float64)
                xyzq=np.array([xyz[v] for j,v,s in matches],np.float64); pixels=np.array([xy[j] for j,v,s in matches],np.float64)
                cv2.setRNGSeed(17)
                ok,rv,tv,inliers=cv2.solvePnPRansac(xyzq,pixels,K,distortion,iterationsCount=100,reprojectionError=cfg['pnp_error_px'],confidence=.99,flags=cv2.SOLVEPNP_EPNP)
                r['status']='pnp_failed';r['inliers']=0 if inliers is None else len(inliers)
                if ok and r['inliers']>=cfg['min_inliers']:
                    R=cv2.Rodrigues(rv)[0];center=(-R.T@tv).ravel();gtR=qvec2rotmat(images[i].qvec)
                    depth=(xyzq[inliers.ravel()]@R.T+tv.ravel())[:,2]
                    angle=float(np.degrees(np.arccos(np.clip((np.trace(R@gtR.T)-1)/2,-1,1))))
                    error=float(np.linalg.norm(center+gtR.T@images[i].tvec))
                    r.update(status='pose' if np.isfinite(center).all() and (depth>0).all() else 'invalid_pose',rotation_deg=angle,center_native=error)
                    r['good_pose']=r['status']=='pose' and angle<=5 and error<=.1
            r['cached_decode_match_pnp_ms']=(time.perf_counter()-start)*1000;results.append(r)
        summary={'good_poses':sum(r['good_pose'] for r in results),'poses':sum(r['status']=='pose' for r in results),'queries':len(dev),'map_points':len(point_ids),
                 'reference_selected_features':total_selected,'associated_observations':sum(counts.values()),'median_matches':float(np.median([r['correspondences'] for r in results])),
                 'median_inliers':float(np.median([r['inliers'] for r in results]))}
        all_results[f'{backend}_{variant}']={'summary':summary,'frames':results};print(backend,variant,json.dumps(summary),flush=True)
        put(DEST/'results.json',{'protocol_sha256':sha(DEST/'protocol.json'),'results':all_results,'complete':len(all_results)==6,'timing':'cached outputs; excludes network inference, not full pipeline or board latency'})

if __name__=='__main__':
    {'prepare':prepare,'evaluate':evaluate_gate}[sys.argv[1]]()
