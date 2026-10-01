"""Evaluate A/B candidates on frozen natural DEV, with real nncase outputs."""
import _bootstrap
import argparse,json,os
from pathlib import Path
import numpy as np
from spk210.settings import SUPERPOINT
from spk210.io import sha,put,guard

BASE=Path(os.environ.get('SP_EXPERIMENT_ROOT', str(SUPERPOINT/'artifacts/descriptor_kd_ab')))
OLD=SUPERPOINT/'artifacts/map_nms_gate'
FIXED=SUPERPOINT/'artifacts/fixed_keypoint_pnp/run_001'


def read(p):return json.loads(p.read_text())


def paths(candidate):
    out=BASE/candidate/'natural';out.mkdir(exist_ok=True)
    p=read(FIXED/'protocol.json')
    nr=read(SUPERPOINT/'artifacts/measurement_gate/run_002/natural_descriptor_probe.json')['rows']
    pairs=[(r['query'],r['reference']) for r in nr if r['model']=='teacher_fp32' and r['offset']==-.5]
    ids=sorted(set(p['query_ids']+[i for refs in p['references'].values() for i in refs]+[i for pair in pairs for i in pair]))
    dep=SUPERPOINT/'artifacts/deployment'/candidate
    return out,p,pairs,ids,dep


def simulate(candidate):
    import nncase
    out,p,pairs,ids,dep=paths(candidate)
    spec=read(dep/'compile.json');assert sha(dep/'model.kmodel')==spec['kmodel_sha256']
    sim=nncase.Simulator();sim.load_model((dep/'model.kmodel').read_bytes())
    assert sim.get_input_tensor(0).dtype==np.dtype('float32')
    assert all(sim.get_output_tensor(j).dtype==np.dtype('float32') for j in range(2))
    target=out/'int8';target.mkdir(exist_ok=True);perm=np.r_[np.rot90(np.arange(64).reshape(8,8),-1).ravel(),64]
    entries={r['id']:r for r in read(OLD/'inputs.json')['records']};records={}
    for i in ids:
        guard();src=OLD/'inputs'/f'{i}.npz';assert sha(src)==entries[i]['input_sha256']
        with np.load(src) as z:x=np.ascontiguousarray(np.rot90(z['image'],-1,(2,3)))
        sim.set_input_tensor(0,nncase.RuntimeTensor.from_numpy(x));sim.run()
        a,b=[np.rot90(sim.get_output_tensor(j).to_numpy().copy(),1,(2,3)) for j in range(2)];a=a[:,np.argsort(perm)]
        assert a.shape==(1,65,40,23) and b.shape==(1,256,40,23) and np.isfinite(a).all() and np.isfinite(b).all()
        f=target/f'{i}.npz';np.savez_compressed(f,logits=a,desc=b);records[str(i)]={'input_sha256':sha(src),'output_sha256':sha(f)}
    put(out/'simulation.json',{'backend':'actual nncase K210 Simulator','checkpoint_sha256':spec['checkpoint_sha256'],
                             'kmodel_sha256':spec['kmodel_sha256'],'source_sha256':sha(__file__),'records':records})
    print('NATURAL_SIM_COMPLETE',candidate,len(ids),flush=True)


def evaluate(candidate):
    from spk210.runtime import torch,ops,cv2
    import torch.nn.functional as F
    from spk210.onnx_export import load_model
    from spk210.metrics import features
    from fixed_keypoint_pnp import context,MODEL,read_points3D_binary,pose,match,camera,qvec2rotmat
    out,p,pairs,ids,dep=paths(candidate);model,res,checkpoint=load_model(candidate);assert res=='r2'
    sim=read(out/'simulation.json');assert sim['checkpoint_sha256']==sha(checkpoint)
    assert sim['kmodel_sha256']==sha(dep/'model.kmodel')
    images,cameras=context();points=read_points3D_binary(MODEL/'points3D.bin')
    for f,digest in p['geometry_hashes'].items():assert sha(MODEL/f)==digest
    common=read(FIXED/'common_inputs.json');queries=p['query_ids'];references={int(k):v for k,v in p['references'].items()}
    refs=set(r for row in references.values() for r in row);assert not refs&set(queries)
    spec={'candidate':candidate,'checkpoint_sha256':sha(checkpoint),'kmodel_sha256':sim['kmodel_sha256'],
          'training_protocol_sha256':sha(BASE/'protocol.json'),'localization_protocol_sha256':sha(FIXED/'protocol.json'),
          'common_inputs_sha256':sha(FIXED/'common_inputs.json'),'source_sha256':sha(__file__),
          'fixed':'exact teacher keypoints/active IDs/references as preceding fixed-keypoint gate',
          'own':'candidate own original-NMS keypoints; same masks cap160 threshold.005 radius4; candidate descriptors both query/reference',
          'natural':'same20 oracle known-track pairs as preceding descriptor gate, offset-.5;512 ref/256 common tracks; per-pair mean',
          'scope':'DEV engineering; fixed is teacher-detector hybrid, own is candidate feature pipeline on PC; no board timing'}
    put(out/'protocol.json',spec)
    result={}
    for backend in ['fp32','int8']:
        banks={};fixed_desc={};own_desc={};own_xy={};tracks={};raw_records={}
        rawdir=out/backend;rawdir.mkdir(exist_ok=True)
        with torch.inference_mode():
            for index,i in enumerate(ids):
                src=OLD/'inputs'/f'{i}.npz';assert sha(src)==sim['records'][str(i)]['input_sha256']
                with np.load(src) as z:x=z['image'].copy();aff=z['affine'].copy();mask=z['mask'].copy()
                if backend=='fp32':
                    a,b=model(torch.from_numpy(x));np.savez_compressed(rawdir/f'{i}.npz',logits=a.numpy(),desc=b.numpy())
                else:
                    assert sha(rawdir/f'{i}.npz')==sim['records'][str(i)]['output_sha256']
                    with np.load(rawdir/f'{i}.npz') as z:a=torch.from_numpy(z['logits'].copy());b=torch.from_numpy(z['desc'].copy())
                raw_records[str(i)]=sha(rawdir/f'{i}.npz')
                def sample(xy):return ops.sample_descriptors(torch.from_numpy(np.asarray(xy,np.float32))[None],F.normalize(b,dim=1),8)[0].T.numpy()
                if i in refs:
                    with np.load(FIXED/'teacher_fp32'/f'{i}.npz') as z:pid=z['ids'].copy();xy=z['network_xy'].copy()
                    banks[i]=(pid,sample(xy))
                if i in queries:
                    with np.load(FIXED/'teacher_fp32'/f'{i}.npz') as z:xy=z['network_xy'].copy()
                    fixed_desc[i]=sample(xy)
                    canonical,desc=features(a,b,mask,'r2')[0]
                    own_xy[i]=(canonical*1.25+[2,0]-aff[2:])/aff[:2]+.5;own_desc[i]=desc
                im=images[i];valid=im.point3D_ids>=0;pid=im.point3D_ids[valid];source=im.xys[valid]
                u,c=np.unique(pid,return_counts=True);keep=~np.isin(pid,u[c>1]);pid,source=pid[keep],source[keep]
                xy=(source-.5)*aff[:2]+aff[2:];ij=np.rint(xy).astype(int)
                inside=(ij[:,0]>=0)&(ij[:,0]<184)&(ij[:,1]>=0)&(ij[:,1]<320);keep=np.zeros(len(pid),bool)
                keep[inside]=mask[0,ij[inside,1],ij[inside,0]]>0;pid,xy=pid[keep],xy[keep];order=np.argsort(pid,kind='stable')
                tracks[i]=(pid[order],sample(xy[order]))
        descrows=[]
        for q,r in pairs:
            qi,qd=tracks[q];ri,rd=tracks[r];ri,rd=ri[:512],rd[:512];shared=np.intersect1d(qi,ri)[:256]
            assert len(shared)>0
            score=qd[np.searchsorted(qi,shared)]@rd.T;true=np.searchsorted(ri,shared)
            ranks=np.argmax(np.argsort(-score,axis=1,kind='stable')==true[:,None],axis=1)+1
            descrows.append({'query':q,'reference':r,'tracks':len(shared),'top1':float(np.mean(ranks==1)),'top5':float(np.mean(ranks<=5))})
        result[backend]={'descriptor':{'top1':float(np.mean([r['top1'] for r in descrows])),
                                     'top5':float(np.mean([r['top5'] for r in descrows]))},'descriptor_rows':descrows,'raw_hashes':raw_records}
        for mode in ['fixed','own']:
            rows=[]
            for q in queries:
                desc=fixed_desc[q] if mode=='fixed' else own_desc[q]
                xy=np.array(common['query_colmap_xy'][str(q)]) if mode=='fixed' else own_xy[q]
                matched=match(desc,banks,references[q],common['active_ids'][str(q)])
                xyz=np.array([points[v].xyz for j,v,s in matched],float).reshape(-1,3);pixels=xy[np.array([j for j,v,s in matched],int)]
                im=images[q];cam=cameras[im.camera_id];row=pose(xyz,pixels,cam,im)
                correct=0
                if len(xyz):
                    K,dist=camera(cam);R=qvec2rotmat(im.qvec);uv,_=cv2.projectPoints(xyz,cv2.Rodrigues(R)[0],im.tvec,K,dist)
                    correct=int(((np.linalg.norm(uv[:,0]-pixels,axis=1)<=4)&((xyz@R.T+im.tvec)[:,2]>0)).sum())
                row.update(query=q,keypoints=len(xy),correct_2d3d_geometric=correct);rows.append(row)
            result[backend][mode]={'good_poses':sum(r['good_pose'] for r in rows),'queries':len(rows),
                                   'mean_inliers':float(np.mean([r['inliers'] for r in rows])),
                                   'mean_correct_matches':float(np.mean([r['correct_2d3d_geometric'] for r in rows])),'rows':rows}
        print(candidate,backend,'descriptor',result[backend]['descriptor'],'fixed',result[backend]['fixed']['good_poses'],'own',result[backend]['own']['good_poses'],flush=True)
    put(out/'results.json',{'status':'COMPLETE','protocol_sha256':sha(out/'protocol.json'),'results':result})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['simulate','evaluate']);p.add_argument('--candidate',required=True)
    a=p.parse_args();(simulate if a.stage=='simulate' else evaluate)(a.candidate)
