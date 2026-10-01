"""Sealed sequence TEST runner. Requires an explicitly frozen release manifest."""
import os
os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('MKL_NUM_THREADS','2')
import _bootstrap
import argparse,json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from spk210.settings import SUPERPOINT,ROOT
from spk210.io import put,sha
BASE=SUPERPOINT/'artifacts/full_training';OUT=BASE/'final_test';NAME='full_capacity_r2_cw'

def contract():
    release=json.loads((BASE/'freeze_manifest.json').read_text());assert release['status']=='FROZEN_FOR_FINAL_TEST' and release['policy_authorized']
    for f,h in release['files'].items():assert sha(Path(f))==h,f
    p=json.loads((BASE/'protocol.json').read_text());assert release['checkpoint_sha256']==sha(BASE/'best.pt');assert len(p['test'])==1000
    assert sha(Path(__file__)) == release['test_runner_sha256']
    return release,p

def begin(stage):
    OUT.mkdir(exist_ok=True)
    with (OUT/f'{stage}.started.json').open('x') as f:
        json.dump({'stage':stage,'started_utc':datetime.now(timezone.utc).isoformat(),'freeze_manifest_sha256':sha(BASE/'freeze_manifest.json'),'policy':'one pass; refusing a second execution of this stage'},f)

def make_pair(row,i):
    import cv2
    image=cv2.imread(row['path'],0);assert image is not None;ih,iw=image.shape;factor=min(288/iw,512/ih);nw,nh=round(iw*factor),round(ih*factor);x0,y0=(288-nw)//2,(512-nh)//2
    common=np.zeros((512,288),np.uint8);mask=np.zeros_like(common);common[y0:y0+nh,x0:x0+nw]=cv2.resize(image,(nw,nh),interpolation=cv2.INTER_AREA);mask[y0:y0+nh,x0:x0+nw]=1
    rng=np.random.default_rng(20260919+100000+i);H=np.eye(3);H[:2]=cv2.getRotationMatrix2D((72,128),rng.uniform(-10,10),rng.uniform(.92,1.08));H[:2,2]+=rng.uniform(-4,4,2);brightness=rng.uniform(.8,1.2);S=np.array([[1.25,0,2],[0,1.25,0],[0,0,1.]]);warp=S@H@np.linalg.inv(S)
    im=np.zeros((320,184),np.uint8);valid=np.zeros_like(im);im[:,2:182]=cv2.resize(common,(180,320),interpolation=cv2.INTER_AREA);valid[:,2:182]=cv2.resize(mask,(180,320),interpolation=cv2.INTER_NEAREST);other=cv2.warpPerspective(im,warp,(184,320));vm=cv2.warpPerspective(valid,warp,(184,320),flags=cv2.INTER_NEAREST)
    pair=np.stack([im,np.clip(other.astype(float)*brightness,0,255).astype(np.uint8)])[:,None].astype(np.float32)/255;masks=np.stack([cv2.erode(v,np.ones((11,11),np.uint8),borderType=cv2.BORDER_CONSTANT,borderValue=0) for v in [valid,vm]])
    m0,m1=masks.copy();masks[0]&=cv2.warpPerspective(m1,np.linalg.inv(warp),(184,320),flags=cv2.INTER_NEAREST);masks[1]&=cv2.warpPerspective(m0,warp,(184,320),flags=cv2.INTER_NEAREST)
    return {'pair':pair,'masks':masks,'homography':H,'content_rect':np.array([x0/2,y0/2,nw/2,nh/2])}

def prepare():
    from spk210.runtime import torch
    from spk210.onnx_export import load_model
    from spk210.metrics import features,pair_metrics
    release,p=contract();begin('prepare');(OUT/'inputs').mkdir(exist_ok=True);model,_,_=load_model(NAME);rows=[];inputs={}
    # No rejection by corner/keypoint count: empty-feature TEST images score zero.
    with torch.inference_mode():
        for i,row in enumerate(p['test']):
            assert sha(row['path'])==row['sha256'];z=make_pair(row,i);f=OUT/'inputs'/f'{i:04d}.npz';np.savez_compressed(f,**z);inputs[str(i)]=sha(f);a,b=model(torch.from_numpy(z['pair']));fs=features(a,b,z['masks'],'r2',nms=release['nms']);m=pair_metrics(*fs,z['homography'],z['content_rect']);m['mean_keypoints']=float(np.mean([len(v[0]) for v in fs]));rows.append({'index':i,'metrics':m})
            if (i+1)%100==0:print('FP32 TEST',i+1,flush=True)
    put(OUT/'fp32.json',{'rows':rows,'macro':{k:float(np.mean([r['metrics'][k] for r in rows])) for k in rows[0]['metrics']}});put(OUT/'inputs.json',inputs)

def simulate(shard,shards):
    import nncase, _nncase
    from importlib.metadata import version
    assert version('nncase')=='1.8.0.20220929' and _nncase.__version__=='1.8.0-55be52f'
    release,p=contract();assert shards==release['test_shards'] and 0<=shard<shards;begin(f'simulate_{shard}');inputs=json.loads((OUT/'inputs.json').read_text());(OUT/'int8').mkdir(exist_ok=True);sim=nncase.Simulator();sim.load_model(Path(release['kmodel']).read_bytes());perm=np.r_[np.rot90(np.arange(64).reshape(8,8),-1).ravel(),64];records={}
    for i in range(shard,len(p['test']),shards):
        f=OUT/'inputs'/f'{i:04d}.npz';assert sha(f)==inputs[str(i)]
        with np.load(f) as z:pair=z['pair'].copy()
        heads=[[],[]]
        for view in pair:
            x=np.ascontiguousarray(np.rot90(view[None],-1,(2,3)));sim.set_input_tensor(0,nncase.RuntimeTensor.from_numpy(x));sim.run()
            for j in [0,1]:
                v=np.rot90(sim.get_output_tensor(j).to_numpy().copy(),1,(2,3));v=v[:,np.argsort(perm)] if j==0 else v;assert np.isfinite(v).all();heads[j].append(v)
        dest=OUT/'int8'/f'{i:04d}.npz';np.savez_compressed(dest,logits=np.concatenate(heads[0]),desc=np.concatenate(heads[1]));records[str(i)]=sha(dest)
        if len(records)%50==0:print('INT8 TEST shard',shard,'pairs',len(records),flush=True)
    put(OUT/f'simulation_{shard}.json',{'shard':shard,'shards':shards,'kmodel_sha256':sha(Path(release['kmodel'])),'outputs':records})

def evaluate(shards):
    from spk210.runtime import torch
    from spk210.metrics import features,pair_metrics
    release,p=contract();assert shards==release['test_shards'];begin('evaluate');inputs=json.loads((OUT/'inputs.json').read_text());outputs={};rows=[]
    for j in range(shards):
        r=json.loads((OUT/f'simulation_{j}.json').read_text());assert r['shards']==shards and r['kmodel_sha256']==sha(Path(release['kmodel']));assert not outputs.keys()&r['outputs'].keys();outputs.update(r['outputs'])
    assert set(outputs)=={str(i) for i in range(len(p['test']))}
    with torch.inference_mode():
        for i in range(len(p['test'])):
            src=OUT/'inputs'/f'{i:04d}.npz';raw=OUT/'int8'/f'{i:04d}.npz';assert sha(src)==inputs[str(i)] and sha(raw)==outputs[str(i)]
            with np.load(src) as z:masks=z['masks'].copy();H=z['homography'].copy();rect=z['content_rect'].copy()
            with np.load(raw) as z:a=torch.from_numpy(z['logits'].copy());b=torch.from_numpy(z['desc'].copy())
            fs=features(a,b,masks,'r2',nms=release['nms']);m=pair_metrics(*fs,H,rect);m['mean_keypoints']=float(np.mean([len(v[0]) for v in fs]));rows.append({'index':i,'metrics':m})
    fp=json.loads((OUT/'fp32.json').read_text());macro={k:float(np.mean([r['metrics'][k] for r in rows])) for k in rows[0]['metrics']};retention=macro['correct_mnn_count']/max(fp['macro']['correct_mnn_count'],1e-12);drop=100*(fp['macro']['mnn_precision']-macro['mnn_precision']);g=release['test_gate'];passed=retention>=g['correct_retention_min'] and drop<=g['precision_drop_pp_max']
    put(OUT/'results.json',{'status':'TEST_RECORDED_NO_TUNING','test_gate_pass':passed,'fp32':fp['macro'],'int8':macro,'correct_retention':retention,'precision_drop_pp':drop,'rows':rows,'count':len(rows),'scope':'One heldout sequence, synthetic homography feature/matching test; not public SuperPoint protocol reproduction or independent localization pose accuracy. No changes selected from these results.'});print('TEST PASS',passed,'retention',retention,'precision drop pp',drop,flush=True)
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('stage',choices=['prepare','simulate','evaluate']);a.add_argument('--shard',type=int,default=0);a.add_argument('--shards',type=int,default=2);v=a.parse_args()
    if v.stage=='prepare':prepare()
    elif v.stage=='simulate':simulate(v.shard,v.shards)
    else:evaluate(v.shards)
