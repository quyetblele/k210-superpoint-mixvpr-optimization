"""Matched four-way quality pilot; logical portrait inputs, no test-set access."""
import os
os.environ.update(OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
import sys, json, hashlib, gc, time, resource, fcntl, argparse
from pathlib import Path
import numpy as np
import cv2
import torch
from torch import nn
import torch.nn.functional as F
sys.path.insert(0, '/home/quyet/k210_lab/scripts')
import create_superpoint_k210_w0 as w0
from superpoint_kpu_wrapper import SuperPointKPUWrapper
sys.path.insert(0, str(w0.EDGE / 'third_party'))
from SuperGluePretrainedNetwork.models import superpoint as ops
torch.set_num_threads(2); torch.use_deterministic_algorithms(True); cv2.setNumThreads(1)
ROOT = Path(__file__).parent / 'quality_pilot'
WIDTHS = {'fr12': [12,24,32,64], 'wider': [24,32,64,96]}
RES = {'r1': (144,256,1.,0), 'r2': (184,320,1.25,2)}
SEED = 20260919
STEPS = 400

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''): h.update(b)
    return h.hexdigest()
def put(path, value):
    t=Path(str(path)+'.tmp'); t.write_text(json.dumps(value,indent=2)+'\n'); t.replace(path)
def save(path, value):
    t=Path(str(path)+'.tmp'); torch.save(value,t); t.replace(path)
def guard():
    available=int(next(x for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')).split()[1])/1024
    if available<1536: raise RuntimeError('Available host RAM below 1536 MiB; resume from saved checkpoint')
    import shutil
    if shutil.disk_usage(ROOT).free<2*1024**3: raise RuntimeError('Less than 2 GiB free disk')

class Model(nn.Module):
    def __init__(self, widths):
        super().__init__(); ci=1
        for stage,co in enumerate(widths,1):
            setattr(self,f'conv{stage}a',nn.Conv2d(ci,co,3,padding=1)); setattr(self,f'conv{stage}b',nn.Conv2d(co,co,3,padding=1)); ci=co
        self.convPa=nn.Conv2d(ci,128,3,padding=1); self.convPb=nn.Conv2d(128,65,1)
        self.convDa=nn.Conv2d(ci,128,3,padding=1); self.convDb=nn.Conv2d(128,256,1)
    def forward(self,x):
        for stage in range(1,5):
            x=F.relu(getattr(self,f'conv{stage}a')(x)); x=F.relu(getattr(self,f'conv{stage}b')(x))
            if stage<4: x=F.max_pool2d(x,2)
        return self.convPb(F.relu(self.convPa(x))),self.convDb(F.relu(self.convDa(x)))

def protocol():
    old=Path('/home/quyet/training_recovery/sp_ab/protocol.json'); source=Path('/mnt/d/k210_official_ab/data.json')
    d=json.loads(old.read_text()); full=json.loads(source.read_text())['rows']; chosen={}
    for split,n in [('train',28),('dev',14)]:
        rr=[r for r in full if r['domain']=='project' and r['split']==split]
        chosen[split]=d[split.upper()]+[rr[int(i)] for i in np.linspace(0,len(rr)-1,n,dtype=int)]
    assert not {r['sha256'] for r in chosen['train']} & {r['sha256'] for r in chosen['dev']}
    p={'status':'FROZEN_BEFORE_RUN','seed':SEED,'updates_each':STEPS,'eval_steps':[0,100,200,300,400],
       'widths':WIDTHS,'resolutions':RES,'head_hidden':128,'outputs':[65,256], 'rows':chosen,
       'data_source_sha256':sha(source),'old_indoor_manifest_sha256':sha(old),'teacher_sha256':sha(w0.CHECKPOINT),
       'preprocessing':'gray /255, common aspect-preserving letterbox at 288x512, then resize common content to144x256 or180x320 +2px sides; full source FOV retained; zero padding masked',
       'geometry':'fresh deterministic per-source affine +/-10deg, scale .92..1.08, translation +/-4 canonical pixels; photometric .8..1.2; one fixed pair/source; canonical144x256 coordinate frame',
       'supervision':'detector teacher KL T1 weight1 over fully valid cells + symmetric homography correspondence InfoNCE T.1 weight1; no descriptor KD',
       'initialization':'deterministic original-teacher L1 output-filter transfer conditioned on previous selected channels; same weights for same width at both resolutions; fresh AdamW',
       'sampling':'same seeded source plan, uniform among140 TRAIN images, two pairs/update; no scene balancing experiment',
       'optimizer':'AdamW lr1e-4 wd1e-4 cosine1e-5 clip10; 400 updates; accumulated mean over2 pairs',
       'resources':{'CPU_threads':2,'workers':0,'physical_batch_images':2,'accumulation_pairs':2,'AMP':False,'reason':'CPU FP32, zero training VRAM','teacher_resident_during_training':False,'minimum_available_RAM_MiB':1536},
       'features':'softmax, valid-content mask with4 canonical-pixel border, NMS radius round(3*scale), threshold.005, top160, original descriptor sampling, CPU L2',
       'metrics':'MNN precision and count; correct MNN; repeatability at3 canonical pixels; correct-match occupied4x4 content bins/16; mutual visible overlap only',
       'selection':'lexicographic equal-scene macro correct_mnn_count, mnn_precision, repeatability; earliest tie. Report domains separately.',
       'scope':'single deterministic initialization strategy, small synthetic-homography DEV screening; related project sequences; not independent pose quality, not full training or INT8 quality',
       'GT1_used':False,'MixVPR_modified':False,'code_sha256':sha(__file__)}
    f=ROOT/'protocol.json'
    if f.exists(): assert json.loads(f.read_text())==json.loads(json.dumps(p)),'Protocol changed'
    else: put(f,p)
    plan=np.random.default_rng(SEED).integers(0,len(chosen['train']),size=(STEPS,2)); np.save(ROOT/'plan.npy',plan)
    return p

def project(points,h):
    v=np.c_[points,np.ones(len(points))]@h.T
    return v[:,:2]/v[:,2:]

def prepare():
    p=protocol()
    if (ROOT/'cache_done.json').exists(): return
    teacher,_=w0.import_original_superpoint(); raw=SuperPointKPUWrapper(teacher).eval().requires_grad_(False)
    for name,width in WIDTHS.items():
        m=Model(width); cmap={}; prev=[0]
        for stage,co in enumerate(width,1):
            for suffix in 'ab':
                key=f'conv{stage}{suffix}'; prev=w0.transfer_conv(getattr(teacher.net,key),getattr(m,key),key,prev,co,cmap)
        for a,b,count in [('convPa','convPb',65),('convDa','convDb',256)]:
            head=w0.transfer_conv(getattr(teacher.net,a),getattr(m,a),a,prev,128,cmap)
            w0.transfer_conv(getattr(teacher.net,b),getattr(m,b),b,head,count,cmap)
        save(ROOT/f'{name}_init.pt',m.state_dict()); put(ROOT/f'{name}_channel_map.json',cmap)
    for res,(w,h,scale,left) in RES.items():
        folder=ROOT/res; folder.mkdir(exist_ok=True)
        for split,rows in p['rows'].items():
            dest=folder/split; dest.mkdir(exist_ok=True)
            cache=np.lib.format.open_memmap(dest/'teacher_logits.npy',mode='w+',dtype=np.float16,shape=(len(rows),2,65,h//8,w//8))
            for i,row in enumerate(rows):
                guard(); assert sha(row['path'])==row['sha256']
                image=cv2.imread(row['path'],0); ih,iw=image.shape; factor=min(288/iw,512/ih)
                nw,nh=round(iw*factor),round(ih*factor); x0,y0=(288-nw)//2,(512-nh)//2
                common=np.zeros((512,288),np.uint8); mask=np.zeros_like(common)
                common[y0:y0+nh,x0:x0+nw]=cv2.resize(image,(nw,nh),interpolation=cv2.INTER_AREA); mask[y0:y0+nh,x0:x0+nw]=1
                rng=np.random.default_rng(SEED+i+(10000 if split=='dev' else 0))
                hc=np.eye(3); hc[:2]=cv2.getRotationMatrix2D((72,128),rng.uniform(-10,10),rng.uniform(.92,1.08)); hc[:2,2]+=rng.uniform(-4,4,2)
                brightness=rng.uniform(.8,1.2); s=np.array([[scale,0,left],[0,scale,0],[0,0,1.]])
                tr=s@hc@np.linalg.inv(s)
                im=np.zeros((h,w),np.uint8); valid=np.zeros_like(im)
                im[:,left:left+round(144*scale)]=cv2.resize(common,(round(144*scale),h),interpolation=cv2.INTER_AREA)
                valid[:,left:left+round(144*scale)]=cv2.resize(mask,(round(144*scale),h),interpolation=cv2.INTER_NEAREST)
                warped=cv2.warpPerspective(im,tr,(w,h)); vm=cv2.warpPerspective(valid,tr,(w,h),flags=cv2.INTER_NEAREST)
                pair=np.stack([im,np.clip(warped.astype(float)*brightness,0,255).astype(np.uint8)])[:,None].astype(np.float32)/255
                # Correspondences originate at the common high-resolution canvas, shared across candidates.
                safe=cv2.erode(mask,np.ones((33,33),np.uint8))
                corners=cv2.goodFeaturesToTrack(common,100,.01,12,mask=safe)
                if corners is None: raise RuntimeError('No training corners: '+row['path'])
                pts=corners[:,0]/2; target=project(pts,hc); q=project(pts,s); tq=project(target,s)
                masks=np.stack([valid,vm]); eroded=np.stack([cv2.erode(v,np.ones((2*int(np.ceil(4*scale))+1,)*2,np.uint8),borderType=cv2.BORDER_CONSTANT,borderValue=0) for v in masks])
                keep=np.ones(len(q),bool)
                for xy,v in zip([q,tq],eroded):
                    ix=np.rint(xy).astype(int); inside=(ix[:,0]>=0)&(ix[:,0]<w)&(ix[:,1]>=0)&(ix[:,1]<h)
                    good=np.zeros(len(ix),bool); good[inside]=v[ix[inside,1],ix[inside,0]]>0; keep &= good
                if keep.sum()<8: raise RuntimeError('Too few valid correspondences')
                cell=masks.reshape(2,h//8,8,w//8,8).min((2,4)).astype(np.float32)
                np.savez_compressed(dest/f'{i:03d}.npz',pair=pair,points=np.stack([q[keep][:64],tq[keep][:64]]).astype(np.float32),homography=hc,masks=eroded,cell=cell,source_to_common=np.array([factor/2,x0/2,y0/2]),content_rect=np.array([x0/2,y0/2,nw/2,nh/2]))
                with torch.inference_mode(): a,b=raw(torch.from_numpy(pair)); cache[i]=a.numpy().astype(np.float16)
                if i%35==0: print('CACHE',res,split,i,flush=True)
            cache.flush(); del cache
        teacher_metrics=evaluate(raw,res,p['rows']['dev']); put(folder/'teacher_dev.json',teacher_metrics)
    del teacher,raw; gc.collect()
    files=sorted(f for f in ROOT.rglob('*') if f.is_file() and f.suffix in ('.npy','.npz','.pt','.json') and f.name!='cache_done.json')
    put(ROOT/'cache_done.json',{'protocol_sha256':sha(ROOT/'protocol.json'),'files':{str(f.relative_to(ROOT)):sha(f) for f in files}})

def features(a,b,masks,res):
    w,h,scale,left=RES[res]; score=a.softmax(1)[:,:64]; n,_,hh,ww=score.shape
    score=score.permute(0,2,3,1).reshape(n,hh,ww,8,8).permute(0,1,3,2,4).reshape(n,h,w)
    score=score*torch.from_numpy(masks); score=ops.simple_nms(score,round(3*scale)); out=[]
    for k in range(n):
        yx=torch.nonzero(score[k]>.005); conf=score[k][tuple(yx.t())]; yx,conf=ops.top_k_keypoints(yx,conf,160); xy=yx.flip([1]).float()
        desc=ops.sample_descriptors(xy[None],F.normalize(b[k:k+1],dim=1),8)[0].T.numpy()
        coords=xy.numpy(); coords=(coords-np.array([left,0]))/scale
        out.append((coords,desc))
    return out

def pair_metrics(f0,f1,h,rect):
    x,d=f0; y,e=f1
    zero={k:0. for k in ['correct_mnn_count','mnn_precision','mnn_count','repeatability','coverage']}
    if not len(x) or not len(y): return zero
    pred=project(x,h); distances=np.linalg.norm(pred[:,None]-y[None],axis=2)
    sim=d@e.T; forward=sim.argmax(1); reverse=sim.argmax(0); ii=np.flatnonzero(reverse[forward]==np.arange(len(x))); correct=ii[distances[ii,forward[ii]]<=3]
    coverage=0
    if len(correct):
        uv=(x[correct]-rect[:2])/rect[2:]; bins=np.clip((uv*4).astype(int),0,3); coverage=len(np.unique(bins,axis=0))/16
    return {'correct_mnn_count':float(len(correct)),'mnn_precision':len(correct)/max(1,len(ii)),'mnn_count':float(len(ii)),'repeatability':float((distances.min(1)<=3).mean()),'coverage':coverage}

@torch.inference_mode()
def evaluate(model,res,rows):
    was=model.training; model.eval(); records=[]; w,h,scale,left=RES[res]
    for i,row in enumerate(rows):
        with np.load(ROOT/res/'dev'/f'{i:03d}.npz') as z:
            a,b=model(torch.from_numpy(z['pair'])); masks=z['masks'].copy(); hc=z['homography']; s=np.array([[scale,0,left],[0,scale,0],[0,0,1.]])
            tr=s@hc@np.linalg.inv(s)
            # Both lists contain only keypoints whose corresponding location is visible in the other view.
            m0,m1=masks.copy(); masks[0] &= cv2.warpPerspective(m1,np.linalg.inv(tr),(w,h),flags=cv2.INTER_NEAREST); masks[1] &= cv2.warpPerspective(m0,tr,(w,h),flags=cv2.INTER_NEAREST)
            f=features(a,b,masks,res); m=pair_metrics(*f,hc,z['content_rect']); m['mean_keypoints']=float(np.mean([len(x[0]) for x in f]))
            records.append({'index':i,'scene':row['scene'],'domain':row['domain'],'metrics':m})
    def avg(rr): return {k:float(np.mean([r['metrics'][k] for r in rr])) for k in records[0]['metrics']}
    scenes={s:avg([r for r in records if r['scene']==s]) for s in sorted({r['scene'] for r in records})}
    macro={k:float(np.mean([v[k] for v in scenes.values()])) for k in records[0]['metrics']}
    model.train(was)
    return {'macro':macro,'per_scene':scenes,'per_domain':{d:avg([r for r in records if r['domain']==d]) for d in ['indoor','project']},'rows':records}

def key(m): return tuple(m['macro'][k] for k in ['correct_mnn_count','mnn_precision','repeatability'])
def train(name,res):
    p=protocol(); guard(); assert (ROOT/'cache_done.json').exists()
    dest=ROOT/(name+'_'+res+('_cw' if res=='r2' else '')); dest.mkdir(exist_ok=True)
    if (dest/'report.json').exists(): return
    init=ROOT/f'{name}_init.pt'; ph=sha(ROOT/'protocol.json'); torch.manual_seed(SEED)
    model=Model(WIDTHS[name]); model.load_state_dict(torch.load(init,weights_only=True))
    opt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-4); sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,STEPS,eta_min=1e-5)
    cache=np.load(ROOT/res/'train/teacher_logits.npy',mmap_mode='r'); plan=np.load(ROOT/'plan.npy')
    state={'step':0,'best_step':0,'evaluations':[],'losses':[]}
    def checkpoint(): save(dest/'latest.pt',{'model':model.state_dict(),'optimizer':opt.state_dict(),'scheduler':sched.state_dict(),'state':state,'protocol_sha256':ph})
    def ev(step):
        m=evaluate(model,res,p['rows']['dev']); state['evaluations'].append({'step':step,'metrics':m})
        if 'best' not in state or key(m)>key(state['best']):
            state.update(best=m,best_step=step); save(dest/'best.pt',{'model':model.state_dict(),'step':step,'metrics':m,'widths':WIDTHS[name],'resolution':res,'protocol_sha256':ph,'initialization_sha256':sha(init)})
        put(dest/'progress.json',state); print('DEV',dest.name,step,json.dumps(m['macro']),flush=True)
    if (dest/'latest.pt').exists():
        c=torch.load(dest/'latest.pt',weights_only=False); assert c['protocol_sha256']==ph
        model.load_state_dict(c['model']); opt.load_state_dict(c['optimizer']); sched.load_state_dict(c['scheduler']); state=c['state']
    else: ev(0); checkpoint()
    started=time.time()
    try:
        for step in range(state['step'],STEPS):
            if step%25==0: guard()
            opt.zero_grad(set_to_none=True); values=[]
            for i in plan[step]:
                with np.load(ROOT/res/'train'/f'{i:03d}.npz') as z:
                    a,b=model(torch.from_numpy(z['pair'])); xy=torch.from_numpy(z['points']); cell=torch.from_numpy(z['cell'])
                ta=torch.from_numpy(cache[i].astype(np.float32)); kl=F.kl_div(a.log_softmax(1),ta.softmax(1),reduction='none').sum(1); det=(kl*cell).sum()/cell.sum().clamp_min(1)
                d=ops.sample_descriptors(xy,F.normalize(b,dim=1),8).transpose(1,2); sim=d[0]@d[1].T/.1; target=torch.arange(len(sim)); geo=(F.cross_entropy(sim,target)+F.cross_entropy(sim.T,target))/2
                loss=(det+geo)/2
                if not torch.isfinite(loss): raise RuntimeError('Nonfinite loss')
                loss.backward(); values.append([det.item(),geo.item()])
            grad=torch.nn.utils.clip_grad_norm_(model.parameters(),10)
            if not torch.isfinite(grad): raise RuntimeError('Nonfinite gradient')
            opt.step(); sched.step(); state['step']=step+1; state['losses'].append({'step':step+1,'detector':float(np.mean(values,axis=0)[0]),'geometry':float(np.mean(values,axis=0)[1])})
            if (step+1)%25==0: checkpoint(); print('TRAIN',dest.name,step+1,flush=True)
            if (step+1)%100==0: ev(step+1); checkpoint()
    except BaseException: checkpoint(); raise
    put(dest/'report.json',{'status':'COMPLETE','state':state,'elapsed_s':time.time()-started,'peak_process_RSS_MiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,'protocol_sha256':ph,'plan_sha256':sha(ROOT/'plan.npy'),'init_sha256':sha(init),'checkpoint_sha256':sha(dest/'best.pt'),'INT8_quality':'NOT_EVALUATED'})

def verify():
    x=np.array([[20.,20.],[40.,30.],[80.,70.],[100.,150.]])
    h=np.array([[1.,0.,3.],[0.,1.,-2.],[0.,0.,1.]])
    m=pair_metrics((x,np.eye(4)),(project(x,h),np.eye(4)),h,np.array([0,0,144,256]))
    assert m['correct_mnn_count']==4 and m['mnn_precision']==1 and m['repeatability']==1
    for _,(_,_,scale,left) in RES.items():
        s=np.array([[scale,0,left],[0,scale,0],[0,0,1.]])
        assert np.allclose(project(project(x,s),s@h@np.linalg.inv(s)),project(project(x,h),s))
    assert pair_metrics((x[:0],np.eye(4)[:0]),(x,np.eye(4)),h,np.array([0,0,144,256]))['correct_mnn_count']==0
    print('METRIC_GEOMETRY_CHECKS_PASS',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('action',choices=['verify','prepare','train']); parser.add_argument('--width',choices=WIDTHS); parser.add_argument('--resolution',choices=RES); args=parser.parse_args()
    ROOT.mkdir(exist_ok=True); lock=(ROOT/'run.lock').open('a'); fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if args.action=='verify': verify()
    elif args.action=='prepare': verify(); prepare()
    else: train(args.width,args.resolution)
