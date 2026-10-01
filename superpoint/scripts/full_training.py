"""Reproducible expanded-data SuperPoint training with a sealed sequence holdout."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import _bootstrap
import json,time,sys,copy
from pathlib import Path
import numpy as np
from spk210.runtime import torch,ops,w0,SuperPointKPUWrapper
import torch.nn.functional as F
from spk210.settings import SUPERPOINT,WORKSPACE,ROOT
from spk210.io import put,save,sha
from spk210.protocol import protocol
from spk210.metrics import features,project,evaluate,key
from spk210.model import Model
from spk210.onnx_export import load_model
from student_point_supervision_ab import anchor_loss
from full_training_data import make_pair
OUT=SUPERPOINT/'artifacts/full_training';SEED=20260919
NAME='full_capacity_r2_cw'

def prepare():
    OUT.mkdir(exist_ok=True)
    if (OUT/'protocol.json').exists():return json.loads((OUT/'protocol.json').read_text())
    probes=json.loads((SUPERPOINT/'artifacts/architecture_headroom_probe/summary.json').read_text())['records']
    valid=[r for r in probes if r.get('general_screen_pass') and r['widths'][0]==24]
    assert valid,'No graph with baseline early-stage KPU footprint and general reserve'
    selected=max(valid,key=lambda r:r['parameters']);rows=[];datasets=WORKSPACE/'datasets/7scenes';old=protocol()
    for scene in sorted(datasets.iterdir()):
        if not scene.is_dir():continue
        sequences=[int(x.strip().replace('sequence','')) for x in (scene/'TrainSplit.txt').read_text().splitlines() if x.strip()]
        files=sorted(f for n in sequences for f in (scene/f'seq-{n:02d}').glob('*.color.png'))
        assert files
        for j in np.linspace(0,len(files)-1,min(400,len(files)),dtype=int):
            f=files[j];rows.append({'path':str(f),'sha256':sha(f),'scene':scene.name,'sequence':f.parent.name,'domain':'indoor'})
    rows += [r for r in old['rows']['train'] if r['domain']=='project']
    dev=old['rows']['dev'];heldout=datasets/'redkitchen/seq-06';testfiles=sorted(heldout.glob('*.color.png'));assert len(testfiles)>=200
    # Sequence exclusion, then exact content checks. No TEST image decoding/inference.
    assert all(str(heldout) not in r['path'] for r in rows+dev+old['rows']['train'])
    test=[{'path':str(f),'sha256':sha(f),'scene':'redkitchen','sequence':'seq-06'} for f in testfiles]
    trainhash={r['sha256'] for r in rows};devhash={r['sha256'] for r in dev};testhash={r['sha256'] for r in test}
    assert not trainhash&devhash and not trainhash&testhash and not devhash&testhash
    p={'seed':SEED,'architecture':selected['widths'],'graph_probe':selected,'train':rows,'dev':dev,'test':test,
       'test_policy':'All redkitchen seq-06 reserved; paths/content hashes only. Never load TEST pixels or run TEST during training/selection. Historical pilot and targeted older training/protocol manifests contain no seq-06. Not a claim about unknown external teacher pretraining.',
       'updates_max':10000,'updates_min':2000,'pairs_per_update':2,'eval_every':500,'patience_evals':4,'patience_relative_improvement':.005,
       'stop_rule':'After >=2000 updates, stop after4 scheduled DEV evaluations without >=0.5% relative correct-MNN improvement; max10000 updates or4hours training; best checkpoint uses original lexicographic DEV key, earliest ties.',
       'max_train_seconds':14400,'optimizer':'AdamW lr1e-4 wd1e-4 cosine10000 eta_min1e-5 clip10; FP32, deterministic CUDA convolution; CPU descriptor grid_sample backward for deterministic sampling',
       'recipe':'retained detector KL + geometry InfoNCE + lambda1 student-point geometric InfoNCE; no rejected detector-ranking loss, cosine KD or QAT',
       'geometry':'retained common-image corner coordinate mapping; actual S H S^-1 warp; fresh seed-derived homography/brightness each draw',
       'anchor_detector_sha256':sha(ROOT/'wider_r2_cw/best.pt'),'anchor_policy':'same frozen original student, original NMS4 .005 cap160, first64 valid per direction; positive via actual warp, negatives >=8px in BOTH views, T=.1',
       'sampling':'deterministic shuffled TRAIN epochs, two sequential draws per update; no replacement within epoch; epoch seed20260919+epoch',
       'init':'teacher importance-selected channels; channels beyond teacher width repeat selected teacher channels with multiplicity-corrected incoming weights and1e-3 relative kernel noise on extra output channels to break symmetry',
       'quality_claim':'capacity selected by compiler screening, not proven strongest accuracy; full-system peak reserve/latency require later integration/board measurement',
       'calibration':'unchanged original TRAIN-only64 tensors for comparability',
       'sources':{str(f):sha(f) for f in [Path(__file__),Path(__file__).with_name('full_training_data.py')]}}
    put(OUT/'protocol.json',p);return p

def initialize(widths):
    teacher,_=w0.import_original_superpoint();m=Model(widths);mapping={};prev=[0]
    def transfer(src,dst,inputs,count,name):
        weight=src.weight.detach().cpu();restricted=weight[:,inputs];order=torch.argsort(restricted.abs().sum((1,2,3)),descending=True,stable=True).tolist()
        selected=list(range(len(weight))) if count>=len(weight) else order[:count]
        selected=(selected*((count+len(selected)-1)//len(selected)))[:count]
        multiplicity=np.bincount(inputs,minlength=weight.shape[1]);v=weight[selected][:,inputs].clone()/torch.tensor(multiplicity[inputs])[None,:,None,None]
        if count>len(weight):v[len(weight):]+=torch.randn_like(v[len(weight):])*v.std()*.001
        with torch.no_grad():dst.weight.copy_(v);dst.bias.copy_(src.bias.detach().cpu()[selected])
        mapping[name]={'input':inputs,'output':selected};return selected
    for stage,width in enumerate(widths,1):
        for suffix in 'ab':
            name=f'conv{stage}{suffix}';prev=transfer(getattr(teacher.net,name),getattr(m,name),prev,width,name)
    for a,b,co in [('convPa','convPb',65),('convDa','convDb',256)]:
        head=transfer(getattr(teacher.net,a),getattr(m,a),prev,128,a);transfer(getattr(teacher.net,b),getattr(m,b),head,co,b)
    put(OUT/'initial_channel_map.json',mapping);return m,SuperPointKPUWrapper(teacher).eval().requires_grad_(False)

@torch.inference_mode()
def anchors(a,b,z):
    cache={};fs=features(a.cpu(),b.cpu(),z['masks'],'r2',nms='original');scores=F.pixel_shuffle(a.cpu().softmax(1)[:,:64],8)[:,0].numpy()
    for view in [0,1]:
        xy=fs[view][0]*1.25+[2,0];ij=np.rint(xy).astype(int);xy=xy[np.lexsort((xy[:,0],xy[:,1],-scores[view,ij[:,1],ij[:,0]]))];other=project(xy,z['H'] if view==0 else np.linalg.inv(z['H']));end=[xy,other] if view==0 else [other,xy];keep=np.ones(len(xy),bool)
        for points,mask in zip(end,z['masks']):
            ij=np.rint(points).astype(int);inside=(ij[:,0]>=0)&(ij[:,0]<184)&(ij[:,1]>=0)&(ij[:,1]<320);good=np.zeros(len(points),bool);good[inside]=mask[ij[inside,1],ij[inside,0]]>0;keep &=good
        end=np.stack([v[keep][:64] for v in end]).astype(np.float32);dist=[np.linalg.norm(v[:,None]-v[None],axis=2) for v in end];allowed=(dist[0]>=8)&(dist[1]>=8);np.fill_diagonal(allowed,True);cache[f'0_{view}_xy']=end;cache[f'0_{view}_allowed']=allowed
        if len(end[0]):np.testing.assert_allclose(project(end[0],z['H']),end[1],atol=5e-5,rtol=0)
    return cache

def train():
    p=prepare();assert torch.cuda.is_available();torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED);torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    for f,h in p['sources'].items():assert sha(f)==h,'Training source changed after freeze'
    for r in p['train']+p['dev']:assert sha(r['path'])==r['sha256']
    m,teacher=initialize(p['architecture']);m=m.cuda();teacher=teacher.cuda();frozen,_,_=load_model('wider_r2_cw');frozen=frozen.cuda().eval().requires_grad_(False)
    opt=torch.optim.AdamW(m.parameters(),lr=.0001,weight_decay=.0001);sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,p['updates_max'],eta_min=1e-5);ph=sha(OUT/'protocol.json')
    state={'step':0,'evaluations':[],'losses':[],'stale':0,'patience_best':-1.,'elapsed_s':0.};start=time.time();base_elapsed=0.
    if (OUT/'latest.pt').exists():
        c=torch.load(OUT/'latest.pt',weights_only=False);assert c['protocol_sha256']==ph;m.load_state_dict(c['model']);opt.load_state_dict(c['optimizer']);sched.load_state_dict(c['scheduler']);state=c['state'];base_elapsed=state['elapsed_s']
    else:save(OUT/'initial.pt',{k:v.cpu() for k,v in m.state_dict().items()})
    def ckpt():
        state['elapsed_s']=base_elapsed+time.time()-start;save(OUT/'latest.pt',{'model':m.state_dict(),'optimizer':opt.state_dict(),'scheduler':sched.state_dict(),'state':state,'protocol_sha256':ph});put(OUT/'progress.json',state)
    def ev():
        cpu=copy.deepcopy(m).cpu();metric=evaluate(cpu,'r2',p['dev']);del cpu
        state['evaluations'].append({'step':state['step'],'metrics':metric})
        if 'best' not in state or key(metric)>key(state['best']):
            state['best']=metric;state['best_step']=state['step'];save(OUT/'best.pt',{'model':{k:v.cpu() for k,v in m.state_dict().items()},'step':state['step'],'metrics':metric,'widths':p['architecture'],'resolution':'r2','protocol_sha256':ph,'initialization_sha256':sha(OUT/'initial.pt')})
        score=metric['macro']['correct_mnn_count']
        if score>state['patience_best']*(1+p['patience_relative_improvement']):state.update(patience_best=score,stale=0)
        else:state['stale']+=1
        ckpt();print('DEV',state['step'],json.dumps(metric['macro']),flush=True)
    if not state['evaluations']:ev()
    n=len(p['train']);epoch=-1;order=None;reason='MAX_UPDATES'
    for step in range(state['step'],p['updates_max']):
        opt.zero_grad(set_to_none=True);totals=[]
        for j in range(2):
            draw=2*step+j;e=draw//n
            if e!=epoch:epoch=e;order=np.random.default_rng(SEED+epoch).permutation(n)
            z=make_pair(p['train'][int(order[draw%n])],draw);x=torch.from_numpy(z['pair']).cuda()
            with torch.no_grad():
                ta,_=teacher(x);fa,fb=frozen(x);cache=anchors(fa,fb,z)
                # Match retained FP16 teacher-cache target precision.
                ta=ta.half().float()
            a,b=m(x);cell=torch.from_numpy(z['cell']).cuda();det=(F.kl_div(a.log_softmax(1),ta.softmax(1),reduction='none').sum(1)*cell).sum()/cell.sum().clamp_min(1)
            head=b.cpu();xy=torch.from_numpy(z['points']);d=ops.sample_descriptors(xy,F.normalize(head,dim=1),8).transpose(1,2);sim=d[0]@d[1].T/.1;target=torch.arange(len(sim));geo=(F.cross_entropy(sim,target)+F.cross_entropy(sim.T,target))/2;anchor=anchor_loss(head,cache,0);loss=(det+(geo+anchor).cuda())/2
            assert torch.isfinite(loss);loss.backward();totals.append([float(det.detach()),float(geo.detach()),float(anchor.detach()),sum(cache[f'0_{v}_xy'].shape[1] for v in [0,1])])
        grad=torch.nn.utils.clip_grad_norm_(m.parameters(),10);assert torch.isfinite(grad);opt.step();sched.step();state['step']=step+1
        state['losses'].append({'step':step+1,'detector':float(np.mean(totals,axis=0)[0]),'geometry':float(np.mean(totals,axis=0)[1]),'anchor':float(np.mean(totals,axis=0)[2]),'anchors':float(np.mean(totals,axis=0)[3])})
        if (step+1)%100==0:ckpt();print('TRAIN',step+1,'seconds',round(state['elapsed_s']),'cuda_peak_MiB',round(torch.cuda.max_memory_allocated()/1024**2),flush=True)
        if (step+1)%p['eval_every']==0:
            ev()
            if step+1>=p['updates_min'] and state['stale']>=p['patience_evals']:reason='DEV_PATIENCE';break
        if base_elapsed+time.time()-start>=p['max_train_seconds']:reason='WALL_BUDGET';break
    if state['evaluations'][-1]['step']!=state['step']:ev()
    ckpt();put(OUT/'report.json',{'status':'TRAINING_COMPLETE','stop_reason':reason,'updates':state['step'],'equivalent_epochs':state['step']*2/n,'best_step':state['best_step'],'best_dev':state['best'],'checkpoint_sha256':sha(OUT/'best.pt'),'protocol_sha256':ph,'test_evaluated':False,'cuda_peak_bytes':torch.cuda.max_memory_allocated(),'elapsed_s':state['elapsed_s']})
    registry=SUPERPOINT/'configs/candidates.json';r=json.loads(registry.read_text());r[NAME]={'width':'wider','widths':p['architecture'],'resolution':'r2','checkpoint':'artifacts/full_training/best.pt','protocol':'artifacts/full_training/protocol.json'};put(registry,r)
    print('DONE',reason,state['step'],flush=True)
if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='prepare':print(json.dumps({k:v for k,v in prepare().items() if k not in ['train','dev','test']}))
    else:train()
