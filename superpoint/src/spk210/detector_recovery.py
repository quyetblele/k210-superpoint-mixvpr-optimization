"""Controlled detector-only continuation; no change to backbone, descriptor or graph."""
import json,time,resource
import numpy as np
from .runtime import torch
import torch.nn.functional as F
from .settings import ROOT,SUPERPOINT,WIDTHS
from .protocol import protocol,verify_cache
from .io import sha,put,save,guard
from .model import Model
from .metrics import evaluate,key

DEST=SUPERPOINT/'artifacts/detector_recovery'
PARENT=ROOT/'wider_r2_cw/best.pt'
BRANCHES={'detector_control_r2_cw':'uniform_KL','detector_balanced_r2_cw':'foreground_balanced_KL'}
SEED=20260920
STEPS=400

def setup():
    DEST.mkdir(exist_ok=True)
    plan=np.random.default_rng(SEED).integers(0,140,size=(STEPS,2))
    p={'purpose':'Isolate detector supervision while preserving the quantization-compatible descriptor path',
       'parent':str(PARENT),'parent_sha256':sha(PARENT),'data_protocol_sha256':sha(ROOT/'protocol.json'),
       'architecture':'unchanged wider [24,32,64,96], hidden128,65+256 raw heads,logical184x320,CW90 deployment',
       'trainable':['convPa.weight','convPa.bias','convPb.weight','convPb.bias'],
       'branches':BRANCHES,'seed':SEED,'updates_each':STEPS,'eval_steps':[0,100,200,300,400],
       'balanced_formula':'0.5 uniform valid-cell KL +0.5 valid-cell KL weighted by teacher foreground probability (1-p_dustbin), normalized per pair',
       'control_formula':'uniform valid-cell KL; same loss scale, all other parameters frozen',
       'optimizer':'fresh AdamW1e-4 wd1e-4 cosine1e-5; clip10; physical2images,accumulation2pairs',
       'selection':'FP32 DEV lexicographic macro correct matches,precision,repeatability; earliest tie; PTQ only selected checkpoint each branch',
       'resource_policy':'CPU2threads,workers0,disk teacher cache,no GPU; RAM guard1536MiB',
       'plan_sha256':__import__('hashlib').sha256(plan.tobytes()).hexdigest(),
       'interpretation_gate':'experimental INT8 correct matches >= control*1.10 AND FP32 correct >= control*0.95 AND INT8 precision>=control-0.02; not a release criterion',
       'GT1_used':False,'QAT':False,'scope':'One-seed controlled detector-only pilot; no claim that foreground weighting solves quantization',
       'code_sha256':sha(__file__)}
    file=DEST/'protocol.json'
    if file.exists():assert json.loads(file.read_text())==p,'Locked protocol changed'
    else:put(file,p)
    np.save(DEST/'plan.npy',plan)
    registry=SUPERPOINT/'configs/candidates.json'
    entries=json.loads(registry.read_text()) if registry.exists() else {}
    for name in BRANCHES:
        entry={'width':'wider','resolution':'r2','checkpoint':f'artifacts/detector_recovery/{name}/best.pt','protocol':'artifacts/detector_recovery/protocol.json'}
        if name in entries:assert entries[name]==entry
        entries[name]=entry
    put(registry,entries)
    return p

def detector_loss(logits,teacher,mask,balanced):
    probability=teacher.softmax(1)
    kl=F.kl_div(logits.log_softmax(1),probability,reduction='none').sum(1)
    uniform=(kl*mask).sum()/mask.sum().clamp_min(1)
    if not balanced:return uniform
    foreground=(1-probability[:,64])*mask
    focused=(kl*foreground).sum()/foreground.sum().clamp_min(1e-8)
    return .5*(uniform+focused)

def train(branch):
    assert branch in BRANCHES
    verify_cache();p=setup();guard()
    folder=DEST/branch;folder.mkdir(exist_ok=True)
    if (folder/'report.json').exists():return
    torch.manual_seed(SEED)
    parent=torch.load(PARENT,map_location='cpu',weights_only=False)
    model=Model(WIDTHS['wider']);model.load_state_dict(parent['model'])
    for name,param in model.named_parameters():param.requires_grad_(name in p['trainable'])
    trainable=[v for v in model.parameters() if v.requires_grad]
    opt=torch.optim.AdamW(trainable,lr=1e-4,weight_decay=1e-4)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,STEPS,eta_min=1e-5)
    cache=np.load(ROOT/'r2/train/teacher_logits.npy',mmap_mode='r');plan=np.load(DEST/'plan.npy')
    state={'step':0,'best_step':0,'evaluations':[],'losses':[]};ph=sha(DEST/'protocol.json')
    def checkpoint():save(folder/'latest.pt',{'model':model.state_dict(),'optimizer':opt.state_dict(),'scheduler':scheduler.state_dict(),'state':state,'protocol_sha256':ph})
    def ev(step):
        m=evaluate(model,'r2',protocol()['rows']['dev']);state['evaluations'].append({'step':step,'metrics':m})
        if 'best' not in state or key(m)>key(state['best']):
            state.update(best=m,best_step=step)
            save(folder/'best.pt',{'model':model.state_dict(),'step':step,'metrics':m,'protocol_sha256':ph,'parent_sha256':p['parent_sha256'],'widths':WIDTHS['wider'],'resolution':'r2'})
        put(folder/'progress.json',state);print('DEV',branch,step,json.dumps(m['macro']),flush=True)
    if (folder/'latest.pt').exists():
        c=torch.load(folder/'latest.pt',weights_only=False);assert c['protocol_sha256']==ph
        model.load_state_dict(c['model']);opt.load_state_dict(c['optimizer']);scheduler.load_state_dict(c['scheduler']);state=c['state']
    else:ev(0);checkpoint()
    started=time.time()
    try:
        for step in range(state['step'],STEPS):
            if step%25==0:guard()
            model.train();opt.zero_grad(set_to_none=True);values=[]
            for i in plan[step]:
                with np.load(ROOT/'r2/train'/f'{i:03d}.npz') as z:
                    a,_=model(torch.from_numpy(z['pair']));mask=torch.from_numpy(z['cell'])
                teacher=torch.from_numpy(cache[i].astype(np.float32))
                loss=detector_loss(a,teacher,mask,BRANCHES[branch]=='foreground_balanced_KL')
                if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
                (loss/2).backward();values.append(float(loss.detach()))
            grad=torch.nn.utils.clip_grad_norm_(trainable,10)
            if not torch.isfinite(grad):raise RuntimeError('Nonfinite gradient')
            opt.step();scheduler.step();state['step']=step+1;state['losses'].append({'step':step+1,'loss':float(np.mean(values))})
            if (step+1)%25==0:checkpoint();print('TRAIN',branch,step+1,flush=True)
            if (step+1)%100==0:ev(step+1);checkpoint()
    except BaseException:checkpoint();raise
    frozen=[name for name in parent['model'] if name not in p['trainable']]
    assert all(torch.equal(model.state_dict()[name],parent['model'][name]) for name in frozen)
    put(folder/'report.json',{'status':'COMPLETE','state':state,'elapsed_s':time.time()-started,'peak_RSS_MiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,'frozen_parameters_exact':True,'checkpoint_sha256':sha(folder/'best.pt'),'parent_sha256':p['parent_sha256'],'plan_sha256':sha(DEST/'plan.npy'),'protocol_sha256':ph})
