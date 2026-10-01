"""One-seed 400-update descriptor KD A/B, preserving original pilot recipe."""
import _bootstrap
import json,time
import numpy as np
import torch.nn.functional as F
from spk210.runtime import torch,ops,w0,SuperPointKPUWrapper
from spk210.settings import ROOT,SUPERPOINT,WIDTHS,SEED,STEPS
from spk210.protocol import protocol,verify_cache
from spk210.model import Model
from spk210.metrics import evaluate,key
from spk210.io import sha,put,save,guard

DEST=SUPERPOINT/'artifacts/descriptor_kd_ab'
BRANCHES={'desc_control_r2_cw':0.,'desc_kd_r2_cw':1.}


def setup():
    verify_cache();DEST.mkdir(exist_ok=True)
    spec={'arms':BRANCHES,'initial_checkpoint':str(ROOT/'wider_init.pt'),'initial_sha256':sha(ROOT/'wider_init.pt'),
          'data_protocol_sha256':sha(ROOT/'protocol.json'),'plan_sha256':sha(ROOT/'plan.npy'),
          'architecture':[24,32,64,96],'seed':SEED,'updates':STEPS,'eval_steps':[0,100,200,300,400],
          'base_recipe':'original detector KL + symmetric correspondence InfoNCE tau.1, accumulated two pairs; ALL weights train',
          'kd':'mean(1-dot(normalized student descriptor, normalized frozen teacher descriptor)) over both views and exactly the same TRAIN correspondence locations as InfoNCE; lambda1.0; no search',
          'pair_backward':'(detector_KL + geometry_InfoNCE + lambda*KD)/2; control omits KD expression exactly',
          'optimizer':'AdamW lr1e-4 weight_decay1e-4; cosine400 eta_min1e-5; clip_norm10',
          'selection':'unchanged FP32 DEV lexicographic equal-scene correct MNN, precision, repeatability; earliest tie',
          'evaluation':'same42 synthetic DEV pairs, same natural-track diagnostic and20 covered localization queries; original shared NMS/settings; own-keypoint as well as fixed teacher-keypoint',
          'retention_decision':'retain only provisionally if descriptor DEV ranking improves, fixed and own-keypoint PnP do not regress vs A in FP32 and real INT8, and at least one own-keypoint PnP count improves; otherwise do not add to final recipe. No statistical significance or release claim.',
          'scope':'one seed/400updates, no full training, no QAT, no MixVPR changes, DEV only; TEST not established',
          'source_sha256':sha(__file__)}
    path=DEST/'protocol.json'
    if path.exists():assert json.loads(path.read_text())==spec
    else:put(path,spec)
    cachepath=DEST/'teacher_descriptors.npz'
    if not cachepath.exists():
        teacher,_=w0.import_original_superpoint();teacher=SuperPointKPUWrapper(teacher).eval()
        arrays={}
        with torch.inference_mode():
            for i in range(len(protocol()['rows']['train'])):
                guard()
                with np.load(ROOT/'r2/train'/f'{i:03d}.npz') as z:
                    _,b=teacher(torch.from_numpy(z['pair']));xy=torch.from_numpy(z['points'].copy())
                arrays[str(i)]=ops.sample_descriptors(xy,F.normalize(b,dim=1),8).transpose(1,2).numpy()
        np.savez_compressed(cachepath,**arrays)
        put(DEST/'teacher_cache.json',{'sha256':sha(cachepath),'teacher_sha256':sha(w0.CHECKPOINT),'split':'TRAIN_ONLY','count':len(arrays)})
    assert sha(cachepath)==json.loads((DEST/'teacher_cache.json').read_text())['sha256']
    reg=SUPERPOINT/'configs/candidates.json';entries=json.loads(reg.read_text())
    for name in BRANCHES:
        entry={'width':'wider','resolution':'r2','checkpoint':f'artifacts/descriptor_kd_ab/{name}/best.pt','protocol':'artifacts/descriptor_kd_ab/protocol.json'}
        assert name not in entries or entries[name]==entry
        entries[name]=entry
    put(reg,entries)


def train(name,weight):
    folder=DEST/name;folder.mkdir(exist_ok=True)
    if (folder/'report.json').exists():return
    torch.manual_seed(SEED);model=Model(WIDTHS['wider']);model.load_state_dict(torch.load(ROOT/'wider_init.pt',weights_only=True))
    opt=torch.optim.AdamW(model.parameters(),lr=.0001,weight_decay=.0001)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,STEPS,eta_min=1e-5)
    logits=np.load(ROOT/'r2/train/teacher_logits.npy',mmap_mode='r');plan=np.load(ROOT/'plan.npy')
    descriptors=np.load(DEST/'teacher_descriptors.npz');state={'step':0,'best_step':0,'evaluations':[],'losses':[]};ph=sha(DEST/'protocol.json')
    def ckpt():save(folder/'latest.pt',{'model':model.state_dict(),'optimizer':opt.state_dict(),'scheduler':sched.state_dict(),'state':state,'protocol_sha256':ph})
    def ev(step):
        m=evaluate(model,'r2',protocol()['rows']['dev']);state['evaluations'].append({'step':step,'metrics':m})
        if 'best' not in state or key(m)>key(state['best']):
            state.update(best=m,best_step=step)
            save(folder/'best.pt',{'model':model.state_dict(),'step':step,'metrics':m,'widths':WIDTHS['wider'],'resolution':'r2','protocol_sha256':ph,'initialization_sha256':sha(ROOT/'wider_init.pt')})
        put(folder/'progress.json',state);print('DEV',name,step,json.dumps(m['macro']),flush=True)
    if (folder/'latest.pt').exists():
        c=torch.load(folder/'latest.pt',weights_only=False);assert c['protocol_sha256']==ph
        model.load_state_dict(c['model']);opt.load_state_dict(c['optimizer']);sched.load_state_dict(c['scheduler']);state=c['state']
    else:ev(0);ckpt()
    started=time.time()
    for step in range(state['step'],STEPS):
        guard();opt.zero_grad(set_to_none=True);values=[]
        for i in plan[step]:
            with np.load(ROOT/'r2/train'/f'{i:03d}.npz') as z:
                a,b=model(torch.from_numpy(z['pair']));xy=torch.from_numpy(z['points']);cell=torch.from_numpy(z['cell'])
            ta=torch.from_numpy(logits[i].astype(np.float32))
            kl=F.kl_div(a.log_softmax(1),ta.softmax(1),reduction='none').sum(1);det=(kl*cell).sum()/cell.sum().clamp_min(1)
            d=ops.sample_descriptors(xy,F.normalize(b,dim=1),8).transpose(1,2)
            sim=d[0]@d[1].T/.1;target=torch.arange(len(sim));geo=(F.cross_entropy(sim,target)+F.cross_entropy(sim.T,target))/2
            kd=(1-(d*torch.from_numpy(descriptors[str(i)])).sum(-1)).mean()
            loss=(det+geo)/2 if weight==0 else (det+geo+weight*kd)/2
            assert torch.isfinite(loss);loss.backward();values.append([det.item(),geo.item(),kd.item()])
        grad=torch.nn.utils.clip_grad_norm_(model.parameters(),10);assert torch.isfinite(grad)
        opt.step();sched.step();state['step']=step+1
        state['losses'].append(dict(step=step+1,**dict(zip(['detector','geometry','descriptor_kd'],np.mean(values,axis=0).tolist()))))
        if (step+1)%25==0:ckpt();print('TRAIN',name,step+1,flush=True)
        if (step+1)%100==0:ev(step+1);ckpt()
    exact=None
    if weight==0:
        old=torch.load(ROOT/'wider_r2_cw/best.pt',weights_only=False);new=torch.load(folder/'best.pt',weights_only=False)
        exact=all(torch.equal(old['model'][k],v) for k,v in new['model'].items())
        assert exact,'Control no longer reproduces current recipe exactly'
    put(folder/'report.json',{'status':'COMPLETE','best_step':state['best_step'],'updates':STEPS,'checkpoint_sha256':sha(folder/'best.pt'),
                             'initial_sha256':sha(ROOT/'wider_init.pt'),'plan_sha256':sha(ROOT/'plan.npy'),'control_exact_original_weights':exact,'elapsed_s':time.time()-started})


if __name__=='__main__':
    setup()
    for n,w in BRANCHES.items():train(n,w)
