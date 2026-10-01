"""One detector-ranking A/B, keeping descriptor recipe B and all other controls."""
import _bootstrap
import json,time,shutil,subprocess,os
import numpy as np
import torch.nn.functional as F
from spk210.runtime import torch,ops
from spk210.settings import ROOT,SUPERPOINT,WORKSPACE,WIDTHS,SEED,STEPS
from spk210.protocol import protocol,verify_cache
from spk210.model import Model
from spk210.metrics import evaluate,key
from spk210.onnx_export import load_model
from spk210.postprocess import greedy_keypoints,default_nms
from spk210.io import sha,put,save,guard
from student_point_supervision_ab import anchor_loss
PREV=SUPERPOINT/'artifacts/student_point_supervision_ab'
DEST=SUPERPOINT/'artifacts/detector_ranking_ab'
BRANCHES={'rank_control_r2_cw':0.,'rank_pairwise_r2_cw':1.}


def ranking_loss(a,cache,index):
    logp=F.pixel_shuffle(a.log_softmax(1)[:,:64],8)[:,0];terms=[]
    for view in [0,1]:
        pairs=cache[f'{index}_{view}']
        if len(pairs):
            p,n=pairs[:,:2],pairs[:,2:]
            terms.append(F.softplus(np.log(2.)+logp[view,n[:,1],n[:,0]]-logp[view,p[:,1],p[:,0]]).mean())
    return torch.stack(terms).mean() if terms else a.sum()*0


def setup():
    verify_cache();assert default_nms()=='original';DEST.mkdir(exist_ok=True)
    anchor=PREV/'student_anchors.npz';assert sha(anchor)==json.loads((PREV/'anchor_audit.json').read_text())['sha256']
    cfg=json.loads((SUPERPOINT/'configs/localization_map.json').read_text());assert cfg['map_id']=='floor6_v1'
    spec={'arms':BRANCHES,'scope':'one400-update detector-ranking pilot; descriptor recipe B retained; no default promotion',
      'architecture':WIDTHS['wider'],'initial_sha256':sha(ROOT/'wider_init.pt'),'plan_sha256':sha(ROOT/'plan.npy'),'data_sha256':sha(ROOT/'protocol.json'),'seed':SEED,'updates':STEPS,
      'base_recipe':'unchanged uniform detector KL + original geometric InfoNCE + student-point geometric descriptor loss from retained recipe B; all weights train',
      'descriptor_anchor_sha256':sha(anchor),'A':'base recipe only; must reproduce point_geometry_r2_cw weights exactly','B':'base recipe plus detector pairwise ranking loss weight1; no descriptor changes',
      'candidate_pool':'frozen point_geometry_r2_cw on TRAIN pairs; greedy radius4 threshold.001 max320points per view; same pool throughout training, never floor6 evaluation images',
      'ranking_targets':'cached teacher logits decoded to scores; up to32 highest teacher-score candidates >=.005 as positives, up to64 lowest-score candidates as negatives; retain pairs with teacher-positive >=4x teacher-negative and distance>=8network pixels; no targets from map geometry',
      'ranking_loss':'mean softplus(log2 + student_log_probability_negative - student_log_probability_positive), averaged over eligible views; empty-view sets skipped and reported; pair loss(det+old_geo+descriptor_anchor+ranking)/2',
      'optimizer':'unchanged AdamW1e-4 wd1e-4 cosine400 min1e-5 clip10, two pairs accumulated','selection':'unchanged original-NMS synthetic42DEV evaluation at0,100,200,300,400; original lexicographic key earliest tie',
      'evaluation':'same42DEV, floor6 original and greedy NMS FP32/faithfulINT8; original reference/map/matcher/PnP; descriptor known-track ranking unchanged; additional teacher donor at candidate points',
      'calibration':'original frozen TRAIN-only64 tensors, same nncase options',
      'keep_B':'provisional only if greedy floor6 INT8 own pose strictly improves vs A, greedy FP32 pose does not regress, teacher donor on greedy INT8 keypoints improves, and floor6 descriptor top1 FP32/INT8 drops<=1percentage point; otherwise retain prior recipe. No posthoc relaxation.',
      'teacher_logits_sha256':sha(ROOT/'r2/train/teacher_logits.npy'),'frozen_detector_sha256':sha(PREV/'point_geometry_r2_cw/best.pt'),'source_sha256':sha(__file__)}
    if (DEST/'protocol.json').exists():assert json.loads((DEST/'protocol.json').read_text())==spec
    else:put(DEST/'protocol.json',spec)
    shutil.copyfile(__file__,DEST/'evaluated_source.py');path=DEST/'ranking_pairs.npz'
    if not path.exists():
        model,_,_=load_model('point_geometry_r2_cw');tl=np.load(ROOT/'r2/train/teacher_logits.npy',mmap_mode='r');arrays={};rows=[]
        with torch.inference_mode():
            for i in range(len(protocol()['rows']['train'])):
                with np.load(ROOT/'r2/train'/f'{i:03d}.npz') as z:a,_=model(torch.from_numpy(z['pair'].copy()));masks=z['masks'].copy()
                score=F.pixel_shuffle(a.softmax(1)[:,:64],8)[:,0];teacher=F.pixel_shuffle(torch.from_numpy(tl[i].astype(np.float32)).softmax(1)[:,:64],8)[:,0].numpy()
                for view in [0,1]:
                    xy=greedy_keypoints(score[view]*torch.from_numpy(masks[view]),4,.001,320).astype(int);values=teacher[view,xy[:,1],xy[:,0]];order=np.lexsort((xy[:,0],xy[:,1],-values));pos=[j for j in order if values[j]>=.005][:32];neg=order[::-1][:64];pairs=[]
                    for pi in pos:
                        for ni in neg:
                            if values[pi]>=4*values[ni] and np.linalg.norm(xy[pi]-xy[ni])>=8:pairs.append([*xy[pi],*xy[ni]])
                    arrays[f'{i}_{view}']=np.array(pairs,np.int64).reshape(-1,4);rows.append({'pair':i,'view':view,'candidates':len(xy),'positives':len(pos),'ranking_pairs':len(pairs)})
        assert sum(r['ranking_pairs']>0 for r in rows)>=len(rows)*.9,'Insufficient valid ranking supervision; stop before training'
        np.savez_compressed(path,**arrays);put(DEST/'ranking_audit.json',{'rows':rows,'sha256':sha(path),'eligible_views':sum(r['ranking_pairs']>0 for r in rows),'views':len(rows)})
    assert sha(path)==json.loads((DEST/'ranking_audit.json').read_text())['sha256']
    reg=SUPERPOINT/'configs/candidates.json';entries=json.loads(reg.read_text())
    for name in BRANCHES:
        entry={'width':'wider','resolution':'r2','checkpoint':f'artifacts/detector_ranking_ab/{name}/best.pt','protocol':'artifacts/detector_ranking_ab/protocol.json'}
        assert name not in entries or entries[name]==entry;entries[name]=entry
    put(reg,entries)


def train(name,weight):
    folder=DEST/name;folder.mkdir(exist_ok=True)
    if (folder/'report.json').exists():return
    torch.manual_seed(SEED);model=Model(WIDTHS['wider']);model.load_state_dict(torch.load(ROOT/'wider_init.pt',weights_only=True))
    opt=torch.optim.AdamW(model.parameters(),lr=.0001,weight_decay=.0001)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,STEPS,eta_min=1e-5)
    logits=np.load(ROOT/'r2/train/teacher_logits.npy',mmap_mode='r');plan=np.load(ROOT/'plan.npy')
    anchors=np.load(PREV/'student_anchors.npz');ranking=np.load(DEST/'ranking_pairs.npz');state={'step':0,'best_step':0,'evaluations':[],'losses':[]};ph=sha(DEST/'protocol.json')
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
            kd=anchor_loss(b,anchors,int(i));rank=ranking_loss(a,ranking,int(i)) if weight else torch.zeros(())
            loss=(det+geo+kd)/2 if weight==0 else (det+geo+kd+weight*rank)/2
            assert torch.isfinite(loss);loss.backward();values.append([det.item(),geo.item(),kd.item(),rank.item()])
        grad=torch.nn.utils.clip_grad_norm_(model.parameters(),10);assert torch.isfinite(grad)
        opt.step();sched.step();state['step']=step+1
        state['losses'].append(dict(step=step+1,**dict(zip(['detector','geometry','student_anchor_geometry','detector_ranking'],np.mean(values,axis=0).tolist()))))
        if (step+1)%25==0:ckpt();print('TRAIN',name,step+1,flush=True)
        if (step+1)%100==0:ev(step+1);ckpt()
    exact=None
    if weight==0:
        old=torch.load(PREV/'point_geometry_r2_cw/best.pt',weights_only=False);new=torch.load(folder/'best.pt',weights_only=False)
        exact=all(torch.equal(old['model'][k],v) for k,v in new['model'].items())
        assert exact,'Control no longer reproduces current recipe exactly'
    put(folder/'report.json',{'status':'COMPLETE','best_step':state['best_step'],'updates':STEPS,'checkpoint_sha256':sha(folder/'best.pt'),
                             'initial_sha256':sha(ROOT/'wider_init.pt'),'plan_sha256':sha(ROOT/'plan.npy'),'control_exact_original_weights':exact,'elapsed_s':time.time()-started})

def validate():
    for name in BRANCHES:
        for stage in ['export','validate','calibrate','compile','simulate','evaluate','floor6_simulate','floor6_evaluate']:
            marker=DEST/f'{name}_{stage}.pass.json'
            if marker.exists():assert json.loads(marker.read_text())['checkpoint_sha256']==sha(DEST/name/'best.pt');continue
            env=dict(os.environ,SP_STUDENT_CANDIDATE=name)
            if stage.startswith('floor6_'):
                action=stage.split('_',1)[1];python='/home/quyet/miniconda3/envs/k210/bin/python' if action=='simulate' else str(WORKSPACE/'.venv-mixvpr-cuda/bin/python')
                cmd=[python,str(SUPERPOINT/'scripts/floor6_student_evaluate.py'),action,'--output',str(DEST/name/'floor6')]
            else:cmd=[str(SUPERPOINT/'run'),stage,'--candidate',name]
            with (DEST/f'{name}_{stage}.log').open('w') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True,env=env)
            put(marker,{'status':'PASS','checkpoint_sha256':sha(DEST/name/'best.pt')});print('PASS',name,stage,flush=True)

if __name__=='__main__':
    setup()
    for name,weight in BRANCHES.items():train(name,weight)
    validate()
