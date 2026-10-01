"""One400-update A/B: extra geometric descriptor supervision at frozen student detections."""
import _bootstrap
import json,time,shutil,subprocess,os
import numpy as np
import torch.nn.functional as F
from spk210.runtime import torch,ops
from spk210.settings import ROOT,SUPERPOINT,WORKSPACE,WIDTHS,SEED,STEPS
from spk210.protocol import protocol,verify_cache
from spk210.model import Model
from spk210.metrics import evaluate,key,features,project
from spk210.onnx_export import load_model
from spk210.io import sha,put,save,guard
DEST=SUPERPOINT/'artifacts/student_point_supervision_ab'
BRANCHES={'point_control_r2_cw':0.,'point_geometry_r2_cw':1.}


def anchor_loss(head,cache,index):
    terms=[]
    for view in [0,1]:
        xy=torch.from_numpy(cache[f'{index}_{view}_xy']);allowed=torch.from_numpy(cache[f'{index}_{view}_allowed'])
        if xy.shape[1]<2:continue
        d=ops.sample_descriptors(xy,F.normalize(head,dim=1),8).transpose(1,2)
        scores=(d[0]@d[1].T/.1).masked_fill(~allowed,float('-inf'));target=torch.arange(len(scores))
        usable=(allowed.sum(1)>1)
        if usable.any():terms.append((F.cross_entropy(scores[usable],target[usable])+F.cross_entropy(scores.T[usable],target[usable]))/2)
    return torch.stack(terms).mean() if terms else head.sum()*0


def setup():
    verify_cache();DEST.mkdir(exist_ok=True)
    cfg=json.loads((SUPERPOINT/'configs/localization_map.json').read_text());assert cfg['map_id']=='floor6_v1'
    contract={'arms':BRANCHES,'architecture':WIDTHS['wider'],'initial_sha256':sha(ROOT/'wider_init.pt'),'plan_sha256':sha(ROOT/'plan.npy'),'data_protocol_sha256':sha(ROOT/'protocol.json'),
      'seed':SEED,'updates':STEPS,'anchor_detector_sha256':sha(ROOT/'wider_r2_cw/best.pt'),
      'A':'unchanged current recipe detectorKL+original geometryInfoNCE; original correspondence cache including its existing mapping unchanged',
      'B':'same recipe plus lambda1.0 symmetric InfoNCE at frozen current-student TRAIN detections; no teacher descriptor KD',
      'positive':'each student keypoint in each cached TRAIN view -> actual S H S^-1 warp or inverse; sample other-view descriptor at exact projected float coordinate; same eroded masks both endpoints; max64anchors per direction',
      'negative':'offdiagonal anchors only when Euclidean endpoint distances >=8network pixels in BOTH views; diagonal positive always allowed; no cross-image mining, no hardest-negative search; exclude anchors with zero eligible negatives from loss',
      'anchor_selection':'current default student frozen, original NMS4/threshold.005/cap160, sorted score descending then y/x; retain first64 geometrically valid per source direction; source views0 and1; no detector gradients through coordinates',
      'normalization':'stride8 original sampler, normalized descriptors,temperature.1; symmetric row/column CE averaged over two anchor sets; total per-pair(det+old_geo+extra)/2; two pairs accumulated',
      'optimizer':'AdamW lr1e-4 wd1e-4 cosine400 eta_min1e-5 clip10; all weights train',
      'selection':'unchanged synthetic DEV lexicographic correctMNN precision repeatability;0,100,200,300,400; earliest tie',
      'evaluation':'unchanged42syntheticDEV and frozenfloor6 teacher/own PnP; fresh donor at each arm FP32 keypoints; additional old-student fixed coordinates shared across arms; floor6 descriptor top1 on query/first frozen reference,512ref/256shared IDs',
      'keep_B':'provisionally retain only if floor6 descriptor top1 at FP32 improves, own-FP32 pose count improves, own-INT8 pose count does not regress, and fixed-old-student-keypoint donor PnP does not regress in FP32 or INT8; otherwise do not retain. Single seed DEV only.',
      'floor6_teacher_protocol_sha256':cfg['teacher_baseline_protocol_sha256'],'calibration':'same original TRAIN-only64 calibration tensors; no floor6 fitting','source_sha256':sha(__file__)}
    if (DEST/'protocol.json').exists():assert json.loads((DEST/'protocol.json').read_text())==contract
    else:put(DEST/'protocol.json',contract)
    shutil.copyfile(__file__,DEST/'evaluated_source.py')
    path=DEST/'student_anchors.npz'
    if not path.exists():
        model,_,_=load_model('wider_r2_cw');arrays={};audit=[];S=np.array([[1.25,0,2],[0,1.25,0],[0,0,1.]])
        with torch.inference_mode():
            for i,row in enumerate(protocol()['rows']['train']):
                with np.load(ROOT/'r2/train'/f'{i:03d}.npz') as z:pair=z['pair'].copy();masks=z['masks'].copy();H=S@z['homography']@np.linalg.inv(S)
                a,b=model(torch.from_numpy(pair));fs=features(a,b,masks,'r2');score=a.softmax(1)[:,:64];score=F.pixel_shuffle(score,8)[:,0].numpy()
                for view in [0,1]:
                    xy=fs[view][0]*1.25+[2,0];ij=np.rint(xy).astype(int);order=np.lexsort((xy[:,0],xy[:,1],-score[view,ij[:,1],ij[:,0]]));xy=xy[order]
                    other=project(xy,H if view==0 else np.linalg.inv(H));endpoints=[xy,other] if view==0 else [other,xy];keep=np.ones(len(xy),bool)
                    for v,mask in zip(endpoints,masks):
                        ij=np.rint(v).astype(int);inside=(ij[:,0]>=0)&(ij[:,0]<184)&(ij[:,1]>=0)&(ij[:,1]<320);valid=np.zeros(len(v),bool);valid[inside]=mask[ij[inside,1],ij[inside,0]]>0;keep &=valid
                    endpoints=np.stack([v[keep][:64] for v in endpoints]).astype(np.float32)
                    np.testing.assert_allclose(project(endpoints[0],H),endpoints[1],atol=5e-5,rtol=0)
                    dist=[np.linalg.norm(v[:,None]-v[None],axis=2) for v in endpoints];allowed=(dist[0]>=8)&(dist[1]>=8);np.fill_diagonal(allowed,True)
                    assert np.array_equal(allowed,allowed.T)
                    arrays[f'{i}_{view}_xy']=endpoints;arrays[f'{i}_{view}_allowed']=allowed
                    audit.append({'pair':i,'source_view':view,'anchors':endpoints.shape[1],'anchors_with_negatives':int((allowed.sum(1)>1).sum()),'min_negatives':int((allowed.sum(1)-1).min()) if len(allowed) else 0})
        assert all(r['anchors_with_negatives']>=8 for r in audit),'Insufficient TRAIN anchors; do not train before diagnosis'
        np.savez_compressed(path,**arrays);put(DEST/'anchor_audit.json',{'rows':audit,'sha256':sha(path),'train_pairs':140})
    assert sha(path)==json.loads((DEST/'anchor_audit.json').read_text())['sha256']
    reg=SUPERPOINT/'configs/candidates.json';entries=json.loads(reg.read_text())
    for name in BRANCHES:
        entry={'width':'wider','resolution':'r2','checkpoint':f'artifacts/student_point_supervision_ab/{name}/best.pt','protocol':'artifacts/student_point_supervision_ab/protocol.json'}
        assert name not in entries or entries[name]==entry;entries[name]=entry
    put(reg,entries)


def train(name,weight):
    folder=DEST/name;folder.mkdir(exist_ok=True)
    if (folder/'report.json').exists():return
    torch.manual_seed(SEED);model=Model(WIDTHS['wider']);model.load_state_dict(torch.load(ROOT/'wider_init.pt',weights_only=True))
    opt=torch.optim.AdamW(model.parameters(),lr=.0001,weight_decay=.0001)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,STEPS,eta_min=1e-5)
    logits=np.load(ROOT/'r2/train/teacher_logits.npy',mmap_mode='r');plan=np.load(ROOT/'plan.npy')
    anchors=np.load(DEST/'student_anchors.npz');state={'step':0,'best_step':0,'evaluations':[],'losses':[]};ph=sha(DEST/'protocol.json')
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
            kd=anchor_loss(b,anchors,int(i)) if weight else torch.zeros(())
            loss=(det+geo)/2 if weight==0 else (det+geo+weight*kd)/2
            assert torch.isfinite(loss);loss.backward();values.append([det.item(),geo.item(),kd.item()])
        grad=torch.nn.utils.clip_grad_norm_(model.parameters(),10);assert torch.isfinite(grad)
        opt.step();sched.step();state['step']=step+1
        state['losses'].append(dict(step=step+1,**dict(zip(['detector','geometry','student_anchor_geometry'],np.mean(values,axis=0).tolist()))))
        if (step+1)%25==0:ckpt();print('TRAIN',name,step+1,flush=True)
        if (step+1)%100==0:ev(step+1);ckpt()
    exact=None
    if weight==0:
        old=torch.load(ROOT/'wider_r2_cw/best.pt',weights_only=False);new=torch.load(folder/'best.pt',weights_only=False)
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
