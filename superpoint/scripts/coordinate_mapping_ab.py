"""Controlled 400-update descriptor coordinate A/B with common valid points."""
import _bootstrap
import json,time,subprocess,os,shutil
import numpy as np
import torch.nn.functional as F
from spk210.runtime import torch,ops
from spk210.settings import ROOT,SUPERPOINT,WORKSPACE,WIDTHS,SEED,STEPS
from spk210.protocol import protocol,verify_cache
from spk210.model import Model
from spk210.metrics import evaluate,key,project
from spk210.io import sha,put,save,guard
DEST=SUPERPOINT/'artifacts/coordinate_mapping_ab'
BRANCHES={'coord_current_r2_cw':False,'coord_center_r2_cw':True}


def setup():
    verify_cache(); DEST.mkdir(exist_ok=True)
    arrays={}; audit=[]; S=np.array([[1.25,0,2],[0,1.25,0],[0,0,1.]])
    for i,row in enumerate(protocol()['rows']['train']):
        f=ROOT/'r2/train'/f'{i:03d}.npz'
        with np.load(f) as z: old=z['points'].copy(); masks=z['masks'].copy(); H=S@z['homography']@np.linalg.inv(S)
        # INTER_AREA common288x512 ->180x320: (c+.5)*.625-.5+[2,0].
        # Cache used c*.625+[2,0]; actual second view warp is S Hc S^-1.
        np.testing.assert_allclose(project(old[0],H),old[1],atol=5e-5,rtol=0)
        new=old.astype(np.float64);new[0]-=.1875;new[1]=project(new[0],H)
        keep=np.ones(old.shape[1],bool)
        for pair in [old,new]:
            for xy,mask in zip(pair,masks):
                ij=np.rint(xy).astype(int);inside=(ij[:,0]>=0)&(ij[:,0]<184)&(ij[:,1]>=0)&(ij[:,1]<320)
                good=np.zeros(len(ij),bool);good[inside]=mask[ij[inside,1],ij[inside,0]]>0;keep &=good
        assert keep.sum()>=8
        arrays[f'{i}_0']=old[:,keep].copy();arrays[f'{i}_1']=new[:,keep].astype(np.float32)
        np.testing.assert_allclose(project(arrays[f'{i}_1'][0],H),arrays[f'{i}_1'][1],atol=5e-5,rtol=0)
        audit.append({'index':i,'input_sha256':sha(f),'excluded_indices':np.flatnonzero(~keep).tolist(),'retained':int(keep.sum())})
    assert sum(len(r['excluded_indices']) for r in audit)==4
    cache=DEST/'shared_points.npz'
    if cache.exists():
        with np.load(cache) as z:
            assert set(z.files)==set(arrays)
            for k,v in arrays.items():np.testing.assert_array_equal(z[k],v)
    else:np.savez_compressed(cache,**arrays)
    spec={'scope':'one-seed controlled coordinate mapping A/B, DEV only, no automatic promotion',
          'arms':BRANCHES,'mapping_A':'original cached TRAIN coordinates','mapping_B':'view0 minus.1875 each axis; view1 actual cached S Hc S^-1 warp of corrected view0',
          'shared_valid':'exclude union of invalid point indices from BOTH arms; identical identities and count',
          'audit':audit,'shared_points_sha256':sha(cache),'initial_sha256':sha(ROOT/'wider_init.pt'),'plan_sha256':sha(ROOT/'plan.npy'),
          'architecture':WIDTHS['wider'],'seed':SEED,'updates':STEPS,'data_protocol_sha256':sha(ROOT/'protocol.json'),
          'loss':'unchanged detector KL + symmetric descriptor InfoNCE tau.1, accumulated two pairs; all weights train; no KD',
          'optimizer':'AdamW lr1e-4 wd1e-4 cosine400 eta_min1e-5 clip10',
          'selection':'unchanged DEV lexicographic correctMNN precision repeatability; steps0,100,200,300,400 earliest tie',
          'evaluation':'unchanged descriptor DEV, teacher-fixed PnP, own-keypoint localization and actual nncase INT8; additional frozen improved corners diagnostic',
          'source_sha256':sha(__file__),'data_source_sha256':sha(SUPERPOINT/'src/spk210/data.py'),
          'evaluator_sha256':sha(SUPERPOINT/'scripts/descriptor_kd_evaluate.py')}
    path=DEST/'protocol.json'
    if path.exists():assert json.loads(path.read_text())==spec
    else:put(path,spec)
    shutil.copyfile(__file__,DEST/'evaluated_source.py')
    reg=SUPERPOINT/'configs/candidates.json'; entries=json.loads(reg.read_text())
    for name in BRANCHES:
        entry={'width':'wider','resolution':'r2','checkpoint':f'artifacts/coordinate_mapping_ab/{name}/best.pt','protocol':'artifacts/coordinate_mapping_ab/protocol.json'}
        assert name not in entries or entries[name]==entry;entries[name]=entry
    put(reg,entries)


def train(name,corrected):
    folder=DEST/name;folder.mkdir(exist_ok=True)
    if (folder/'report.json').exists():return
    torch.manual_seed(SEED);model=Model(WIDTHS['wider']);model.load_state_dict(torch.load(ROOT/'wider_init.pt',weights_only=True))
    opt=torch.optim.AdamW(model.parameters(),lr=.0001,weight_decay=.0001)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,STEPS,eta_min=1e-5)
    logits=np.load(ROOT/'r2/train/teacher_logits.npy',mmap_mode='r');plan=np.load(ROOT/'plan.npy')
    coordinates=np.load(DEST/'shared_points.npz');state={'step':0,'best_step':0,'evaluations':[],'losses':[]};ph=sha(DEST/'protocol.json')
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
                a,b=model(torch.from_numpy(z['pair']));xy=torch.from_numpy(coordinates[f'{int(i)}_{int(corrected)}']);cell=torch.from_numpy(z['cell'])
            ta=torch.from_numpy(logits[i].astype(np.float32))
            kl=F.kl_div(a.log_softmax(1),ta.softmax(1),reduction='none').sum(1);det=(kl*cell).sum()/cell.sum().clamp_min(1)
            d=ops.sample_descriptors(xy,F.normalize(b,dim=1),8).transpose(1,2)
            sim=d[0]@d[1].T/.1;target=torch.arange(len(sim));geo=(F.cross_entropy(sim,target)+F.cross_entropy(sim.T,target))/2
            loss=(det+geo)/2
            assert torch.isfinite(loss);loss.backward();values.append([det.item(),geo.item()])
        grad=torch.nn.utils.clip_grad_norm_(model.parameters(),10);assert torch.isfinite(grad)
        opt.step();sched.step();state['step']=step+1
        state['losses'].append(dict(step=step+1,**dict(zip(['detector','geometry'],np.mean(values,axis=0).tolist()))))
        if (step+1)%25==0:ckpt();print('TRAIN',name,step+1,flush=True)
        if (step+1)%100==0:ev(step+1);ckpt()
    put(folder/'report.json',{'status':'COMPLETE','best_step':state['best_step'],'updates':STEPS,'checkpoint_sha256':sha(folder/'best.pt'),
                             'initial_sha256':sha(ROOT/'wider_init.pt'),'plan_sha256':sha(ROOT/'plan.npy'),'shared_points_sha256':sha(DEST/'shared_points.npz'),'elapsed_s':time.time()-started})


def validation():
    env=dict(os.environ,SP_EXPERIMENT_ROOT=str(DEST))
    for name in BRANCHES:
        for stage in ['export','validate','calibrate','compile','simulate','evaluate','natural_simulate','natural_evaluate']:
            marker=DEST/f'{name}_{stage}.pass.json'
            if marker.exists():continue
            if stage.startswith('natural_'):
                action=stage.split('_',1)[1]
                python='/home/quyet/miniconda3/envs/k210/bin/python' if action=='simulate' else str(WORKSPACE/'.venv-mixvpr-cuda/bin/python')
                cmd=[python,str(SUPERPOINT/'scripts/descriptor_kd_evaluate.py'),action,'--candidate',name]
            else:cmd=[str(SUPERPOINT/'run'),stage,'--candidate',name]
            with (DEST/f'{name}_{stage}.log').open('w') as f:subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
            put(marker,{'status':'PASS','checkpoint_sha256':sha(DEST/name/'best.pt')});print('PASS',name,stage,flush=True)

if __name__=='__main__':
    setup()
    for name,corrected in BRANCHES.items():train(name,corrected)
    validation()
