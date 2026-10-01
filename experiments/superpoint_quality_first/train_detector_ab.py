"""Controlled detector-supervision replacement; all other training fixed."""
from diagnose import *
import gc,random,fcntl
DEST=OUT/'detector_ab';SEED=20260914;STEPS=400

def put(path,x):
    t=Path(str(path)+'.tmp');t.write_text(json.dumps(x,indent=2)+'\n');t.replace(path)
def checkpoint(path,x):
    t=Path(str(path)+'.tmp');torch.save(x,t);t.replace(path)
def free_memory_mib():
    data=Path('/proc/meminfo').read_text().splitlines()
    return int(next(s for s in data if s.startswith('MemAvailable:')).split()[1])/1024

def run():
    DEST.mkdir(exist_ok=True)
    lock=(DEST/'run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    source=BASE/'SUP/best.pt';source_hash=digest(source)
    old=json.loads((BASE/'protocol.json').read_text());teacher_cache=BASE/'train/teacher_logits.npy'
    plan=np.random.default_rng(SEED).integers(0,len(old['TRAIN']),size=(STEPS,2))
    prot={'purpose':'diagnostic A/B: replace Shi-Tomasi detector CE with official-teacher detector KL; descriptor geometry objective unchanged',
      'branches':{'CONTINUE_SUP':'weighted detector CE + geometry InfoNCE','TEACHER_DETECTOR':'detector KL(T=1, coefficient1) + same geometry InfoNCE; no detector CE, no descriptor KD'},
      'initialization':str(source),'init_sha256':source_hash,'architecture':'existing V1FR12 [12,24,32,64], head128, detector65 descriptor256, input256x192',
      'seed':SEED,'updates':STEPS,'eval_every':100,'physical_batch_images':2,'accumulation_pairs':2,'CPU_threads':2,'workers':0,'AMP':False,'reason_no_AMP':'CPU FP32 training, no GPU/VRAM use',
      'optimizer':'fresh AdamW lr1e-4 wd1e-4; cosine1e-5; grad clip10',
      'plan_sha256':hashlib.sha256(plan.tobytes()).hexdigest(),'source_protocol_sha256':digest(BASE/'protocol.json'),'teacher_logits_sha256':digest(teacher_cache),'code_sha256':digest(__file__),
      'selection':'best DEV lexicographic precision, repeatability3, MNN count; earliest tie',
      'gate':'TEACHER_DETECTOR best precision >= CONTINUE_SUP best +0.03; repeatability >= control-0.02; absolute precision>=0.50',
      'GT1_used':False,'scope':'one-seed short warm-start recipe diagnosis, not publication ablation or final model selection',
      'minimum_host_available_MiB':1536,'checkpoints':'best and latest only; atomic writes; no old artifacts removed'}
    if (DEST/'protocol.json').exists():assert json.loads((DEST/'protocol.json').read_text())==prot
    else:put(DEST/'protocol.json',prot)
    np.save(DEST/'plan.npy',plan)
    for branch in prot['branches']:
        folder=DEST/branch;folder.mkdir(exist_ok=True)
        if (folder/'report.json').exists():continue
        torch.manual_seed(SEED);random.seed(SEED);np.random.seed(SEED)
        model=p.w0.V1FR12();model.load_state_dict(torch.load(source,map_location='cpu',weights_only=False)['model'])
        optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-4)
        schedule=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,STEPS,eta_min=1e-5)
        weights=torch.ones(65);weights[64]=.1
        cache=np.load(teacher_cache,mmap_mode='r') if branch=='TEACHER_DETECTOR' else None
        state={'step':0,'best_step':0,'evaluations':[],'losses':[]}
        def evaluate(step):
            model.eval();m=p.evaluate(model,old['DEV'],ops);model.train()
            state['evaluations'].append({'step':step,'metrics':m})
            if 'best' not in state or p.key(m)>p.key(state['best']):
                state.update(best=m,best_step=step);checkpoint(folder/'best.pt',{'model':model.state_dict(),'step':step,'metrics':m,'protocol_sha256':digest(DEST/'protocol.json'),'parent_checkpoint_sha256':source_hash})
            put(folder/'progress.json',state);print('DEV',branch,step,json.dumps(m),flush=True)
        def save():checkpoint(folder/'latest.pt',{'model':model.state_dict(),'optimizer':optimizer.state_dict(),'schedule':schedule.state_dict(),'state':state,'protocol_sha256':digest(DEST/'protocol.json')})
        if (folder/'latest.pt').exists():
            c=torch.load(folder/'latest.pt',map_location='cpu',weights_only=False);assert c['protocol_sha256']==digest(DEST/'protocol.json')
            model.load_state_dict(c['model']);optimizer.load_state_dict(c['optimizer']);schedule.load_state_dict(c['schedule']);state=c['state']
        else:evaluate(0);save()
        started=time.time()
        try:
            for step in range(state['step'],STEPS):
                if step%25==0 and free_memory_mib()<1536:raise RuntimeError('Host memory below safe threshold; checkpoint saved, no repeated OOM attempts')
                optimizer.zero_grad(set_to_none=True);losses=[]
                for i in plan[step]:
                    with np.load(BASE/'train'/('%03d.npz'%i)) as z:
                        x=torch.from_numpy(z['pair']);labels=torch.from_numpy(z['labels']);points=torch.from_numpy(z['points'])
                    a,b=model(x)
                    if branch=='CONTINUE_SUP':det=F.cross_entropy(a,labels,weight=weights)
                    else:
                        ta=torch.from_numpy(cache[i].astype(np.float32));det=F.kl_div(a.log_softmax(1),ta.softmax(1),reduction='none').sum(1).mean()
                    d=p.sample(b,points,ops);sim=d[0]@d[1].T/.1;target=torch.arange(len(d[0]));geo=(F.cross_entropy(sim,target)+F.cross_entropy(sim.T,target))/2
                    loss=(det+geo)/2
                    if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
                    loss.backward();losses.append([float(det.detach()),float(geo.detach())])
                grad=torch.nn.utils.clip_grad_norm_(model.parameters(),10)
                if not torch.isfinite(grad):raise RuntimeError('Nonfinite gradient')
                optimizer.step();schedule.step();state['step']=step+1;state['losses'].append({'step':step+1,'detector':float(np.mean(losses,axis=0)[0]),'geometry':float(np.mean(losses,axis=0)[1])})
                if (step+1)%25==0:save();print('TRAIN',branch,step+1,flush=True)
                if (step+1)%100==0:evaluate(step+1);save()
        except BaseException:save();raise
        put(folder/'report.json',{'status':'COMPLETE','state':state,'elapsed_s':time.time()-started,'rss_peak_MiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,'protocol_sha256':digest(DEST/'protocol.json'),'plan_sha256':prot['plan_sha256']})
        del model,optimizer,schedule,cache;gc.collect()
    reports={b:json.loads((DEST/b/'report.json').read_text()) for b in prot['branches']}
    a=reports['CONTINUE_SUP']['state'];b=reports['TEACHER_DETECTOR']['state']
    assert a['evaluations'][0]==b['evaluations'][0]
    m,n=a['best'],b['best'];passed=n['mnn_precision']>=m['mnn_precision']+.03 and n['repeatability_3px']>=m['repeatability_3px']-.02 and n['mnn_precision']>=.5
    result={'status':'COMPLETE','fairness_pass':True,'gate_pass':passed,'control_best':m,'teacher_detector_best':n,'control_best_step':a['best_step'],'teacher_detector_best_step':b['best_step'],'precision_delta':n['mnn_precision']-m['mnn_precision'],'scope':prot['scope'],'next':'replicate and expand geometry data before full training' if passed else 'inspect learning curves and supervision/capacity evidence; no automatic full training'}
    put(DEST/'summary.json',result);print('AB_COMPLETE',json.dumps(result),flush=True)
if __name__=='__main__':run()
