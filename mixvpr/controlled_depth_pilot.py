"""Paired D4/D6 continuation pilot using the frozen Gate2 recipe and DEV only."""
from __future__ import annotations
import os
os.environ.update(OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
                  CUBLAS_WORKSPACE_CONFIG=':4096:8', PYTHONDONTWRITEBYTECODE='1')
from pathlib import Path
import sys, json, random, hashlib, time, subprocess, re
from collections import Counter
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'mixvpr/artifacts/depth_pilot'
BASE = Path('/home/quyet/training_recovery/final_gate2')
DATA = Path('/mnt/d/k210_official_ab')
CAL = BASE / 'calibration_train_f32.npy'
SEED, BUDGET, INTERVAL = 20260913, 1000, 250
BRANCHES = ('D4', 'D6')

def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''): h.update(b)
    return h.hexdigest()

def put(p, value):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + '.tmp'); tmp.write_text(json.dumps(value, indent=2) + '\n'); tmp.replace(p)

def deps():
    sys.path.insert(0, '/home/quyet/training_recovery/official_ab')
    import experiment as old
    return old

def build(branch):
    import torch
    import torch.nn as nn
    sys.path.insert(0, str(ROOT / 'experiments/edge_vpr_students'))
    from s512_br1_sq240_m100_c160h96_model import S512BR1SQ240M100C160H96, FeatureMixer
    model = S512BR1SQ240M100C160H96()
    if branch == 'D6':
        model.mixers = nn.Sequential(*list(model.mixers.children()), FeatureMixer(), FeatureMixer())
    return model

def setup_inputs(old, data):
    """Cache exactly the existing DEV gallery images; no new split or labels."""
    ids = sorted({int(i) for s in data['dev_protocol'] for i in np.load(DATA / f'{s}_dev.npz')['ids']})
    path = OUT / 'dev_inputs.npy'
    x = np.lib.format.open_memmap(path, mode='w+', dtype=np.float32, shape=(len(ids), 3, 240, 240))
    for j, i in enumerate(ids):
        row = data['rows'][i]
        assert row['split'] == 'dev' and sha(row['path']) == row['sha256']
        x[j] = old.image(row['path'], 240).numpy()
    x.flush(); del x
    put(OUT / 'dev_inputs.json', {'ids': ids, 'sha256': sha(path), 'count': len(ids),
                                'scope': 'Existing locked DEV only, both query and gallery use same backend'})

def prepare():
    import torch
    old = deps(); data = old.load_data()
    OUT.mkdir(parents=True, exist_ok=False)
    identity = json.loads((BASE / 'identity.json').read_text())
    assert sha(BASE / 'candidate.pt') == identity['checkpoint_sha256']
    cfg = json.loads(Path('/mnt/d/k210_gate2/protocol.json').read_text())['config']
    bank = json.loads((DATA / 'official_bank/manifest.json').read_text())
    assert sha(DATA / 'official_bank/descriptors_f16.npy') == bank['descriptors_sha256']
    locked = {str(f): sha(f) for f in [BASE/'candidate.pt', BASE/'identity.json', CAL,
              DATA/'data.json', DATA/'official_bank/manifest.json', DATA/'official_bank/descriptors_f16.npy',
              Path(old.__file__), Path(old.core.__file__),
              ROOT/'experiments/edge_vpr_students/s512_br1_sq240_m100_c160h96_model.py',
              ROOT/'superpoint/artifacts/full_training/freeze_manifest.json',
              ROOT/'superpoint/artifacts/full_training/final_test/results.json']}
    for s in data['dev_protocol']: locked[str(DATA/f'{s}_dev.npz')] = sha(DATA/f'{s}_dev.npz')
    protocol = {
        'seed': SEED, 'max_additional_updates_per_branch': BUDGET, 'eval_interval': INTERVAL,
        'source_checkpoint_sha256': identity['checkpoint_sha256'], 'source_cumulative_updates': 5750,
        'recipe': cfg, 'recipe_override': 'Only D6 adds two mixer blocks; inherited losses, mining, augmentation and sampler unchanged.',
        'schedule': 'Both restart AdamW as original Gate2 continuation: lr1e-4,wd1e-4,cosine T_max6000,eta_min3e-6; pilot is bounded prefix, not shortened cosine.',
        'initialization': 'Every compatible D4 tensor copied bitwise; new D6 blocks use seeded default init with fc2 weight/bias zero, making residual blocks identity at update0.',
        'precision': 'Original fp16 autocast, independent GradScaler init1024, FP32 loss, norm clip10;8 images x4 accumulation, retry same data on AMP overflow.',
        'selection': 'Unchanged lexicographic Overall0.9 Indoor scene-macro+0.1 Project: R1,R5,R10,margin; earliest tie; initial checkpoint eligible.',
        'stop_rule': 'Joint stop after>=500 updates only if neither branch improved Overall R1 by0.001 for500 updates; otherwise stop both at1000. No branch gets extra updates.',
        'primary_decision': 'D6 merits longer training only if selected FP32 R1 improves >=1pp in at least one domain, neither domain regresses, last two paired eval deltas are nonnegative on average in both domains, and both deployment guardrails PASS. Otherwise NO_DEMONSTRATED_BENEFIT (not proof D6 never helps).',
        'guardrails': {'R1_retention_min_each_domain': .95, 'R1_drop_max_pp_each_domain': 2.,
                       'compiler_total_max_bytes': 3*1024**2, 'compile_gencode_sim': 'PASS', 'cpu_conv': 0},
        'test_access': False, 'new_KD': False, 'full_training': False,
        'project_caveat': 'Existing image-retrieval DEV has18 queries; not retired stair-map localization or independent TEST.',
        'locked_files': locked, 'runner_sha256': sha(__file__),
    }
    put(OUT/'protocol.json', protocol)
    torch.manual_seed(SEED)
    state = torch.load(BASE/'candidate.pt', map_location='cpu', weights_only=False)['model_state_dict']
    transfer = {}
    for branch in BRANCHES:
        m = build(branch)
        missing, unexpected = m.load_state_dict(state, strict=False)
        assert not unexpected
        if branch == 'D6':
            assert len(missing) == 12 and all(k.startswith(('mixers.4.', 'mixers.5.')) for k in missing)
            for block in list(m.mixers)[4:]:
                torch.nn.init.zeros_(block.fc2.weight); torch.nn.init.zeros_(block.fc2.bias)
        else: assert not missing
        assert all(torch.equal(m.state_dict()[k], v) for k, v in state.items())
        folder=OUT/branch;folder.mkdir()
        torch.save({'model_state_dict': m.state_dict(), 'step': 0}, folder/'init.pt')
        transfer[branch] = {'copied_tensors':len(state), 'copied_parameters':sum(v.numel() for v in state.values()),
                            'new_keys':missing, 'init_sha256':sha(folder/'init.pt')}
    # Same source sample/microbatch/augmentation generation as Gate2.
    rows, places = data['rows'], data['train_places']; neighbors = bank['neighbors']
    indoor = sorted(k for k,v in places.items() if rows[v[0]]['domain']=='indoor')
    project = sorted(k for k,v in places.items() if rows[v[0]]['domain']=='project')
    rng = random.Random(SEED); plan = np.empty((BUDGET,4,8,3),np.int64)
    for step in range(BUDGET):
        for micro in range(4):
            anchor=rng.choice(project if rng.random()<.1 else indoor);negative=rng.choice(neighbors[anchor][:16]);cursor=0
            for label,k in enumerate((anchor,negative)):
                chosen=rng.sample(places[k],min(4,len(places[k])))
                while len(chosen)<4:chosen.append(rng.choice(places[k]))
                for i in chosen:plan[step,micro,cursor]=[i,label,rng.randrange(2**31)];cursor+=1
    sampled=set(plan[:,:,:,0].ravel().tolist());assert all(rows[i]['split']=='train' for i in sampled)
    assert all(sha(rows[i]['path'])==rows[i]['sha256'] for i in sampled)
    np.save(OUT/'plan.npy',plan)
    setup_inputs(old,data)
    put(OUT/'initialization.json', {'transfer':transfer,'plan_sha256':sha(OUT/'plan.npy'),
                                  'unique_train_images_verified':len(sampled),'source_weights_equal':True})

def metrics(values, data):
    import torch
    old=deps();ids=json.loads((OUT/'dev_inputs.json').read_text())['ids'];offset={i:j for j,i in enumerate(ids)};scenes={}
    for name in data['dev_protocol']:
        d=np.load(DATA/f'{name}_dev.npz');v=torch.from_numpy(values[[offset[int(i)] for i in d['ids']]])
        sim=v[d['query_indices']]@v.T;allowed=torch.from_numpy(d['allowed']);pos=torch.from_numpy(d['positive']);neg=torch.from_numpy(d['negative']);sim.masked_fill_(~allowed,-1e4);order=sim.argsort(1,descending=True)
        r={f'R@{k}':float(pos.gather(1,order[:,:k]).any(1).float().mean()) for k in (1,5,10)}
        r['margin']=float((sim.masked_fill(~pos,-1e4).max(1).values-sim.masked_fill(~neg,-1e4).max(1).values).mean());r['queries']=len(sim);scenes[name]=r
    indoor={k:float(np.mean([r[k] for s,r in scenes.items() if s!='project_stair'])) for k in old.KEYS};project={k:scenes['project_stair'][k] for k in old.KEYS}
    return {'Indoor':indoor,'Project':project,'Overall':{k:.9*indoor[k]+.1*project[k] for k in old.KEYS},'scenes':scenes}

def evaluate(model, data, device, save_path=None):
    import torch
    import torch.nn.functional as F
    model.eval();inputs=np.load(OUT/'dev_inputs.npy',mmap_mode='r');values=[]
    with torch.inference_mode():
        for i in range(0,len(inputs),2):
            x=torch.from_numpy(np.array(inputs[i:i+2])).to(device)
            values.append(F.normalize(model.raw(x).float().cpu(),dim=1).numpy())
    values=np.concatenate(values)
    if save_path:np.save(save_path,values)
    return metrics(values,data)

def train():
    import torch
    import torch.nn.functional as F
    old=deps();device=old.setup()
    if not (OUT/'protocol.json').exists():prepare()
    protocol=json.loads((OUT/'protocol.json').read_text());assert sha(__file__)==protocol['runner_sha256']
    assert all(sha(p)==h for p,h in protocol['locked_files'].items())
    init=json.loads((OUT/'initialization.json').read_text());assert sha(OUT/'plan.npy')==init['plan_sha256']
    data=old.load_data();rows=data['rows'];plan=np.load(OUT/'plan.npy')
    bank=json.loads((DATA/'official_bank/manifest.json').read_text());cache=np.load(DATA/'official_bank/descriptors_f16.npy',mmap_mode='r');offset={i:j for j,i in enumerate(bank['ids'])}
    torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED);random.seed(SEED);np.random.seed(SEED)
    models={};optimizers={};schedulers={};scalers={}
    for b in BRANCHES:
        models[b]=build(b).to(device);models[b].load_state_dict(torch.load(OUT/b/'init.pt',map_location='cpu',weights_only=False)['model_state_dict'])
        optimizers[b]=torch.optim.AdamW(models[b].parameters(),lr=1e-4,weight_decay=1e-4)
        schedulers[b]=torch.optim.lr_scheduler.CosineAnnealingLR(optimizers[b],T_max=6000,eta_min=3e-6)
        scalers[b]=torch.amp.GradScaler('cuda',init_scale=1024)
    state={'step':0,'evaluations':{b:[] for b in BRANCHES},'best_step':{b:0 for b in BRANCHES},
           'best_metrics':{},'last_signal':{b:0 for b in BRANCHES},'signal_r1':{},'amp_retries':{b:0 for b in BRANCHES},'trace':[]}
    resume=OUT/'latest.pt'
    if resume.exists():
        c=torch.load(resume,map_location='cpu',weights_only=False);assert c['protocol_sha256']==sha(OUT/'protocol.json');state=c['state']
        for b in BRANCHES:
            models[b].load_state_dict(c[b]['model']);optimizers[b].load_state_dict(c[b]['optimizer']);schedulers[b].load_state_dict(c[b]['scheduler']);scalers[b].load_state_dict(c[b]['scaler'])
        old.core.restore_rng(c['rng']);del c
    else:
        # Verify cached-input evaluator against the actual existing evaluator before training.
        for b in BRANCHES:
            result=evaluate(models[b],data,device);state['evaluations'][b].append({'step':0,'metrics':result});state['best_metrics'][b]=result;state['signal_r1'][b]=result['Overall']['R@1']
            old.save(OUT/b/'best.pt',{'model_state_dict':models[b].state_dict(),'step':0,'dev_metrics':result})
        assert state['best_metrics']['D4']==state['best_metrics']['D6'],'Identity initialization changed outputs'
        reference=old.evaluate(models['D4'],rows,device)
        assert reference==state['best_metrics']['D4'],'Evaluator mismatch'
        identity=json.loads((BASE/'identity.json').read_text())
        for domain in ('Indoor','Project'):
            assert abs(reference[domain]['R@1']-identity['DEV'][domain]['R@1'])<1e-7
        put(OUT/'evaluator_parity.json',{'status':'PASS','full_DEV_cached_vs_legacy':'EXACT','D4_D6_step0':'EXACT','historical_R1_reproduced':True})
    def checkpoint():
        value={'state':state,'rng':old.core.rng_state(),'protocol_sha256':sha(OUT/'protocol.json')}
        for b in BRANCHES:value[b]={'model':models[b].state_dict(),'optimizer':optimizers[b].state_dict(),'scheduler':schedulers[b].state_dict(),'scaler':scalers[b].state_dict()}
        old.save(resume,value);put(OUT/'progress.json',state)
    checkpoint();started=time.time()
    while state['step']<BUDGET:
        step=state['step'];old.check_resources()
        batches=[]
        for item in plan[step]:
            x=torch.stack([old.image(rows[int(i)]['path'],240,int(a)) for i,label,a in item]).to(device)
            labels=torch.tensor(item[:,1].copy(),device=device)
            teacher=F.normalize(torch.from_numpy(np.stack([cache[offset[int(i)]].astype(np.float32) for i in item[:,0]])).to(device),dim=1)
            batches.append((x,labels,teacher))
        record={'step':step+1}
        for b in BRANCHES:
            model=models[b];model.train();optimizer=optimizers[b];scaler=scalers[b]
            for attempt in range(4):
                optimizer.zero_grad(set_to_none=True);sums=np.zeros(3)
                for x,labels,t in batches:
                    with torch.amp.autocast('cuda',dtype=torch.float16):s=model(x)
                    s=F.normalize(s.float(),dim=1);task=old.core.retrieval_loss(s,labels);kd,confidence,_=old.core.ranking_kd(s,t,labels,.07)
                    loss=(task+kd)/4
                    assert torch.isfinite(loss)
                    scaler.scale(loss).backward();sums+=np.array([task.item(),kd.item(),confidence.item()])/4
                scaler.unscale_(optimizer);norm=torch.nn.utils.clip_grad_norm_(model.parameters(),10.)
                if not torch.isfinite(norm):scaler.update();state['amp_retries'][b]+=1;continue
                scaler.step(optimizer);scaler.update();schedulers[b].step();break
            else:raise RuntimeError('Repeated nonfinite gradients')
            record[b]={'SUP_loss':sums[0],'KD_loss':sums[1],'confidence':sums[2],'grad_norm':float(norm),'lr':optimizer.param_groups[0]['lr']}
        assert record['D4']['lr']==record['D6']['lr'];state['step']+=1;state['trace'].append(record)
        if state['step']%50==0:checkpoint();print('PAIRED_UPDATE',state['step'],flush=True)
        if state['step']%INTERVAL==0:
            for b in BRANCHES:
                result=evaluate(models[b],data,device);state['evaluations'][b].append({'step':state['step'],'metrics':result})
                if old.metric_key(result)>old.metric_key(state['best_metrics'][b]):
                    state['best_step'][b]=state['step'];state['best_metrics'][b]=result
                    old.save(OUT/b/'best.pt',{'model_state_dict':models[b].state_dict(),'step':state['step'],'dev_metrics':result})
                if result['Overall']['R@1']>=state['signal_r1'][b]+.001:
                    state['last_signal'][b]=state['step'];state['signal_r1'][b]=result['Overall']['R@1']
                print('DEV',b,state['step'],result['Indoor']['R@1'],result['Project']['R@1'],flush=True)
            checkpoint()
            if state['step']>=500 and all(state['step']-state['last_signal'][b]>=500 for b in BRANCHES):break
    state.update(status='PILOT_COMPLETE',stop_reason='BUDGET' if state['step']==BUDGET else 'JOINT_PLATEAU',resources=old.resource_stats(),elapsed_seconds_this_process=time.time()-started)
    for b in BRANCHES:
        old.save(OUT/b/'last.pt',{'model_state_dict':models[b].state_dict(),'step':state['step']})
        c=torch.load(OUT/b/'best.pt',map_location='cpu',weights_only=False);models[b].load_state_dict(c['model_state_dict'])
        actual=evaluate(models[b],data,device,OUT/b/'fp32_dev.npy');assert actual==state['best_metrics'][b]
    put(OUT/'training.json',state);print('PILOT_COMPLETE',state['step'],state['stop_reason'],flush=True)

def export():
    import torch,onnx,onnxruntime as ort
    sys.path.insert(0,str(ROOT/'experiments/edge_vpr_students'))
    from sq240_resource_probe_export import PreL2,canonicalize
    torch.set_num_threads(2);cal=np.load(CAL)
    for b in BRANCHES:
        d=OUT/b;m=build(b);c=torch.load(d/'best.pt',map_location='cpu',weights_only=False);m.load_state_dict(c['model_state_dict']);m.eval()
        torch.onnx.export(PreL2(m),torch.from_numpy(cal[:1]),str(d/'raw.onnx'),opset_version=13,dynamo=False,input_names=['normalized_rgb'],output_names=['raw_descriptor'])
        onnx.save(onnx.shape_inference.infer_shapes(onnx.load(d/'raw.onnx'),strict_mode=True),d/'raw.onnx');canonicalize(d/'raw.onnx',d/'model.onnx')
        options=ort.SessionOptions();options.intra_op_num_threads=2;options.inter_op_num_threads=1
        s=ort.InferenceSession(str(d/'model.onnx'),options,providers=['CPUExecutionProvider']);errors=[]
        for x in cal[::8]:
            with torch.inference_mode():expected=m.raw(torch.from_numpy(x[None])).numpy()
            actual=s.run(None,{'normalized_rgb':x[None]})[0];np.testing.assert_allclose(actual,expected,atol=1e-4,rtol=1e-3);errors.append(float(abs(actual-expected).max()))
        put(d/'export.json',{'status':'PASS','checkpoint_sha256':sha(d/'best.pt'),'onnx_sha256':sha(d/'model.onnx'),'max_abs':max(errors),'samples':8,'calibration_sha256':sha(CAL)})

def compile_one(b):
    import nncase,_nncase
    from importlib.metadata import version
    assert version('nncase')=='1.8.0.20220929' and _nncase.__version__=='1.8.0-55be52f'
    d=OUT/b;assert json.loads((d/'export.json').read_text())['status']=='PASS'
    r={k:'NOT_RUN' for k in ('compile','gencode','sim')};stage='compile'
    try:
        options=nncase.CompileOptions();options.target='k210';options.quant_type=options.w_quant_type='uint8';options.dump_asm=True;options.dump_dir=str(d/'dump')
        c=nncase.Compiler(options);c.import_onnx((d/'model.onnx').read_bytes(),nncase.ImportOptions());cal=np.load(CAL);q=nncase.PTQTensorOptions();q.samples_count=len(cal);q.set_tensor_data(cal.tobytes());c.use_ptq(q);c.compile();r[stage]='PASS'
        stage='gencode';blob=c.gencode_tobytes();(d/'model.kmodel').write_bytes(blob);r.update(gencode='PASS',model_bytes=len(blob),kmodel_sha256=sha(d/'model.kmodel'))
        info=(d/'dump/kmodel_info.txt').read_text();r['memory']={k:int(v) for k,v in re.findall(r'(input|output|data|MODEL|TOTAL):?\s+[^\n]*?\((\d+) B\)',info)}
        r['mapping']={}
        for backend in ('stackvm','k210'):
            txt='\n'.join(f.read_text() for f in (d/'dump'/backend).rglob('runtime_ops.txt'));r['mapping'][backend]=dict(Counter(re.findall(r'\[([A-Za-z0-9_]+)\]',txt)))
        stage='sim';sim=nncase.Simulator();sim.load_model(blob);sim.set_input_tensor(0,nncase.RuntimeTensor.from_numpy(cal[:1].copy()));sim.run();v=sim.get_output_tensor(0).to_numpy();assert v.shape==(1,512) and np.isfinite(v).all() and np.linalg.norm(v)>1e-12;r[stage]='PASS'
    except Exception as e:r.update(failed_stage=stage,error=str(e));r[stage]='FAIL'
    put(d/'compile.json',r);print(b,r,flush=True)

def simulate(shard,shards):
    import nncase
    inputs=np.load(OUT/'dev_inputs.npy',mmap_mode='r');sims={};hashes={}
    for b in BRANCHES:
        d=OUT/b;r=json.loads((d/'compile.json').read_text())
        if r['sim']!='PASS':continue
        assert sha(d/'model.kmodel')==r['kmodel_sha256'];s=nncase.Simulator();s.load_model((d/'model.kmodel').read_bytes())
        assert s.get_input_tensor(0).dtype==np.dtype('float32') and s.get_output_tensor(0).dtype==np.dtype('float32')
        sims[b]=s;hashes[b]=r['kmodel_sha256'];(d/'int8_dev').mkdir(exist_ok=True)
    assert sims
    indices=list(range(shard,len(inputs),shards));completed=[]
    for i in indices:
        for b,s in sims.items():
            dest=OUT/b/'int8_dev'/f'{i:04d}.npy'
            if dest.exists():
                v=np.load(dest);assert v.shape==(512,) and np.isfinite(v).all();continue
            s.set_input_tensor(0,nncase.RuntimeTensor.from_numpy(np.array(inputs[i:i+1])));s.run();v=s.get_output_tensor(0).to_numpy().astype(np.float32).reshape(-1)
            assert v.shape==(512,) and np.isfinite(v).all() and np.linalg.norm(v)>1e-12
            v=v/max(np.linalg.norm(v),1e-12);np.save(dest,v)
        completed.append(i)
        if len(completed)%50==0:print('SIM_DEV',shard,len(completed),'/',len(indices),flush=True)
    put(OUT/f'simulator_{shard}.json',{'shard':shard,'shards':shards,'indices':completed,'kmodel_hashes':hashes,
                                   'input_sha256':sha(OUT/'dev_inputs.npy'),'output_hashes':{b:{str(i):sha(OUT/b/'int8_dev'/f'{i:04d}.npy') for i in indices} for b in sims}})

if __name__=='__main__':
    stage=sys.argv[1]
    if stage=='train':train()
    elif stage=='export':export()
    elif stage=='compile':compile_one(sys.argv[2])
    elif stage=='simulate':simulate(int(sys.argv[2]),int(sys.argv[3]))
    else:raise SystemExit('train | export | compile D4/D6 | simulate SHARD SHARDS')
