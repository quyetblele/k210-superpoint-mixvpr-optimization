"""Read-only checkpoint diagnosis; writes isolated evidence, never trains."""
import os
os.environ.update(OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
import sys, json, hashlib, time, resource, importlib.util
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
OUT=Path(__file__).parent
BASE=Path('/home/quyet/training_recovery/sp_ab')
spec=importlib.util.spec_from_file_location('sp_pilot',BASE/'experiment.py')
p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)
sys.path.insert(0,str(p.w0.EDGE/'third_party'))
from SuperGluePretrainedNetwork.models import superpoint as ops

def write(name,x):
    tmp=OUT/(name+'.tmp');tmp.write_text(json.dumps(x,indent=2)+'\n');tmp.replace(OUT/name)
def digest(path):return p.sha(path)
def oracle(dense,points):
    d=p.sample(dense,points,ops);sim=d[0]@d[1].T;n=len(sim)
    ids=torch.arange(n);pos=sim.diag();neg=sim.masked_fill(torch.eye(n,dtype=torch.bool),-2).max(1).values
    return {'oracle_top1':float((sim.argmax(1)==ids).float().mean()),'positive_cosine':float(pos.mean()),'hardest_negative_cosine':float(neg.mean()),'positive_negative_margin':float((pos-neg).mean())}
def metrics(logits,dense,z):
    f=p.features(logits,dense,ops)
    m=p.homography_metrics(f[0],f[1],z['homography'],p.W,p.H)
    m['correct_mnn_count']=m['mnn_count']*m['mnn_precision']
    return m

@torch.inference_mode()
def main():
    start=time.time();protocol=json.loads((BASE/'protocol.json').read_text())
    models={};identities={}
    for branch in ('SUP','KD'):
        file=BASE/branch/'best.pt';ck=torch.load(file,map_location='cpu',weights_only=False)
        model=p.w0.V1FR12().eval();model.load_state_dict(ck['model']);models[branch]=model
        identities[branch]={'path':str(file),'sha256':digest(file),'step':ck['step'],'metrics':ck['metrics']}
    assert digest(p.w0.CHECKPOINT)==protocol['teacher_sha256']
    caches={k:np.load(BASE/'dev'/('teacher_'+k+'.npy'),mmap_mode='r') for k in ('logits','desc')}
    predeclared={'purpose':'head-swap diagnosis plus trained-weight deployment feasibility; no new training, architecture or MixVPR changes',
       'selection':'existing DEV lexicographic precision, repeatability, MNN count; no selection on PTQ metrics',
       'data':'exact existing 28 DEV homography pairs; related to previous pilot, not independent test',
       'teacher':'FP16 disk cache restored to FP32; original teacher checkpoint identity checked',
       'variants':['teacher/teacher','SUP/SUP','SUP/teacher','teacher/SUP','KD/KD','KD/teacher','teacher/KD'],
       'notation':'detector source / descriptor source; oracle matches known geometric point pairs',
       'physical_inference_batch':2,'CPU_threads':2,'GPU':False,'workers':0,'GT1_used':False,
       'checkpoint_identities':identities,'source_protocol_sha256':digest(BASE/'protocol.json'),'code_sha256':digest(__file__)}
    if (OUT/'protocol.json').exists():assert json.loads((OUT/'protocol.json').read_text())==predeclared
    else:write('protocol.json',predeclared)
    rows=[]
    for i,row in enumerate(protocol['DEV']):
        assert digest(row['path'])==row['sha256']
        with np.load(BASE/'dev'/('%03d.npz'%i)) as zz:z={k:zz[k] for k in zz.files}
        x=torch.from_numpy(z['pair']);points=torch.from_numpy(z['points'])
        outputs={'teacher':tuple(torch.from_numpy(caches[k][i].astype(np.float32)) for k in ('logits','desc'))}
        outputs.update({name:model(x) for name,model in models.items()})
        for variant in predeclared['variants']:
            detector,descriptor=variant.split('/')
            m=metrics(outputs[detector][0],outputs[descriptor][1],z)
            rows.append({'index':i,'scene':row['scene'],'variant':variant,'metrics':m})
        for name,(a,b) in outputs.items():
            rows.append({'index':i,'scene':row['scene'],'variant':name+'/oracle','metrics':oracle(b,points)})
        if i%7==0:print('DIAG',i,len(protocol['DEV']),flush=True)
    grouped={}
    for name in sorted({r['variant'] for r in rows}):
        rr=[r for r in rows if r['variant']==name]
        def avg(rs):return {k:float(np.mean([r['metrics'][k] for r in rs])) for k in rs[0]['metrics']}
        grouped[name]={'mean':avg(rr),'per_scene':{scene:avg([r for r in rr if r['scene']==scene]) for scene in sorted({r['scene'] for r in rr})}}
    # Verify reproduction of the old evaluator before interpreting crossed heads.
    old=json.loads((BASE/'summary.json').read_text())
    for b in ('SUP','KD'):
        for k in ('mnn_precision','repeatability_3px','mnn_count'):
            assert abs(grouped[b+'/'+b]['mean'][k]-old[b+'_final'][k])<1e-6,(b,k)
    winner=max(identities,key=lambda b:p.key(identities[b]['metrics']))
    model=models[winner];activations=[];hooks=[]
    for name,module in model.named_modules():
        if isinstance(module,torch.nn.Conv2d):
            def hook(m,inputs,output,name=name):
                mac=output.numel()*(m.in_channels//m.groups)*m.kernel_size[0]*m.kernel_size[1]
                activations.append({'name':name,'shape':list(output.shape),'parameters':sum(t.numel() for t in m.parameters()),'MAC':mac,'logical_output_fp32_bytes':output.numel()*4,'projected_uint8_bytes':output.numel()})
            hooks.append(module.register_forward_hook(hook))
    model(torch.zeros(1,1,p.H,p.W))
    for h in hooks:h.remove()
    resource_report={'parameters':sum(v.numel() for v in model.parameters()),'conv_MAC':sum(v['MAC'] for v in activations),'layers':activations,
       'logical_input_fp32_bytes':p.H*p.W*4,'logical_two_head_fp32_bytes':(p.H//8)*(p.W//8)*321*4,
       'actual_SRAM_peak':'NOT_MEASURED','note':'Per-tensor sizes and compiler memory statistics are not actual board peak'}
    write('resource_profile.json',resource_report)
    result={'status':'COMPLETE','old_metrics_reproduced':True,'variants':grouped,'selected_existing_pilot':winner,
        'elapsed_s':time.time()-start,'rss_peak_MiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        'scope':predeclared['data'],'not_final_candidate':True}
    write('diagnosis_rows.json',rows);write('diagnosis.json',result)
    print(json.dumps({'selected':winner,'metrics':{k:v['mean'] for k,v in grouped.items()},'RSS_MiB':result['rss_peak_MiB']}),flush=True)
if __name__=='__main__':main()
