"""Sequential, resumable quality screening and auditable report."""
import os
os.environ.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
import json, subprocess, hashlib, csv, time
from pathlib import Path
BASE=Path(__file__).parent
ROOT=BASE/'quality_pilot_v2'
PY='/home/quyet/k210_lab/.venv-mixvpr-cuda/bin/python'
CANDIDATES=[('fr12','r1'),('wider','r1'),('fr12','r2'),('wider','r2')]
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def put(p,x):
    t=Path(str(p)+'.tmp');t.write_text(json.dumps(x,indent=2)+'\n');t.replace(p)
def run():
    done=json.loads((ROOT/'cache_done.json').read_text())
    for name,digest in done['files'].items():assert sha(ROOT/name)==digest,name
    put(ROOT/'run_status.json',{'status':'RUNNING','candidate':None})
    for width,res in CANDIDATES:
        name=width+'_'+res+('_cw' if res=='r2' else '')
        put(ROOT/'run_status.json',{'status':'RUNNING','candidate':name})
        with (ROOT/(name+'.log')).open('a') as log:
            subprocess.run([PY,str(BASE/'quality_pilot_v2.py'),'train','--width',width,'--resolution',res],stdout=log,stderr=subprocess.STDOUT,check=True)
    report()

def report():
    reports={}; rows=[]; plans=set(); init={}
    for width,res in CANDIDATES:
        name=width+'_'+res+('_cw' if res=='r2' else ''); r=json.loads((ROOT/name/'report.json').read_text());reports[name]=r
        assert r['state']['step']==400 and r['status']=='COMPLETE'
        plans.add(r['plan_sha256']);init.setdefault(width,set()).add(r['init_sha256'])
        m=r['state']['best']['macro']; t=json.loads((ROOT/res/'teacher_dev.json').read_text())['macro']
        rows.append({'candidate':name,'best_step':r['state']['best_step'],**m,'correct_retention_vs_teacher':m['correct_mnn_count']/max(t['correct_mnn_count'],1e-9),'elapsed_s':r['elapsed_s'],'peak_RSS_MiB':r['peak_process_RSS_MiB']})
    assert len(plans)==1 and all(len(v)==1 for v in init.values())
    winner=max(rows,key=lambda r:(r['correct_mnn_count'],r['mnn_precision'],r['repeatability']))['candidate']
    with (ROOT/'comparison.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(13,3.8))
    for name,r in reports.items():
        ev=r['state']['evaluations'];x=[v['step'] for v in ev]
        for ax,key,label in zip(axes,['correct_mnn_count','mnn_precision','coverage'],['Correct mutual matches / pair','MNN precision','Correct-match spatial coverage']):
            ax.plot(x,[v['metrics']['macro'][key] for v in ev],marker='o',label=name);ax.set(xlabel='Optimizer updates',ylabel=label);ax.grid(alpha=.2)
    axes[0].legend(fontsize=7);fig.tight_layout();fig.savefig(ROOT/'learning_curves.png',dpi=150);fig.savefig(ROOT/'learning_curves.pdf');plt.close(fig)
    result={'status':'COMPLETE','fairness_checks_pass':True,'selected_for_next_gate':winner,'comparison':rows,'GT1_used':False,'INT8_quality':'NOT_EVALUATED_FOR_THESE_TRAINED_WEIGHTS','board_latency_and_SRAM_peak':'NOT_MEASURED','next_gate':'trained-weight ONNX parity -> TRAIN-only PTQ -> real nncase simulator descriptor/keypoint/matching quality for quality/cost Pareto candidates; no full train automatically','limitations':['One seed and400updates, not training to convergence','Fixed synthetic homography pairs, no real-viewpoint or independent localization evaluation','TRAIN/DEV project images from related capture sequences','Teacher detector supervision is common recipe, this does not isolate KD benefit','Equal update budget, not equal compute; wider/higher-resolution candidates cost more','Random-weight compile feasibility is not trained-weight INT8 quality']}
    put(ROOT/'summary.json',result)
    lines=['# SuperPoint: matched four-configuration quality pilot','',f'Completed400updates per configuration. Selected for the next engineering gate: **{winner}**. This is a pilot selection, not a final K210 release model.','',
           'This v2 reruns all candidates from their original initialization after an audit found resolution-dependent border filtering in v1. v1 results are excluded from ranking. Both resolutions now use exactly the same canonical training correspondences. All variants use the same140TRAIN/42DEV source images, seeded sample plan, deterministic teacher-based initialization strategy, optimizer budget and supervision. Same-width variants start from identical checkpoint hashes. DEV metrics use a common144x256 coordinate frame,3px geometry tolerance, top160 points and valid mutual overlap. Selection: equal-scene macro correct matches, then precision, then repeatability. GT1 and MixVPR were not used.','',
           '| Candidate | Best update | Correct matches | Precision | Repeatability | Coverage | Correct/teacher |','|---|---:|---:|---:|---:|---:|---:|']
    for r in rows:lines.append(f"|{r['candidate']}|{r['best_step']}|{r['correct_mnn_count']:.2f}|{r['mnn_precision']:.2%}|{r['repeatability']:.2%}|{r['coverage']:.2%}|{r['correct_retention_vs_teacher']:.2%}|")
    lines += ['','## Domain breakdown','', '| Candidate | Domain | Correct matches | Precision | Repeatability |','|---|---|---:|---:|---:|']
    for name,r in reports.items():
        for domain,m in r['state']['best']['per_domain'].items():lines.append(f"|{name}|{domain}|{m['correct_mnn_count']:.2f}|{m['mnn_precision']:.2%}|{m['repeatability']:.2%}|")
    lines += ['','Teacher anchors are in `r1/teacher_dev.json` and `r2/teacher_dev.json`; full per-scene/per-image outcomes, loss trends and DEV curves are in each candidate `report.json`.','', '![Learning curves](learning_curves.png)','', '## Limits and next gate','']
    lines += ['- '+x for x in result['limitations']]
    lines += ['',result['next_gate']+'.','', 'Training usedCPU2threads, workers0, physical batch2images and2-pair gradient accumulation; FP32 CPU means no training VRAM or GPU teacher residency. Atomic best/latest checkpoints andRAM/disk guards allow safe resumption. Process RSS is laptop process memory, not K210 SRAM.','',f"Maximum reported process RSS: {max(r['peak_RSS_MiB'] for r in rows):.1f}MiB."]
    (ROOT/'REPORT.md').write_text('\n'.join(lines)+'\n')
    put(ROOT/'artifacts.sha256.json',{str(f.relative_to(ROOT)):sha(f) for f in sorted(ROOT.rglob('*')) if f.is_file() and f.suffix in ('.pt','.json','.csv','.png','.pdf','.npy','.npz') and f.name not in ['artifacts.sha256.json','run_status.json']})
    put(ROOT/'run_status.json',{'status':'COMPLETE','selected_for_next_gate':winner})
    print(json.dumps(result),flush=True)
if __name__=='__main__':
    try:run()
    except BaseException as e:
        put(ROOT/'run_status.json',{'status':'STOPPED_ERROR','error':repr(e),'resumable':True});raise
