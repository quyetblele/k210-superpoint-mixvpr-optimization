"""Run each frozen recovery checkpoint through the canonical deployment stages."""
import _bootstrap
import json,subprocess,hashlib,csv
from pathlib import Path
from spk210.settings import SUPERPOINT,ROOT
from spk210.io import put,sha

dest=SUPERPOINT/'artifacts/detector_recovery'
protocol=json.loads((dest/'protocol.json').read_text())
names=list(protocol['branches'])
reports={name:json.loads((dest/name/'report.json').read_text()) for name in names}
assert all(r['state']['step']==400 and r['frozen_parameters_exact'] for r in reports.values())
assert len({r['plan_sha256'] for r in reports.values()})==1
assert len({r['parent_sha256'] for r in reports.values()})==1
assert reports[names[0]]['state']['evaluations'][0]==reports[names[1]]['state']['evaluations'][0]
results={}
for name in names:
    deployment=SUPERPOINT/'artifacts/deployment'/name
    try:
        quality=deployment/'int8_quality.json'
        if quality.exists():
            q=json.loads(quality.read_text())
            assert q['checkpoint_sha256']==sha(dest/name/'best.pt') and q['kmodel_sha256']==sha(deployment/'model.kmodel')
        else:
            for stage in ['export','validate','calibrate','compile','simulate','evaluate']:
                put(dest/'status.json',{'status':'DEPLOYMENT','candidate':name,'stage':stage})
                with (dest/f'{name}_{stage}.log').open('w') as log:
                    subprocess.run([str(SUPERPOINT/'run'),stage,'--candidate',name],stdout=log,stderr=subprocess.STDOUT,check=True)
            q=json.loads(quality.read_text())
        results[name]={'status':'MEASURED','quality':q,'compile':json.loads((deployment/'compile.json').read_text()),'parity':json.loads((deployment/'onnx_parity.json').read_text()),'best_step':reports[name]['state']['best_step']}
    except Exception as ex:
        results[name]={'status':'BLOCKED','error':str(ex)}
        put(dest/'partial_deployment.json',results)
if any(r['status']!='MEASURED' for r in results.values()):
    put(dest/'status.json',{'status':'BLOCKED','results':results});raise SystemExit(1)
control=results[names[0]]['quality'];balanced=results[names[1]]['quality']
c=control['int8']['macro'];b=balanced['int8']['macro']
passed=(b['correct_mnn_count']>=c['correct_mnn_count']*1.1 and balanced['fp32']['macro']['correct_mnn_count']>=control['fp32']['macro']['correct_mnn_count']*.95 and b['mnn_precision']>=c['mnn_precision']-.02)
summary={'status':'COMPLETE','gate_pass':passed,'gate':protocol['interpretation_gate'],'results':results,'fairness':'same parent/plan/update budget; equal initial DEV; backbone+descriptor tensors unchanged','GT1_used':False,'next_gate':'replicate successful recipe before promoting' if passed else 'retain parent; foreground-balanced detector KL did not pass this pilot; measure quantization-induced detector peak/ranking instability before another training recipe'}
put(dest/'summary.json',summary)
rows=[]
for name,r in results.items():
    for backend in ['fp32','int8']:
        rows.append({'candidate':name,'best_step':r['best_step'],'backend':backend,**r['quality'][backend]['macro']})
with (dest/'comparison.csv').open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
lines=['# Detector-only recovery A/B','',f'Experimental gate: **{"PASS" if passed else "FAIL"}**. {summary["fairness"]}.','',
       '| Candidate | Selected update | Backend | Correct matches | Precision | Repeatability |','|---|---:|---|---:|---:|---:|']
for r in rows:lines.append(f"|{r['candidate']}|{r['best_step']}|{r['backend']}|{r['correct_mnn_count']:.2f}|{r['mnn_precision']:.2%}|{r['repeatability']:.2%}|")
lines += ['', 'Control is uniform teacher detector KL. Treatment averages uniform KL and teacher-foreground-weighted KL. Only convPa/convPb train; no QAT, architecture change, descriptor/backbone update, GT1 use or teacher residency.400updates per branch; checkpoint chosen by FP32 DEV, then actual trained-weight PTQ/nncase simulator evaluation.','',
          'The gate was fixed before training: INT8 correct matches >=control*1.10, FP32 correct >=control*0.95, INT8 precision>=control-0.02. This is an experimental signal criterion, not board-release qualification.','',
          f'Maximum training process RSS: {max(r["peak_RSS_MiB"] for r in reports.values()):.1f}MiB. CPU2threads, workers0, physical2images, accumulation2pairs.','',summary['next_gate'], '',
          'Teacher probabilities and normalized loss weights are derived from TRAIN caches only. DEV remains synthetic-homography engineering data from related sources, not independent localization proof. See each branch progress.json for full learning/loss curves and per-scene outcomes.']
(dest/'REPORT.md').write_text('\n'.join(lines)+'\n')
put(dest/'status.json',{'status':'COMPLETE','gate_pass':passed,'next_gate':summary['next_gate']})
print(json.dumps({'gate_pass':passed,'comparison':rows}),flush=True)
