"""Sequential deployment and natural-view evaluation for the two frozen KD arms."""
import _bootstrap
import json,subprocess,time
from pathlib import Path
from spk210.settings import SUPERPOINT,WORKSPACE
from spk210.io import sha,put

dest=SUPERPOINT/'artifacts/descriptor_kd_ab'
names=['desc_control_r2_cw','desc_kd_r2_cw']
deadline=time.monotonic()+1200
while not all((dest/n/'report.json').exists() for n in names):
    if time.monotonic()>deadline:raise RuntimeError('Training reports not complete within bounded wait')
    time.sleep(5)
reports={n:json.loads((dest/n/'report.json').read_text()) for n in names}
assert all(r['updates']==400 and r['status']=='COMPLETE' for r in reports.values())
assert len({r['initial_sha256'] for r in reports.values()})==len({r['plan_sha256'] for r in reports.values()})==1
assert reports[names[0]]['control_exact_original_weights']
for n in names:
    for stage in ['export','validate','calibrate','compile','simulate','evaluate']:
        put(dest/'status.json',{'status':'RUNNING','candidate':n,'stage':stage})
        with (dest/f'{n}_{stage}.log').open('w') as f:
            subprocess.run([str(SUPERPOINT/'run'),stage,'--candidate',n],stdout=f,stderr=subprocess.STDOUT,check=True)
        print('PASS',n,stage,flush=True)
    for stage,python in [('simulate','/home/quyet/miniconda3/envs/k210/bin/python'),('evaluate',str(WORKSPACE/'.venv-mixvpr-cuda/bin/python'))]:
        put(dest/'status.json',{'status':'RUNNING','candidate':n,'stage':'natural_'+stage})
        with (dest/f'{n}_natural_{stage}.log').open('w') as f:
            subprocess.run([python,str(SUPERPOINT/'scripts/descriptor_kd_evaluate.py'),stage,'--candidate',n],stdout=f,stderr=subprocess.STDOUT,check=True)
        print('PASS',n,'natural_'+stage,flush=True)
put(dest/'status.json',{'status':'MEASUREMENTS_COMPLETE','training_protocol_sha256':sha(dest/'protocol.json')})
