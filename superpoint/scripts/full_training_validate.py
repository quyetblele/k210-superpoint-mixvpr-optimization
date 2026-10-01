"""Checkpoint-bound PC deployment validation; never evaluates the sealed TEST."""
import _bootstrap
import os,subprocess,json
from pathlib import Path
from spk210.settings import SUPERPOINT,WORKSPACE
from spk210.io import sha,put
OUT=SUPERPOINT/'artifacts/full_training';NAME='full_capacity_r2_cw'

def run():
    report=json.loads((OUT/'report.json').read_text());assert report['status']=='TRAINING_COMPLETE'
    checkpoint=sha(OUT/'best.pt');assert checkpoint==report['checkpoint_sha256']
    env=dict(os.environ,SP_ORT_PARITY_OPTIMIZATION='disabled',SP_STUDENT_CANDIDATE=NAME)
    for stage in ['export','validate','calibrate','compile','simulate','evaluate','floor6_simulate','floor6_evaluate']:
        marker=OUT/f'{stage}.pass.json'
        if marker.exists():assert json.loads(marker.read_text())['checkpoint_sha256']==checkpoint;continue
        if stage.startswith('floor6_'):
            action=stage.split('_',1)[1];python='/home/quyet/miniconda3/envs/k210/bin/python' if action=='simulate' else str(WORKSPACE/'.venv-mixvpr-cuda/bin/python');cmd=[python,str(SUPERPOINT/'scripts/floor6_student_evaluate.py'),action,'--output',str(OUT/'floor6')]
        else:cmd=[str(SUPERPOINT/'run'),stage,'--candidate',NAME]
        with (OUT/f'{stage}.log').open('w') as f:subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
        put(marker,{'status':'PASS','checkpoint_sha256':checkpoint});print('PASS',stage,flush=True)
    dep=SUPERPOINT/'artifacts/deployment'/NAME
    assert sha(dep/'calibration_train.npy')==sha(SUPERPOINT/'artifacts/deployment/rank_control_r2_cw/calibration_train.npy')
    put(OUT/'deployment_summary.json',{'status':'PC_STAGES_PASS','checkpoint_sha256':checkpoint,'compile':json.loads((dep/'compile.json').read_text()),'quality':json.loads((dep/'int8_quality.json').read_text())['macro_retention'],'floor6':json.loads((OUT/'floor6/results.json').read_text())['summary'],'test_evaluated':False,'board_measured':False})
if __name__=='__main__':run()
