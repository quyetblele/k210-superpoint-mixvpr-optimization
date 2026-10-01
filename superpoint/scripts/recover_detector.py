import _bootstrap
import argparse,subprocess,sys,fcntl
from spk210.detector_recovery import DEST,BRANCHES,setup,train
from spk210.io import put
p=argparse.ArgumentParser();p.add_argument('--candidate',choices=BRANCHES);a=p.parse_args()
setup()
if a.candidate:
    lock=(DEST/'run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    train(a.candidate)
else:
    try:
        for name in BRANCHES:
            put(DEST/'status.json',{'status':'TRAINING','candidate':name})
            with (DEST/(name+'.log')).open('a') as log:
                subprocess.run([sys.executable,__file__,'--candidate',name],stdout=log,stderr=subprocess.STDOUT,check=True)
        put(DEST/'status.json',{'status':'TRAINING_COMPLETE_DEPLOYMENT_PENDING'})
    except BaseException as ex:
        put(DEST/'status.json',{'status':'STOPPED_ERROR','error':repr(ex)});raise
