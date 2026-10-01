import _bootstrap
import argparse, subprocess, sys, fcntl
from spk210.settings import ROOT, CANDIDATES, candidate_name, parse_candidate
from spk210.protocol import verify_cache
from spk210.io import put
p = argparse.ArgumentParser(description='Matched400-update SuperPoint pilot; sequential CPU training')
p.add_argument('--candidate')
args = p.parse_args()
verify_cache()
if args.candidate:
    from spk210.training import train
    lock = (ROOT / 'run.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    train(*parse_candidate(args.candidate))
else:
    try:
        for (width, res) in CANDIDATES:
            name = candidate_name(width, res)
            put(ROOT / 'run_status.json', {'status': 'RUNNING', 'candidate': name})
            with (ROOT / (name + '.log')).open('a') as log:
                subprocess.run([sys.executable, __file__, '--candidate', name], stdout=log, stderr=subprocess.STDOUT, check=True)
        from spk210.reporting import report
        report()
    except BaseException as ex:
        put(ROOT / 'run_status.json', {'status': 'STOPPED_ERROR', 'error': repr(ex), 'resumable': True})
        raise
