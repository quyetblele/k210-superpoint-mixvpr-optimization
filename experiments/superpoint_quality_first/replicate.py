"""Replicate the fixed A/B by training-data-order seed; same warm-start weights."""
import argparse,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).parent
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--seed',type=int);args=parser.parse_args()
    if args.seed is not None:
        import train_detector_ab as ab
        ab.SEED=args.seed;ab.DEST=ROOT/'replicates'/('seed_'+str(args.seed));ab.run()
    else:
        folder=ROOT/'replicates';folder.mkdir(exist_ok=True)
        protocol={'seeds':[20260914,20260915,20260916],'first_seed_already_observed':True,'additional_seeds_locked_before_running':True,'shared_init':'original SUP pilot best step400, identical for all replicas','variation':'training pair order only; not independent initialization seeds','recipe':'unchanged train_detector_ab.py, 400 added updates per branch','gate':'mean precision improvement>=0.03, positive precision improvement in all3 seeds, mean teacher-detector precision>=0.50 and mean repeatability drop <=0.02','selection':'per-run best on existing DEV; no new INT8 evaluation selects recipe','scope':'exploratory replication on same 28 DEV homography pairs; not independent test'}
        path=folder/'protocol.json'
        if path.exists():assert json.loads(path.read_text())==protocol
        else:path.write_text(json.dumps(protocol,indent=2)+'\n')
        for seed in protocol['seeds'][1:]:
            with (folder/('seed_'+str(seed)+'.log')).open('a') as log:
                subprocess.run([sys.executable,__file__,'--seed',str(seed)],stdout=log,stderr=subprocess.STDOUT,check=True)
            print('REPLICATE_COMPLETE',seed,flush=True)
        rows=[]
        for seed in protocol['seeds']:
            path=ROOT/'detector_ab/summary.json' if seed==20260914 else folder/('seed_'+str(seed))/'summary.json'
            d=json.loads(path.read_text());rows.append({'seed':seed,**{k:d[k] for k in ['control_best','teacher_detector_best','precision_delta','gate_pass','control_best_step','teacher_detector_best_step']}})
        def mean(branch,key):return sum(r[branch][key] for r in rows)/len(rows)
        delta=mean('teacher_detector_best','mnn_precision')-mean('control_best','mnn_precision')
        passed=delta>=.03 and all(r['precision_delta']>0 for r in rows) and mean('teacher_detector_best','mnn_precision')>=.5 and mean('teacher_detector_best','repeatability_3px')>=mean('control_best','repeatability_3px')-.02
        result={'status':'COMPLETE','rows':rows,'mean_precision_delta':delta,'mean_control_precision':mean('control_best','mnn_precision'),'mean_teacher_detector_precision':mean('teacher_detector_best','mnn_precision'),'gate_pass':passed,'scope':protocol['scope'],'shared_initialization':True,'deployment_checkpoint':'first seed remains fixed; no best-of-three checkpoint cherry-pick'}
        (folder/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
