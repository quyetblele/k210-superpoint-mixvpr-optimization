"""Summarize existing paired real-image evidence; never tune on TEST."""
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]

def main():
    teacher_path=ROOT/'superpoint/artifacts/floor6_teacher_baseline/run_001/results.json'
    student_path=ROOT/'superpoint/artifacts/full_training/greedy_regression/results.json'
    teacher=json.loads(teacher_path.read_text())
    student=json.loads(student_path.read_text())
    baseline={r['query']:r for r in teacher['rows']}
    rows=[]
    for arm in ['fp32/greedy','int8/greedy']:
        current={r['query']:r for r in student['floor6'][arm]['rows']}
        if set(current)!=set(baseline):raise ValueError('paired query IDs differ')
        values=list(current.values())
        def percentile(key):
            v=[r[key] for r in values if r.get(key) is not None]
            return float(np.percentile(v,95)) if v else None
        rows.append({'arm':arm,'queries':len(values),
            'good_poses':sum(r['good_pose'] for r in values),
            'mean_inliers':float(np.mean([r['inliers'] for r in values])),
            'mean_correct_matches':float(np.mean([r['correct_matches'] for r in values])),
            'inlier_retention_vs_teacher':sum(r['inliers'] for r in values)/sum(r['inliers'] for r in baseline.values()),
            'rotation_p95_deg':percentile('rotation_deg'),
            'center_p95_native_units':percentile('center_native')})
    log=ROOT/'reports/success_first/student_screening_per_image_bf16_v2.log'
    last={}
    if log.exists():
        for line in log.read_text().splitlines():
            try:r=json.loads(line)
            except json.JSONDecodeError:continue
            if 'identity' in r and 'step' in r:last[r['identity']]=r
    report={'scope':'Read-only audit of existing internal DEV real-image experiments, not a new independent test',
        'source_hashes':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [teacher_path,student_path]},
        'teacher':teacher['summary'],'student':rows,'mixvpr_latest_log_records':last,
        'verdict':'REAL_WORLD_STABILITY_NOT_ESTABLISHED',
        'limitations':['20 internal DEV queries whose observations occur in the reconstruction',
                       'map native units are not verified meters',
                       'fixed retrieval isolates local features, not current MixVPR end-to-end quality',
                       'log records do not establish that a training process is alive'],
        'next_gate':'Complete stable MixVPR DEV screening; evaluate both selected models on an independent calibrated trajectory before architecture/retraining decision.'}
    out=ROOT/'reports/real_readiness';out.mkdir(parents=True,exist_ok=True)
    (out/'paired_floor6_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'student':rows,'training_steps':{k:v['step'] for k,v in last.items()}}))

if __name__=='__main__':main()
