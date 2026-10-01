"""Summarize the locked descriptor-KD pilot without selecting further hyperparameters."""
import csv,hashlib,json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'superpoint/artifacts/descriptor_kd_ab'
NAMES=['desc_control_r2_cw','desc_kd_r2_cw']


def rd(p):return json.loads(p.read_text())


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()


rows=[];data={}
for name in NAMES:
    train=rd(BASE/name/'report.json');natural=rd(BASE/name/'natural/results.json');assert natural['status']=='COMPLETE'
    dep=ROOT/'superpoint/artifacts/deployment'/name
    quality=rd(dep/'int8_quality.json');compile_spec=rd(dep/'compile.json');parity=rd(dep/'onnx_parity.json')
    assert train['checkpoint_sha256']==quality['checkpoint_sha256']==compile_spec['checkpoint_sha256']==parity['checkpoint_sha256']
    assert quality['kmodel_sha256']==compile_spec['kmodel_sha256']==sha(dep/'model.kmodel')
    assert parity['status']=='PASS'
    data[name]={'training':train,'natural':natural,'quality':quality,'compile':compile_spec}
    for backend in ['fp32','int8']:
        n=natural['results'][backend]
        rows.append({'arm':'A' if name==NAMES[0] else 'B','backend':backend,'selected_step':train['best_step'],
                     'descriptor_top1':n['descriptor']['top1'],'descriptor_top5':n['descriptor']['top5'],
                     'fixed_keypoint_good_poses':n['fixed']['good_poses'],'own_keypoint_good_poses':n['own']['good_poses'],
                     'own_mean_inliers':n['own']['mean_inliers'],'synthetic_correct_MNN':quality[backend]['macro']['correct_mnn_count'],
                     'synthetic_precision':quality[backend]['macro']['mnn_precision']})
a=data[NAMES[0]];b=data[NAMES[1]]
assert a['training']['control_exact_original_weights']
assert a['training']['initial_sha256']==b['training']['initial_sha256']
assert a['training']['plan_sha256']==b['training']['plan_sha256']
assert a['training']['updates']==b['training']['updates']==400
assert a['compile']['calibration_sha256']==b['compile']['calibration_sha256']
for path,digest in rd(BASE/'source_hashes.json').items():
    if path=='superpoint/src/spk210/onnx_export.py' and (BASE/'parity_checker_amendment.json').exists():
        amendment=rd(BASE/'parity_checker_amendment.json')
        assert amendment['original_sha256']==digest==sha(BASE/'source/onnx_export.py')
        assert sha(ROOT/path)==amendment['updated_sha256']
    else:assert sha(ROOT/path)==digest,path
# The control must replay the preceding fixed-keypoint descriptor intervention.
prior=rd(ROOT/'superpoint/artifacts/fixed_keypoint_pnp/run_001/summary.json')
for backend,label in [('fp32','student_fp32'),('int8','student_int8')]:
    assert a['natural']['results'][backend]['fixed']['good_poses']==prior['arms'][label]['good_poses']
improves=all(b['natural']['results'][k]['descriptor']['top1']>a['natural']['results'][k]['descriptor']['top1'] for k in ['fp32','int8'])
nonregress=all(b['natural']['results'][k][mode]['good_poses']>=a['natural']['results'][k][mode]['good_poses'] for k in ['fp32','int8'] for mode in ['fixed','own'])
own_gain=any(b['natural']['results'][k]['own']['good_poses']>a['natural']['results'][k]['own']['good_poses'] for k in ['fp32','int8'])
keep=improves and nonregress and own_gain
decision='PROVISIONAL_KEEP_PENDING_REPLICATION' if keep else 'DO_NOT_ADD_TO_FINAL_RECIPE_YET'
next_decision=('Keep this KD setting provisionally and confirm the controlled result with a second seed before final/full training.' if keep else
               'Keep A as the unchanged baseline; do not start full training or add KD to the final recipe from this pilot. Use the fixed-versus-own-keypoint evidence to resolve the remaining localization limitation before another training intervention.')
summary={'status':'COMPLETE','decision':decision,'locked_criterion':rd(BASE/'protocol.json')['retention_decision'],
         'criterion_checks':{'descriptor_top1_improved_both_precisions':improves,'pose_nonregression_all_modes':nonregress,'own_keypoint_pose_gain':own_gain},
         'comparison':rows,'one_next_decision':next_decision,'board':'NOT_MEASURED','test':'NOT_ESTABLISHED','full_training':False}
(BASE/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
with (BASE/'comparison.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
table='| Arm | Backend | Selected update | Natural descriptor top1 | Fixed-teacher-keypoint PnP | Own-keypoint PnP | Synthetic correct MNN |\n|---|---|---:|---:|---:|---:|---:|\n'
for r in rows:
    table+=f"|{r['arm']}|{r['backend']}|{r['selected_step']}|{r['descriptor_top1']:.2%}|{r['fixed_keypoint_good_poses']}/20|{r['own_keypoint_good_poses']}/20|{r['synthetic_correct_MNN']:.2f}|\n"
report='''# Descriptor KD controlled A/B — completed short pilot

Decision: **'''+decision+'''**. This is a one-seed experimental decision, not a deployment release criterion.

'''+table+'''
## Controlled change

A reproduces the current recipe. B adds lambda1.0 mean cosine descriptor loss at exactly the same TRAIN correspondence points used by the existing InfoNCE. Teacher descriptors are FP32, frozen and cached from TRAIN only. Both arms use the original wider_init checkpoint, original400-step plan/seed, architecture, all trainable parameters, detector KL, data/augmentation, AdamW/cosine schedule, accumulation, gradient clipping and DEV checkpoint selection. No KD-weight search, new data, QAT, MixVPR or NMS change.

The control's selected model reproduces every tensor of the original current recipe exactly. Checkpoint serialization hashes differ because protocol metadata differs. Source snapshots/hashes, initial/plan identities and per-step curves are retained. Both candidates use identical calibration tensor bytes, ONNX qualification and fresh actual nncase compile/simulation. The synthetic and natural INT8 results are real kmodel outputs, not fake quantization. No candidate was promoted.

ONNX checker amendment: B initially failed strict keypoint ARRAY ORDER on TRAIN probe79. Diagnostics found identical keypoint sets with a near-tie ordering permutation. The checker now aligns coordinate/descriptor pairs before comparing, retaining the original2e-4 tolerances and strict-order outcome in reports. Exact42DEV metric equality remains mandatory. A and B were both validated with the same amended checker. Decoder, training, thresholds and evaluation metrics were not changed. Original failed log, numerical diagnosis, original source snapshot and `parity_checker_amendment.json` are preserved. This qualifies sparse feature pairs; order-sensitive downstream algorithms must not assume bitwise ordering across numerical backends.

## Evaluation meaning

Natural descriptor top1 uses the same20 nearby-view oracle track pairs, up to256 common queries/512 reference observations, explicit −0.5 coordinate conversion and duplicate-track exclusions as the preceding gate. It bypasses detector selection. Fixed-keypoint PnP uses the unchanged teacher query locations; own-keypoint PnP uses each candidate's original-NMS keypoints and its own descriptors. Reference geometry, active Point3D IDs, retrieval lists, matching thresholds and solver remain common. Each donor has its own query/reference descriptor space.

All20 localization queries are retained, including failures. PnP requires>=8 inliers,<=5degrees and<=.1 native map units, with100iterations and4-pixel residual. Native units are not established meters. Full per-query pose errors, inliers, geometric correct-match proxies and failures are saved in each candidate's `natural/results.json`. Old map observation resampling is not feature-specific reconstruction. Fixed-keypoint results are hybrid diagnostics, not the deployable student.

The actual KP/NMS outputs are allowed to change indirectly because KD updates the shared backbone; only the detector supervision and postprocessing recipes are held fixed. An observed gain cannot be attributed exclusively to one isolated head.

## Evidence and uncertainty

Retention criterion was recorded before results: descriptor ranking improves, neither fixed nor own localization regresses in FP32/INT8, and at least one own-keypoint pose count improves. Criterion checks: '''+json.dumps(summary['criterion_checks'])+'''.

Public benchmark reproduction and untouched TEST remain incomplete. One seed, one KD coefficient,400updates and nearby DEV queries cannot establish convergence, statistical robustness or universal effectiveness of descriptor KD. A negative result rejects promotion of this setting, not all possible descriptor KD designs. A positive hybrid result alone does not establish deployment usefulness. Board I/O, latency and RAM remain unmeasured.

## ONE next decision

'''+next_decision+'''

## Reproduction and files

From workspace root, `PYTHONDONTWRITEBYTECODE=1 .venv-mixvpr-cuda/bin/python superpoint/scripts/descriptor_kd_ab.py` prepares/resumes this locked pilot and skips completed training branches. `run_descriptor_kd_validation.py` reruns deployment/evaluation stages into these experiment-specific candidate directories; it does not train. Do not use it to preserve byte-for-byte historical log files; archive the run before intentional re-evaluation. No production map/default candidate is changed.

Read `protocol.json`, `source_hashes.json`, `comparison.csv`, `summary.json`, per-arm `progress.json`, per-arm `natural/{protocol,simulation,results}.json` and per-candidate deployment reports. `source/` stores exact source snapshots.
'''
(BASE/'REPORT.md').write_text(report)
status=ROOT/'docs/EXPERIMENT_STATUS.md';old=status.read_text()
intro='''# Experiment status — descriptor KD A/B

Latest: [completed descriptor KD pilot](../superpoint/artifacts/descriptor_kd_ab/REPORT.md).

'''+table+'\nDecision: **'+decision+'**. '+next_decision+'''

No full training/QAT/architecture/MixVPR change. User preference remains PC preparation before Sipeed Maix Bit board trials. Public benchmark/untouched TEST and hardware qualification remain incomplete.

Previous gates below are historical context; their proposed descriptor-KD A/B has now been executed above.

'''
if not old.startswith('# Experiment status — descriptor KD A/B'):status.write_text(intro+old)
put_status={'status':'COMPLETE','decision':decision,'one_next_decision':next_decision}
(BASE/'status.json').write_text(json.dumps(put_status,indent=2)+'\n')
print(json.dumps(summary,indent=2))
