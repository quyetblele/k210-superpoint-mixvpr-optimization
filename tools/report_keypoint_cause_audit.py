"""Render the six-hypothesis audit, retaining all unsuccessful variants."""
import json,hashlib
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'superpoint/artifacts/keypoint_cause_audit'
read=lambda p:json.loads(p.read_text())
a=read(BASE/'run_001/results.json')['summary'];b=read(BASE/'run_003/results.json')['summary']
baseline='| Model | Keypoints/image | Within4px of visible active Point3D | Pose |\n|---|---:|---:|---:|\n'
for name in ['teacher','fp32','int8']:
    r=a[name+'_baseline'];baseline+=f"|{name}|{r['mean_keypoints']:.2f}|{r['mean_geometric_usable_keypoints_4px']:.2f}|{r['good_poses']}/20|\n"
table='| Intervention | FP32 poses | INT8 poses |\n|---|---:|---:|\n'
for key in ['baseline','greedy_nms','threshold_001','content_border','spatial_grid','subpixel_geometry','subpixel_resample','sampling_center','mapping_legacy','gftt']:
    table+=f"|{key}|{a['fp32_'+key]['good_poses']}/20|{a['int8_'+key]['good_poses']}/20|\n"
for key in ['gftt_common','gftt_common_subpixel']:
    table+=f"|{key}|{b['fp32_'+key]['good_poses']}/20|{b['int8_'+key]['good_poses']}/20|\n"
recovery={k:b[k+'_gftt_common_subpixel'] for k in ['fp32','int8']}
report='''# Keypoint failure: six-hypothesis diagnosis and partial recovery

**Main measured limitation: too few accurately placed, map-supported student keypoints.** Total count is not the issue. FP32 fails before quantization; INT8 adds severe local clustering but fixing that alone does not solve localization. A common-resolution classical corner detector with subpixel refinement recovers some poses, without retraining or modifying descriptors/map. It is not a completed deployment solution.

'''+baseline+'''
The4px criterion is in the original camera coordinate system, about0.67network pixels for these source/input sizes. “Useful” here means geometric proximity to a projected active Point3D also observed in the query reconstruction, not verified descriptor correspondence or externally measured visibility. Existing PnP thresholds were not loosened.

## Six hypotheses

| Hypothesis | Evidence | Interpretation |
|---|---|---|
|1. Score distribution differs|Mean per-image selected-score median: teacher0.1284, FP32 student0.00627, INT80.00848. FP32 has284.75 post-NMS candidates and selects160.|Confirmed score/ranking difference. But even before top160, only3.6 FP32 candidates/image are within4px of supported projections. Changing score cutoff/ranking alone is unlikely to recover adequate geometry. Low score itself is not a proof of training cause.|
|2. NMS/border/threshold|FP32 clustering0%; INT885.47% have another selected point within radius4. Greedy NMS, score.001 and un-eroded valid-content mask each still yield0/20. Only3.8 projected supported tracks/image are masked on average.|INT8 NMS pathology is real, not the primary explanation for FP32 failure. More points or more uniform grid coverage do not guarantee map-supported points.|
|3. Resize/coordinate mapping|Stored affine agrees with independent composition of two pixel-center resizes/padding for all54images. Corrected COLMAP and legacy-offset control both leave student0/20. Teacher uses the same path and reaches16/20.|No observed gross shared resize/axis mapping bug explaining student failure. This validates the checked algebra, not all lens models or external map accuracy.|
|4. Too few useful keypoints|Student selects160points; only2.7 FP32 /3.3 INT8 meet4px geometry proximity, versus16.1 for teacher with111.05 points. Even geometry-assisted unique associations yield only1/20 FP32 and0/20 INT8 poses.|Strong evidence that student location quality/support is a bottleneck before descriptor matching. PnP cannot generally recover from too few usable locations. Oracle is diagnostic, not a mathematical upper bound.|
|5. Descriptor sampling at student points|Using HLoc center-consistent sampler for BOTH query and reference still gives0/20. Refining native positions and optionally resampling also gives0/20.|Sampling convention alone does not repair current detector locations. This does not rule out sampling/descriptor interactions after detection is improved.|
|6. Insufficient Point3D overlap|All arms use identical coverage-audited reference lists and active IDs. Teacher16/20, student0/20. Student points miss supported projections; changing only locations recovers some poses.|Distinguish database coverage from keypoint-to-map support. The original contiguous-reference split was coverage-poor; in this covered test the remaining failure cannot be explained by retrieval coverage alone. Old-track map representation remains a limitation.|

Teacher/FP32/INT8 nearest-projection distance CDF and score histograms: `run_001/diagnostic_curves.png`. A median-distance or grid-coverage score alone is misleading: a method may add enough precise correspondences to estimate pose while also selecting many far-away points.

## All tested interventions

'''+table+'''
All configurations were recorded before their respective run. The second-stage common-resolution tests were chosen after the first-stage evidence, so this is adaptive exploratory DEV work, not independent validation or an unbiased estimate of the selected remedy. No failed variant or query was removed.

## Concrete partial remedy

Reuse the existing288×512 grayscale intermediate BEFORE downsampling to the184×320 network input. Run Shi–Tomasi(max160,quality.01,minDistance6,block3), then cornerSubPix(win3,max30iter,eps.01). Reject shifts over3intermediate pixels and shifts outside the unchanged valid mask. Transform these coordinates through the exact second resize/padding, and sample the existing student descriptor map there. Query/reference descriptor donor identity, reference bank, matcher, PnP, model weights and kmodel remain fixed.

Final masked variant: FP32 '''+str(recovery['fp32']['good_poses'])+'''/20, faithful INT8 '''+str(recovery['int8']['good_poses'])+'''/20. Mean geometrically useful keypoints: '''+f"{recovery['fp32']['mean_geometric_usable_keypoints_4px']:.2f}"+'''. Geometry-oracle pose count with those positions: '''+str(recovery['fp32']['oracle_good_poses'])+'''/20. This separates a recoverable detection/location problem from remaining descriptor/matching limitations. It does not prove an ideal detector upper bound.

This remedy uses CPU image processing and preserves the neural graph; it may cost extra CPU time and requires retaining the intermediate image. No board latency/RAM claim is made. The native-image subpixel variants in the first stage require original high-resolution images and are not assumed available on the final camera path.

`run_002` is retained as an intermediate attempt: its subpixel update allowed36 keypoints across both student backends to cross the initial eroded mask. `run_003` rejects those updates, restoring the original mask contract; use run003 for the recovery result. No weights or production defaults were changed to hide this correction.

## What is not yet established

These tests identify a detector/location limitation, not its exact training cause. They do not distinguish insufficient training from compact-network capacity, initialization or supervision weighting. Descriptor KD lambda1 was already inconclusive/unhelpful; more of that training is not justified by this location audit. Pose remains below the teacher reference even after partial recovery; matching/descriptors and the old-track bank can still limit it.

No independent TEST, public benchmark reproduction, final map reconstruction or board qualification was completed. Native map units remain non-metric. All experiments use the same20 nearby DEV queries; no statistical significance claim is made.

## Next decision

Keep the neural recipe/model frozen. Use the common-resolution subpixel detector as an explicitly named CPU recovery candidate, not a promoted default. The next bounded comparison should isolate descriptor/matching behavior at THESE fixed improved locations before another training change; sufficient geometric locations are now available in many more queries. Production integration/final training should wait for useful own-pipeline pose performance, rather than a keypoint count target.

## Reproduce

```bash
./superpoint/run keypoint-audit --output superpoint/artifacts/keypoint_cause_audit/NEW_AUDIT
./superpoint/run keypoint-audit --followup --output superpoint/artifacts/keypoint_cause_audit/NEW_RECOVERY
```

Each run requires a new directory. `run_001/evaluated_source.py` preserves the exact initial audit, `run_003/evaluated_source.py` the corrected recovery. Read their protocols, results, per-query CSV, keypoint/match assignments and checks. Source images, raw head tensors, map and prior experiment identities are checked rather than regenerated or overwritten.
'''
(BASE/'REPORT.md').write_text(report)
(BASE/'recovery_recipe.json').write_text(json.dumps({'status':'PARTIAL_RECOVERY_NOT_PROMOTED','model':'current A unchanged','detector':'gftt_common_subpixel','implementation':'superpoint/scripts/keypoint_cause_audit.py','evidence_run':'run_003','FP32_good_poses':recovery['fp32']['good_poses'],'INT8_good_poses':recovery['int8']['good_poses'],'queries':20,'board_performance':'NOT_MEASURED','test':'DEV exploratory'},indent=2)+'\n')
p=ROOT/'docs/EXPERIMENT_STATUS.md';old=p.read_text()
intro='''# Experiment status — keypoint root-cause audit

Latest: [six-hypothesis audit and partial recovery](../superpoint/artifacts/keypoint_cause_audit/REPORT.md).

Student has160 keypoints but only2.7 FP32 /3.3 INT8 geometrically useful at4px, versus16.1 teacher. Mapping algebra passes; threshold/border/NMS/sampling changes alone do not recover pose. CPU Shi–Tomasi on288×512 plus masked subpixel refinement gives '''+str(recovery['fp32']['good_poses'])+'''/20 FP32 and '''+str(recovery['int8']['good_poses'])+'''/20 INT8 (previously0/20). This is PARTIAL RECOVERY on exploratory DEV, not release-ready.

Model, training recipe, calibration, production map and MixVPR unchanged. Next: investigate descriptor/matching at fixed improved locations; no automatic training or promotion. PC-first / Sipeed Maix Bit target remains unchanged.

Previous gates below are historical context.

'''
if not old.startswith('# Experiment status — keypoint root-cause audit'):p.write_text(intro+old)
print(BASE/'REPORT.md')
