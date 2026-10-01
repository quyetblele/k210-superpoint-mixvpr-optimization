"""Render reviewable reports from a completed measurement gate; never runs training."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text())


def sha(path):
    d=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): d.update(b)
    return d.hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run',type=Path)
    args=parser.parse_args(); out=args.run.resolve()
    assert out.is_relative_to(ROOT/'superpoint/artifacts/measurement_gate')
    assert read(out/'completion.json')['status']=='DIAGNOSTIC_COMPLETE'
    m=read(out/'manifest.json');f=read(out/'feature_comparison.json');n=read(out/'natural_descriptor_probe.json');l=read(out/'localization_validation.json')
    for path,digest in m['source_hashes'].items(): assert sha(Path(path))==digest,path
    shutil.copyfile(ROOT/'superpoint/scripts/measurement_gate.py',out/'evaluated_source.py')
    columns=['teacher_fp32','student_fp32','student_int8']
    table='| Model | Keypoints | MNN | Correct MNN | Precision | Repeatability* | Zero-correct pairs |\n|---|---:|---:|---:|---:|---:|---:|\n'
    for name in columns:
        a=f[name]['macro']
        table+=f"| {name} | {a['mean_keypoints']:.2f} | {a['mnn_count']:.2f} | {a['correct_mnn_count']:.2f} | {100*a['mnn_precision']:.2f}% | {100*a['repeatability']:.2f}% | {f[name]['zero_correct_pairs']}/42 |\n"
    natural='| Model | Top-1 legacy | Top-1 −0.5 px | Top-5 legacy |\n|---|---:|---:|---:|\n'
    for name in columns:
        a=n['summary'][name+'_offset_0.0'];b=n['summary'][name+'_offset_-0.5']
        natural+=f"| {name} | {100*a['top1']:.2f}% | {100*b['top1']:.2f}% | {100*a['top5']:.2f}% |\n"
    (out/'EVALUATOR_VALIDATION.md').write_text('''# Evaluator validation — INCONCLUSIVE overall

New measurements: pinned SuperGlue/SuperPoint upstream forward and the raw-head decoder agree on keypoint coordinates and normalized descriptors for six views from three fixed DEV pairs. Descriptor maximum error must be <=2e-6. Upstream comparison uses an all-valid mask and remove_borders=0 to isolate decoding; it does not certify the custom overlap-mask policy.

Independent permutation, wrong-correspondence, empty-feature and projective-coordinate checks pass. All 42 pair-cache and real simulator-output hashes pass. FP32 and INT8 per-pair metrics exactly reproduce the prior stored results.

**Public benchmark reproduction: NOT_RUN.** No HPatches dataset was found in the inventory's explicitly listed search roots/depth. The previous literature audit proposed public reproduction but did not freeze a particular executable public evaluator. No published score is used as a pass threshold here. This gate may support engineering diagnosis only; it is not paper-level benchmark qualification.

Pilot differences: custom source images and warps, threshold .005, top160, original max-pool NMS radius4, original upstream descriptor sampling, mutually visible masks, MNN and <=3 pixels in canonical W144×H256. Repeatability is one-way nearest-keypoint proximity, not a one-to-one or symmetric public definition. Precision has a zero convention for no matches; all 42 pairs remain in the denominator. No homography-estimation metric is claimed.

Canonical-pixel error is not original-camera error. The natural-track probe explicitly reports both legacy coordinates and COLMAP −0.5-pixel conversion; no coordinate default was changed. Full upstream end-to-end localization equivalence remains INCONCLUSIVE.

Evidence: `evaluator_validation.json`, `manifest.json`, `feature_pairs.csv`, `evaluated_source.py`.
''')
    (out/'FEATURE_COMPARISON.md').write_text('''# Feature comparison — measured DEV diagnostics

All three models use identical R2 images, masks, descriptor sampling, original NMS, keypoint cap and MNN metric. Figures below are equal-scene macro means, including failures. *Repeatability is the custom asymmetric pilot measure, not a public benchmark score. No training occurred.

'''+table+'''
INT8 comes from previously generated actual nncase kmodel simulator tensors, rehashed and replayed against identical inputs. It is not fake quant and not a fresh simulator execution. FP32/INT8 pair rows reproduce the historical result exactly.

Teacher→student measures the whole trained student recipe, not isolated architecture cost. INT8 loses correct-match count while precision rises; higher precision alone therefore does not mean better localization.

## Natural-view descriptor diagnostic (new)

Detector-free descriptor sampling at existing COLMAP observations, same locations and candidates for all models. Twenty DEV queries are paired with existing DEV support views using maximum shared tracks (geometric oracle). Up to256 common queries and512 reference observations, sorted by Point3D ID. Means below weight image pairs equally. Legacy uses3547 track comparisons; −0.5 uses3556 because boundary validity changes. No empty pairs. These are nearby views, not independent TEST and not all map distractors.

'''+natural+'''
Three DEV images contain one duplicated Point3D ID each (86,95,106). The probe excludes ambiguous track IDs from sampled images and logs exact exclusions; it does not repair the map. Different offset variants have slightly different valid sets, so their difference is a sensitivity diagnostic, not an isolated causal estimate.

The FP32 descriptor gap survives removal of detector selection. The additional INT8 descriptor-ranking gap is small in this diagnostic. This does not establish absolute descriptor sufficiency or explain all pose failures.

Raw data: `feature_comparison.json`, `feature_pairs.csv`, `natural_descriptor_probe.json`, `natural_descriptor_pairs.csv`.
''')
    (out/'LOCALIZATION_VALIDATION.md').write_text('''# Localization validation — INCONCLUSIVE overall

Reused historical evidence was verified against recorded report hashes, protocol/result associations and current geometry hashes. No pose run or full teacher-bank regeneration was performed in this gate.

| Component | Status | Evidence / limitation |
|---|---|---|
| Internal geometry + known-track PnP | PASS, scoped |41/41; median image-median reprojection1.3293 original pixels; not external ground truth|
| Original TRAIN-only reference coverage | Insufficient diagnostic coverage |Historical median query-track coverage13.74%; no universal release threshold implied|
| Nearby-frame covered sanity | Measured historical |Same20queries: teacher low16/20; student FP32 and INT8 both0/20|
| Descriptor input identity in new probe | PASS |Current checkpoint/kmodel/input/raw-output hashes checked; donor stays consistent within each comparison|
| Historical teacher feature bank | INCONCLUSIVE full provenance |Report hashes verified; full bank regeneration/content audit not repeated|
| Map observation uniqueness | Anomalies found |Three DEV images with duplicate track IDs; excluded only in new probe|
| Full upstream coordinate/PnP equivalence | INCONCLUSIVE |Original COLMAP→pixel-index conversion requires end-to-end qualification|
| Public/independent localization benchmark | NOT_RUN |Map includes query observations; nearby support is DEV diagnostic|
| Board online runtime / latency / peak RAM | NOT_MEASURED |No qualified runtime measurement in this gate|

The historical teacher uses original NMS while the students use greedy-corner NMS. The16/20 vs0/20 result is not a network-only comparison. Pose success required >=8 inliers, <=5 degrees, <=0.1 native map units; native units have not been established as meters. No failed query was removed to improve a pose score.

Map resampling at old tracks does not constitute reconstruction with student-specific features. Geometry sanity only supports internal consistency. The new detector-free ranking diagnostic provides complementary evidence; it is not pose performance.

Evidence: `localization_validation.json`, `natural_descriptor_probe.json` and linked hashes of historical files.
''')
    next_experiment='''## ONE NEXT EXPERIMENT — descriptor donor A/B with fixed teacher detections

Purpose: determine whether the measured student descriptor gap translates to pose failure when detector locations are held fixed. No training required.

Use the existing20-query/21-support DEV sanity solely for diagnosis. Fix low-resolution teacher query keypoints for both arms. A samples teacher descriptors for both query and reference observations; B samples current student FP32 descriptors for both. Preserve retrieval, reference coverage, unique Point3D assignments, resolution, matcher and PnP budgets. Create separate diagnostic banks; do not modify the production map or match across descriptor spaces. Verify the COLMAP/pixel-index conversion against upstream before locking the experiment, and exclude/report duplicate observations consistently in both arms.

Report all queries, geometrically correct correspondences, inliers and pose errors/failures; no post-hoc threshold tuning. If A localizes and B collapses, descriptor-recipe intervention has direct downstream support. If B localizes despite the ranking gap, prioritize detector/localization interaction. If both fail, the diagnostic pipeline remains unresolved. No result is predicted and no numerical acceptance threshold is invented.

Status: PROPOSED, NOT EXECUTED. Training/KD/QAT/calibration/NMS changes require the user's choice after this gate. This is one experiment, not authorization for a sequence of interventions.
'''
    (out/'BOTTLENECK_REPORT.md').write_text('''# Bottleneck report

**The current limitation cannot be reduced to INT8 detector quality.** The frozen student already has a measurable FP32 descriptor gap on natural observations. Additional deployed INT8 matching loss exists, but the natural-track descriptor-ranking loss is small.

| Observation | Evidence | Interpretation | Confidence / scope | What would challenge it |
|---|---|---|---|---|
| Student FP32 precision68.57% vs teacher98.63% |New42-pair common-protocol replay|Whole student recipe gap exists|High on this DEV protocol; not architecture isolation|Identity/protocol error or non-reproduction|
| Natural top-1 teacher87.38%, student68.06% |New20-pair known-track probe|Descriptor gap persists without detector selection|High for these oracle nearby views; limited generality|Independent natural pairs or corrected observation contract eliminates the gap|
| Natural INT8 top-1 67.84% vs FP32 68.06% |Same tracks/input/donor-consistent sampling|INT8 adds little descriptor ranking loss here|Moderate; not proof quantization is harmless globally|Wider viewpoints/other domains reveal large loss|
| INT8 correct MNN27.52 vs FP32 65.49 |Replayed real nncase outputs|Additional deployment-path loss is substantial|High for original postprocessing on synthetic DEV|Input/output hash mismatch or failed repeatability|
| Greedy NMS historically recovers much of MNN count |Existing NMS report, secondary evidence|Selection/plateaus contribute; not a new promoted recipe|Prior diagnostic, not rerun in this gate|Mismatch in NMS experiment identities or failure on natural views|
| Covered localization teacher16/20, both students0/20 |Historical verified reports|PTQ alone cannot explain all observed student failure|Moderate; NMS/map/coordinate confounders remain|Matched-postprocessing upstream localization reproduction reverses result|
| Known-track PnP41/41 |Verified geometry evidence|No evidence to justify rebuilding geometry first|Internal consistency only|Independent metric ground truth reveals geometry error|

Still inconclusive: detector contribution to pose failure; descriptor gap as a sufficient cause of zero poses; public benchmark quality; independent TEST; full bank and coordinate qualification; board resources. The protocol does not support a claim that QAT is needed, that training longer will fix it, or that the system is ready.

'''+next_experiment)
    summary='''# Experiment status — SuperPoint measurement gate

Gate run: `superpoint/artifacts/measurement_gate/'''+out.name+'''/`. Diagnostic measurements complete; public benchmark and final system qualification incomplete. No new training, no MixVPR change, no production map update.

## 1. CURRENT CANDIDATE IDENTITY

`wider_r2_cw`, widths24/32/64/96, head128, descriptor256. Checkpoint SHA256 `'''+m['checkpoint_sha256']+'''`; kmodel SHA256 `'''+m['compile']['kmodel_sha256']+'''`. Logical W184×H320, CW90 deployment. Source hashes accompany the dirty/untracked Git state; commit alone is not used as implementation identity.

## 2. DATA SPLIT / DEV / TEST STATUS

140 TRAIN /42 DEV source images for the fixed synthetic pilot. Additional natural-view diagnostic uses existing20DEV queries and21DEV support frames. **TEST = NOT YET ESTABLISHED.** Map reference/query roles are separate from network training roles. Inventory completed before any download; no dataset downloaded or deleted. Pilot evidence retained.

## 3. EVALUATOR VALIDATION RESULT

INCONCLUSIVE overall. Upstream decoder comparison on six views, metric contract checks and42-pair replay PASS. Public benchmark reproduction NOT_RUN. No existing frozen public evaluator was selected; these results retain engineering scope.

## 4–6. TEACHER FP32 / STUDENT FP32 / STUDENT INT8 REAL-NNCASE RESULTS

'''+table+'''
Actual INT8 simulator outputs were reused after identity checks, not regenerated. *Custom pilot repeatability. Full distributions and pair-level failures are in JSON/CSV.

New natural-view known-track descriptor ranking:

'''+natural+'''
## 7. LOCALIZATION / GEOMETRY STATUS

Historical internal geometry PASS41/41; covered teacher16/20 vs student FP32/INT8 0/20. Report/geometry hashes verified; full localization re-execution and full bank qualification not performed. New probe excludes ambiguous duplicate track IDs. Full localization verdict INCONCLUSIVE. Board latency/RAM NOT_MEASURED.

## 8. BOTTLENECK CONCLUSION

FP32 student recipe/descriptor quality gap is present. Additional INT8 loss in synthetic matching is also present; natural descriptor ranking changes little under INT8. Neither observation alone establishes the cause of zero poses. Do not start detector recovery or QAT by default.

## 9. WHAT IS STILL INCONCLUSIVE

Public benchmark performance, independent TEST, detector-versus-descriptor contribution to pose failure, end-to-end coordinate contract, full historical bank provenance, hardware feasibility of the complete online pipeline. No claim of training completion or deployable quality.

## 10. ONE NEXT EXPERIMENT

Fixed teacher keypoints, teacher-descriptor versus student-FP32-descriptor A/B through the same localization diagnostic. Proposed only; details and falsification conditions in BOTTLENECK_REPORT.md.

## 11. EXACT FILES TO READ

All relative to this gate's artifact directory:

- `manifest.json`: frozen identity and artifact chain.
- `EVALUATOR_VALIDATION.md`: pass conditions and public-benchmark limitation.
- `FEATURE_COMPARISON.md`: fresh measurements and scope.
- `LOCALIZATION_VALIDATION.md`: reused evidence and unresolved contracts.
- `BOTTLENECK_REPORT.md`: interpretation and single proposed next experiment.
- `feature_pairs.csv`, `natural_descriptor_pairs.csv`: individual measurements.
- `evaluated_source.py`: exact executed implementation.

Run the gate with `./superpoint/run measure --output superpoint/artifacts/measurement_gate/NEW_RUN_NAME` (new directory required). Render with `python3 tools/report_measurement_gate.py <run-directory>`. Review output before changing current experiment pointers. Output report creation does not train or promote a candidate.
'''
    (out/'REPORT.md').write_text(summary)
    (ROOT/'docs/EXPERIMENT_STATUS.md').write_text(summary.replace('# Experiment status — SuperPoint measurement gate\n', '# Experiment status — SuperPoint measurement gate\n\nDeployment target and order: [DEPLOYMENT_TARGET](DEPLOYMENT_TARGET.md). User requests PC preparation before board trials.\n', 1))
    hashes={p.name:sha(p) for p in out.iterdir() if p.is_file() and p.name!='evidence_hashes.json'}
    (out/'evidence_hashes.json').write_text(json.dumps(hashes,indent=2)+'\n')
    print(out/'REPORT.md')


if __name__=='__main__':main()
