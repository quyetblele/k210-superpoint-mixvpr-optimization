# MixVPR D4 — standard Pitts30k-test benchmark in preparation

2026-09-13. User authorized standard public benchmark, superseding the custom-capture-only next step. D4 remains locked. Official MixVPR source commit4043915cef24818003ece1a8112bc8a24e69abe0 and released ResNet50/MixVPR512 checkpoint downloaded. Dataset/output files reside on D:/k210_benchmarks/pitts30k_test; no benchmark archives on C.

Official split:10000 reference,6816 query,25m GT radius, CPU FAISS exact L2 recall@1/5/10. D4 keeps RGB240 bicubic and CPU L2; official baseline uses RGB320 bilinear with its512D head. Evaluator synthetic recall matches official code; preprocessing exact; faithful nncase worker reproduces frozen DEV output bitwise. Model/Conv+Linear costs: D4 731300params/258553600MACs; official10092898params/8105369600MACs. No TEST metrics yet: image downloads/checksum verification ongoing.

Reproduce after dataset download: `.venv-mixvpr-cuda/bin/python mixvpr/setup_pitts30k.py`, then `.venv-mixvpr-cuda/bin/python mixvpr/run_pitts30k_benchmark.py`. Runner locks dataset/model/source hashes before extraction and evaluates all queries. No training, tuning, calibration or checkpoint changes. Public outdoor benchmark quality is separate from indoor application acceptance; do not fabricate a release gate after scoring.

---

# MixVPR D4 completion audit — BLOCKED: no valid independent TEST

2026-09-13. [Final data audit](../mixvpr/artifacts/test_protocol_audit/final_audit.json) · [New independent capture protocol](../mixvpr/artifacts/test_protocol_audit/NEW_CAPTURE_PROTOCOL.md).

All40 candidate locked-file hashes and104 provenance-evidence hashes PASS. Rechecked46 existing 7Scenes sequences and all37 known raw-video file hashes; no new video source in audited asset roots. Existing sequences/captures were used for TRAIN/DEV or engineering. Two clips remain UNKNOWN provenance and lack independent reference/GT; neither is eligible. Zero certified TEST queries/references. This audit covers known project data sources, not undocumented external data.

**TEST NOT RUN. Final release freeze BLOCKED, not a measured model-quality FAIL.** D4 remains the unchanged final candidate. Do not use DEV, Floor6, Floor7 or SuperPoint TEST. No training, pruning, KD/QAT, PTQ, thresholds/checkpoint changes, DB rebuild or full-online execution.

New-capture protocol specifies a new location, separate reference/query capture sessions, model-independent GT, split/hash locking, R@1/R@5/R@10 and counts/coverage, and no tuning after TEST. Approximately300 query images is a collection budget only; independence/uncertainty must be assessed. Absolute application-quality acceptance and final statistical protocol must be approved before TEST; prior DEV retention guardrails alone cannot qualify release. No final freeze manifest fabricated.

**Unblock:** supply independent new reference/query captures and GT, then finalize and lock the TEST manifest/protocol/gates before a single scoring run. D4 requires no further training.

---

# MixVPR D4 final candidate confirmed — challengers stopped

2026-09-13. User confirmed retain D4 and stop further parent/pruning experiments. [Parent B report](../mixvpr/artifacts/strong_parent_b/REPORT.md): bounded pilot completed at500, best update0; Indoor delta at250/500 = -2.55/-4.33pp, Project unchanged16/18. No benefit, no promotion. D4 weights, architecture and all40 frozen dependencies verified unchanged. No PTQ/TEST was run for parent.

D4 existing DEV R1 remains FP32/faithful INT8: Indoor83.92%/84.31%, Project88.89%/88.89%. Candidate is locked; independent-TEST-qualified release is still blocked. Audit found zero certified untouched query/reference samples in the known local sources. Two videos have unresolved provenance, not approved TEST data.

[Draft TEST manifest](../mixvpr/artifacts/test_protocol_audit/draft_test_manifest.json) records split/label/metric requirements and pending gates. NOT executable: data, ground truth, counts and absolute quality acceptance are missing. Do not reinterpret existing DEV or engineering scenes as untouched TEST. No DB rebuild or full-online until MixVPR release qualification.

**Next required input:** independent labeled reference/query captures (or provenance establishing an unused existing source) and approval of the final TEST protocol. Keep D4 unchanged; no more architecture search or training.

---

# MixVPR D4 retained — lock revalidated; TEST still blocked

2026-09-13. Latest user instruction supersedes bounded parent challenger: no additional training, D6, new KD or pruning. No challenger training was started. D4 checkpoint, RGB240 preprocessing, 512D output, CPU L2 and PTQ/nncase configuration remain unchanged.

[Revalidation](../mixvpr/artifacts/d4_final_candidate/revalidation_20260913.json): all 40 freeze-manifest file hashes PASS. Existing FP32/ONNX/faithful INT8 DEV evidence remains valid; no inference rerun needed. Indoor R@1 FP32/INT8: 83.92%/84.31%; Project: 88.89%/88.89%.

**Candidate locked, release NOT qualified. TEST NOT RUN.** No approved independent query/reference manifest, ground truth and TEST acceptance protocol exists. Existing SuperPoint TEST1000 is MixVPR DEV; previously used engineering captures cannot be renamed held-out TEST. Prior blocker/evidence preserved. No DB rebuild or full-online integration.

**ONE next decision:** establish and approve independent TEST data, ground truth and acceptance protocol before scoring; retain D4 meanwhile.

---

# Experiment status — MixVPR D4 candidate locked; heldout TEST blocked

[Candidate freeze manifest](../mixvpr/artifacts/d4_final_candidate/freeze_manifest.json) · [Verification/report](../mixvpr/artifacts/d4_final_candidate/REPORT.md)

D4 checkpoint5750 + RGB240 +512D CPU L2 + PTQ/nncase configuration locked. FP32/faithful8bit DEV evidence verified: Indoor R1=83.92/84.31%; Project=88.89/88.89%. All1,686simulator output hashes and exact metric reaggregation PASS. No inference rerun.

**TEST NOT RUN; release not qualified.** Existing independent_evaluation.json says locked labeled independent data is missing. All1,000SuperPoint TEST images belong to MixVPR DEV; GT1 is historically inspected engineering data. Missing independent TEST manifest/labels/acceptance protocol must be resolved before scoring. Candidate remains locked; no training, D6, new KD, DB rebuild or full-online.

---

# Experiment status — MixVPR D4/D6 pilot complete; retain D4

2026-09-12 · [Pilot report](../mixvpr/artifacts/depth_pilot/REPORT.md) · [Results](../mixvpr/artifacts/depth_pilot/results.json) · [Learning curve](../mixvpr/artifacts/depth_pilot/learning_curve.png)

Controlled continuation from frozen5750-update D4 weights. D6 copies all compatible tensors and adds two identity-initialized residual mixers. Same TRAIN/DEV, sampler/augmentation plan, existing SUP+ranking KD (no new loss), optimizer/LR schedule, seed20260913 and budget. Both stopped at500additional updates by the predeclared joint no-improvement rule; max budget1000. No TEST or full training.

| Additional update | D4 Indoor R1 | D6 Indoor R1 | D4/D6 Project R1 |
|---|---:|---:|---:|
|0|83.92%|83.92%|88.89% /88.89%|
|250|81.66%|80.79%|88.89% /88.89%|
|500|80.31%|81.11%|88.89% /88.89%|

Both DEV-selected checkpoints remain +0. Best-checkpoint quality below must not be mistaken for quality at +500.

| Model | FP32 R1 Indoor/Project | Faithful8bit R1 Indoor/Project | Retention Indoor/Project | Best added update | TOTAL(B) | Kmodel(B) |
|---|---|---|---|---:|---:|---:|
|D4|83.92% / 88.89%|84.31% / 88.89%|100.47% / 100.00%|0|2,947,880|766,632|
|D6|83.92% / 88.89%|84.31% / 88.89%|100.47% / 100.00%|0|2,997,224|815,976|

Both pass ONNX parity, compile/gencode/sim and faithful retrieval DEV over1,686images. Guardrails pass. D6 costs49,344B more compiler TOTAL/model bytes, with no selected-quality gain; mean post-init Indoor delta versusD4 is−0.0337pp. Single seed and18Project queries: no universal claim about D6 capacity. Compiler TOTAL is not measured board peak/latency.

**ONE next decision:** retain D4; no demonstrated reason to prioritize full-training D6. Stop here for user decision. No defaults promoted; frozen MixVPR and SuperPoint evidence preserved.

---

# Experiment status — MixVPR Stage1 complete; awaiting pilot decision

2026-09-12 · [Full audit/frontier](../mixvpr/artifacts/stage1_frontier/REPORT.md) · [Machine-readable results](../mixvpr/artifacts/stage1_frontier/summary.json)

Frozen baseline C160/H96/D4/P128,240²,100tokens,512D: checkpoint5750 cumulative updates,731,300params,258.554M Conv+Linear MACs. Historical Indoor DEV R1=83.92%, Project DEV88.89%; Floor6 engineering retrieval FP32/8bit R1=15%/15%, not canonical SuperPoint pose or untouched TEST. Baseline checkpoint/preprocessing/source hashes verified; recompiled kmodel byte-identical.

| Candidate | C/H/D/P | Params | MACs(M) | Compile | Gencode | Sim | KPU/CPU Conv | TOTAL(B) | Kmodel(B) |
|---|---|---:|---:|---|---|---|---|---:|---:|
| baseline | 160/96/4/128 | 731,300 | 258.554 | PASS | PASS | PASS | 20/0 | 2947880 | 766632 |
| late192 | 192/96/4/128 | 869,092 | 291.488 | PASS | FAIL | NOT_RUN | N/A | — | — |
| late224 | 224/96/4/128 | 1,025,316 | 328.570 | PASS | FAIL | NOT_RUN | N/A | — | — |
| mixer128 | 160/128/4/128 | 757,028 | 262.650 | PASS | FAIL | NOT_RUN | N/A | — | — |
| depth6 | 160/96/6/128 | 770,492 | 264.698 | PASS | PASS | PASS | 24/0 | 2997224 | 815976 |
| projection256 | 160/96/4/256 | 751,706 | 260.602 | PASS | FAIL | NOT_RUN | N/A | — | — |
| late192_mixer128 | 192/128/4/128 | 894,820 | 296.403 | PASS | FAIL | NOT_RUN | N/A | — | — |

All6 new graphs untrained; fixed early widths/input/calibration. Five gencode allocator OOMs preserved; no executable means no simulator run. CPU normalization/data movement remains despite0 CPUConv. Compiler TOTAL is not measured board peak or full-system reserve.

FACT: original full model and separate backbone/aggregator hit gencode OOM; current compact baseline executes. INFERENCE: historical quality gap is larger than quantization loss. UNKNOWN: dominant quality component, actual latency/memory and whether D6 improves retrieval.

**ONE next decision:** only D6 passes the graph and provisional≤3MiB resource screen. User selects whether to pilot baselineD4 versusD6. No training, KD, TEST or default promotion. SuperPoint frozen artifacts unchanged. Full verification and mapping-count metadata correction are recorded in the linked report.

---

# SuperPoint frozen release — final TEST PASS

Update 500 and greedy NMS frozen before opening TEST. No training, model, calibration, preprocessing or threshold changes during TEST.

| Metric | FP32 greedy | Faithful INT8 greedy |
|---|---:|---:|
| DEV correct matches | 81.17 | 81.62 |
| DEV precision (%) | 88.75 | 88.34 |
| Floor6 DEV pose | 20/20 | 20/20 |
| TEST correct matches/pair | 88.73 | 91.01 |
| TEST precision (%) | 88.40 | 88.12 |
| TEST repeatability (%) | 70.59 | 70.62 |
| TEST mean keypoints | 147.06 | 151.54 |

TEST gate PASS: correct-match retention 102.56% (minimum 90%); precision drop 0.277 percentage points (maximum 2).

TEST: 1,000 heldout redkitchen/seq-06 source images, one deterministic synthetic homography pair per image; each backend evaluated once. All 1,000 retained, no TEST tuning. This is feature/matching validation on one sequence, not independent pose TEST or measured K210 performance. DEV is scene-macro; TEST is pair-macro.

The historical NMS regression gate remains FAIL, including original INT8 precision/repeatability regressions. The user approved the DEV amendment before TEST because original INT8 NMS had same-score/nearby-keypoint pathology (80.75%); the amended comparison uses identical greedy NMS for FP32 and INT8. Numerical strict raw-head FAIL also remains preserved with its prior rounding explanation.

Verification PASS: 34 frozen file hashes unchanged, old FAIL evidence unchanged, approval < freeze < TEST, two disjoint 500-pair shards and exclusive stage markers. Freeze manifest SHA256: `d85fc625f5c9549d75e854b2728373960713b2b512d9e68d916714c1f4f2a0e2`.

ONE next decision: integrate the frozen release into the full online localization pipeline; verify resource use and end-to-end behavior on PC before board testing. Legacy default aliases remain historical; integration must explicitly consume this frozen release.

Evidence: [freeze manifest](../superpoint/artifacts/full_training/freeze_manifest.json), [TEST results](../superpoint/artifacts/full_training/final_test/results.json), [verification](../superpoint/artifacts/full_training/final_test/verification.json).

---

Historical entries below describe the state at their original timestamps.

# Experiment status — faithful INT8 complete; greedy recovery measured; policy decision pending

2026-09-11 · [INT8 report](../superpoint/artifacts/full_training/INT8_REPORT.md) · [Rounding explanation](../superpoint/artifacts/full_training/parity_resolution/REPORT.md)

Update500 checkpoint unchanged. Raw discrepancy explained with96local Conv bounds plus semantic parity; strict raw FAIL retained under an explicit numerical-acceptance amendment. Faithful nncase trained-model compile/simulation/evaluation PASS,12KPU/0CPU Conv, unchanged64TRAIN calibration.

| Metric | FP32 original | INT8 original | FP32 greedy | INT8 greedy |
|---|---:|---:|---:|---:|
| DEV correct matches | 80.34 | 54.71 | 81.17 | 81.62 |
| DEV precision (%) | 88.62 | 92.20 | 88.75 | 88.34 |
| DEV repeatability (%) | 71.08 | 76.54 | 71.34 | 70.24 |
| DEV keypoints | 131.16 | 155.35 | 132.31 | 136.17 |
| Floor6 pose | 20/20 | 19/20 | 20/20 | 20/20 |

Original NMS fails freeze retention gate (68.1% <90%). INT8 equal-score nearby-keypoint rate80.75%; greedy removes clusters. Greedy meets all freeze-quality limits relative to FP32, but historical NMS regression still FAILS precision/repeatability versus original INT8. Neither failed gate is relabeled PASS.

**Pending user decision:** approve an explicit DEV amendment using FP32 as reference, then freeze greedy and open TEST; or retain the old gate. No freeze/default promotion, TEST unopened, no integration changes yet. TEST runner prepared and verified only on TRAIN. Existing online code inspected read-only for subsequent integration.

---

# Experiment status — full training complete; FP32 floor6 20/20; INT8 gate blocked

[Full report](../superpoint/artifacts/full_training/REPORT.md) · [Graph probes](../superpoint/artifacts/architecture_headroom_probe/REPORT.md)

Architecture `[24,64,128,192]`: largest probed capacity passing declared reserve screening,12KPU/0CPU Conv, compiler total2.959MiB. No measured full-system reserve/latency claim. Expanded TRAIN2,828 / DEV42 / sealed TEST1,000 (redkitchen/seq-06). No download. Seed20260919, two pairs/update, max10,000, eval500, minimum2,000, patience4 with0.5% correct-MNN improvement threshold; actual stop2,500 by DEV patience. Best500, ~351seconds training,1.768 effective dataset passes.

| Same evaluator | Retained pilot | New best |
|---|---:|---:|
| FP32 DEV correct matches | 64.77 | 80.34 |
| FP32 DEV precision | 67.37% | 88.62% |
| FP32 floor6 original NMS | 9/20 | 20/20 |
| FP32 floor6 greedy NMS | 9/20 | 20/20 |

Before/after, not architecture-only causal A/B. Internal DEV, not TEST accuracy. Checkpoints/source hashes, finite losses and120PnP replays PASS. ONNX8 sparse-feature checks and42DEV metrics match; raw-head tolerance still FAIL (1–3elements depending on ORT backend). Native FP32 also exceeds FP64 tolerance on3elements. Full-trained INT8 not compiled; gate was not bypassed. TEST has no inference/tuning use. Default model/NMS and MixVPR unchanged.

**ONE next decision:** keep this checkpoint fixed, resolve numerical parity, then faithful nncase + floor6 INT8; no more training first.

---

# Experiment status — detector ranking A/B complete; retain prior recipe

Latest: [controlled detector-ranking report](../superpoint/artifacts/detector_ranking_ab/REPORT.md), 2026-09-10. Both arms completed 400 updates and faithful nncase validation.

| Arm | Descriptor top1 FP32 / INT8 | Greedy FP32 pose | Greedy INT8 pose | INT8 keypoints + teacher descriptor | INT8 correct/image | INT8 inliers/image | INT8 inlier ratio |
|---|---:|---:|---:|---:|---:|---:|---:|
| A | 72.40% / 72.39% | 9/20 | 8/20 | 9/20 | 2.80 | 7.50 | 12.84% |
| B | 73.01% / 73.56% | 13/20 | 7/20 | 11/20 | 2.75 | 7.55 | 10.65% |

**REJECT B for final recipe:** primary predeclared greedy INT8 pose improvement failed (8→7), despite FP32 improvement (9→13) and teacher-donor improvement (9→11 at INT8 keypoints). Retain `point_geometry_r2_cw`; original NMS remains default. Results are internal DEV, one seed, not independent TEST or measured board performance.

**ONE next decision:** No further training: freeze B INT8 greedy keypoints and compare FP32 versus faithful INT8 descriptor donors with the same matching/reference-observation protocol to isolate the remaining quantization loss. B remains diagnostic only.

Validation: exact control weights and floor6 replay, 16 deployment/evaluation stages, 360 extra PnP replays PASS. Parity required disabling ORT graph optimizations identically for both arms; original tolerance, exported graph, nncase and evaluator metrics unchanged. See the explicit numerical-validation addendum in the report.

---

# Experiment status — INT8 greedy score/ranking/coverage audit complete

Latest: [score/ranking/top-K/spatial audit](../superpoint/artifacts/floor6_score_ranking_audit/REPORT.md),2026-09-10. INT8 descriptor field/bank fixed across all probes.

| Probe; same B INT8 descriptor | Pose | Mean keypoints | Correct matches/image | Inliers/image |
|---|---:|---:|---:|---:|
| INT8 greedy baseline | 8/20 | 133.25 | 2.80 | 7.50 |
| FP32 greedy keypoints | 9/20 | 132.80 | 1.95 | 7.50 |
| Only top-K320 | 9/20 | 168.80 | 3.10 | 8.15 |
| Only score threshold.001 | 9/20 | 160.00 | 3.35 | 7.75 |
| Up to10points/cell,4x4grid | 3/20 | 77.00 | 1.50 | 6.30 |
| Teacher reranks same INT8 survivor pool | 12/20 | 133.25 | 2.75 | 7.95 |

Ranking and placement remain limiting. Teacher reranking rescues four queries via changed point sets; topK/threshold alone reach only9/20 and quota-grid falls to3/20. Teacher has34.7keypoints/image within4px of observed active projections versus4.4INT8-greedy; lowering the threshold still leaves a poor-ranked candidate pool. FP32-versus-INT8 aggregate9/8 hides three FP32-only and two INT8-only successes.

**ONE next decision:** prioritize controlled detector supervision/ranking with descriptor recipe B retained; do not promote the tested K/threshold/grid settings. No training, QAT, architecture, map or MixVPR changes. Default NMS remains original.

Validation:20 exact baseline replays,120 PnP replays, source hashes and teacher-rerank set checks PASS. Internal DEV only; no external accuracy or capacity conclusion.

Historical context below.

# Experiment status — greedy NMS regression FAIL; original default retained

Latest: [42-pair NMS regression and canonical floor6](../superpoint/artifacts/greedy_nms_regression/REPORT.md),2026-09-10.

| Backend / NMS | Mean keypoints (scene macro) | Nearby rate (image mean) | Correct MNN | MNN precision | Floor6 poses |
|---|---:|---:|---:|---:|---:|
| fp32 / original | 147.83 | 0.00% | 64.77 | 67.37% | 9/20 |
| fp32 / greedy | 148.16 | 0.00% | 65.06 | 67.85% | 9/20 |
| int8 / original | 159.34 | 84.27% | 28.21 | 73.59% | 3/20 |
| int8 / greedy | 148.25 | 0.00% | 59.30 | 66.20% | 8/20 |

**FAIL only on the predeclared INT8 precision guard:**73.59→66.20%, exceeding the assistant-selected2point limit. All FP32 checks pass; INT8 correct matches and floor6 pose improve substantially. Do not reinterpret this as evidence that greedy worsens FP32 or all INT8 quality.

Original NMS remains default; greedy is not promoted. No conditional post-promotion run performed. Both canonical floor6 backends were already evaluated as downstream regression (FP32 9→9/20, INT8 3→8/20).

**ONE next decision:** retain greedy as a candidate for a separately specified precision-versus-localization acceptance criterion on held-out data; no posthoc relaxation on this DEV run.

No training, QAT, model, architecture or MixVPR changes. Internal DEV only.

Historical context below.

# Experiment status — B INT8 NMS plateau audit complete

Latest: [B fixed-descriptor keypoint audit](../superpoint/artifacts/floor6_b_keypoint_audit/REPORT.md),2026-09-10.

| Keypoint source; descriptor always B INT8 | Pose | Correct matches/image | Inliers/image | Inlier ratio | Cluster fraction <=4px |
|---|---:|---:|---:|---:|---:|
| FP32 original NMS | 10/20 | 1.95 | 7.60 | 13.07% | 0.00% |
| INT8 original NMS | 3/20 | 1.80 | 6.40 | 18.39% | 92.45% |
| INT8 greedy NMS | 8/20 | 2.80 | 7.50 | 12.84% | 0.00% |

Confirmed contributor:92.45% of original INT8 keypoints have an equal-score neighbor within NMS radius4. One strict greedy-NMS intervention removes these clusters and improves3→8/20 with descriptor field/bank unchanged. This is partial recovery, not a release or architecture-capacity conclusion.

**ONE next gate:** regression-check greedy NMS on the unchanged42-pair synthetic DEV FP32/faithful INT8 before final postprocess selection. No training/QAT/default promotion.

Validation:40 exact baseline replays,60 deterministic PnP replays, strict radius and score-tie checks PASS. Internal floor6 DEV only; all stair geometry conclusions remain invalid.

Historical context below.

# Experiment status — student-point geometry A/B complete

Latest: [400-update controlled descriptor supervision A/B](../superpoint/artifacts/student_point_supervision_ab/REPORT.md),2026-09-10.

Pose cells: success; mean correct matches; mean inliers; mean inlier ratio.

| Measure | A current | B + student-point geometry |
|---|---:|---:|
| Floor6 descriptor top1 FP32 / INT8 | 71.06% / 71.62% | 72.40% / 72.39% |
| Synthetic correct MNN FP32 / INT8 | 65.49 / 27.52 | 64.77 / 28.21 |
| Own FP32 keypoints + teacher descriptor | 11/20; 2.85; 8.30; 13.58% | 10/20; 2.25; 8.20; 14.67% |
| Own FP32 keypoints + student FP32 descriptor | 5/20; 2.20; 7.35; 12.00% | 9/20; 2.20; 7.70; 13.01% |
| Own FP32 keypoints + faithful INT8 descriptor | 5/20; 2.15; 7.15; 11.99% | 10/20; 1.95; 7.60; 13.07% |
| Shared old-student keypoints + FP32 descriptor | 5/20; 2.20; 7.35; 12.00% | 9/20; 2.70; 7.40; 12.14% |
| Shared old-student keypoints + INT8 descriptor | 5/20; 2.15; 7.15; 11.99% | 7/20; 2.85; 7.60; 12.48% |
| Full INT8 keypoints + INT8 descriptor | 1/20; 1.70; 6.60; 17.50% | 3/20; 1.80; 6.40; 18.39% |

**Decision: retain B provisionally for descriptor supervision.** Predeclared rule passes: descriptor DEV improves, own-FP32 poses5→9, full INT8 poses1→3, shared-old-keypoint FP32/INT8 poses5→9/7. Synthetic FP32 metrics and own-keypoint teacher-donor success regress slightly; this is not a final recipe or deployment pass. Default checkpoint unchanged.

**ONE next gate:** audit B FP32-versus-INT8 keypoints with descriptor fixed; B INT8 descriptor gives10/20 at FP32 keypoints but full INT8 gives3/20. No automatic full train/QAT/architecture change.

Validation: exact control weights and80 floor6 rows; same calibration; both400 updates;16 stages PASS; real nncase;200 extra PnP replays. Floor6 internal DEV only; stair map results remain invalid.

Historical context below.

# Experiment status — floor6 fixed-student-keypoint donor A/B/C complete

Latest: [floor6 descriptor donor test](../superpoint/artifacts/floor6_descriptor_donors/REPORT.md),2026-09-10. All three arms use identical student **FP32** keypoints.

| Donor tại cùng student FP32 keypoints | Pose success | Correct matches/ảnh | Inliers/ảnh | Inlier ratio |
|---|---:|---:|---:|---:|
| A — Teacher FP32 | 11/20 | 2.85 | 8.30 | 13.58% |
| B — Student FP32 | 5/20 | 2.20 | 7.35 | 12.00% |
| C — Student faithful INT8 | 5/20 | 2.15 | 7.15 | 11.99% |

Teacher descriptors improve5→11/20 at the same student locations: a material descriptor-at-location deficit. Teacher-keypoint teacher baseline remains20/20, so student keypoint placement/set also limits this pipeline. B/C successful query sets are identical; this does not prioritize QAT. C is not the prior own-INT8-keypoint1/20 run.

**ONE next decision:** design one bounded descriptor-supervision A/B at deployment-relevant student keypoints; keep architecture and detector supervision fixed. No training performed or checkpoint promoted.

Validation:60 PnP replays,20 exact B baseline replays,885 source artifact hashes PASS. Internal DEV only; native map units; no external accuracy or board claim.

Historical context below.

# Experiment status — floor6 student evaluation complete

Latest: [student FP32/faithful INT8 on floor6](../superpoint/artifacts/floor6_student_evaluation/REPORT.md), run_001,2026-09-10.

| Descriptor/backend | Teacher-fixed keypoints | Own keypoints | Mean inliers fixed / own |
|---|---:|---:|---:|
| Teacher FP32 baseline | 20/20 | 20/20 | 27.20 / 27.20 |
| Student FP32 | 20/20 | 5/20 | 20.35 / 7.35 |
| Student INT8 | 20/20 | 1/20 | 20.85 / 6.60 |

Student descriptors support20/20 at teacher keypoints; own-feature localization drops to5/20 FP32 and1/20 INT8. Prioritize separation of placement from descriptor-at-student-location effects. No conclusion that architecture is insufficient or detector alone is responsible.

**ONE next gate:** floor6 student FP32 keypoints + teacher versus student descriptor donor, same map/matcher/native fisheye PnP. No training or architecture changes.

Validation:177 real nncase image outputs;403 initial artifact hashes;80 independent PnP and geometric-match-count replays PASS. Internal DEV only; no board or independent TEST claim. All stair-dependent conclusions remain invalid.

Historical context below.

# Experiment status — floor6 teacher baseline PASS

Latest: [fresh floor6 teacher baseline](../superpoint/artifacts/floor6_teacher_baseline/REPORT.md), 2026-09-10.

Teacher FP32: **20/20 poses**,27.20 mean inliers,29.68% mean inlier ratio; native fisheye geometry sanity20/20. Generated1124 reference +20 query global descriptors and177 fresh local-image feature artifacts. No stair cache or bank reused.

Validation:541 artifact hashes;20 independent deterministic PnP replays;1000 fisheye coordinate round trips PASS. Internal map-based DEV only; no independent TEST/metric-scale/board claim.

**ONE next gate:** evaluate current student FP32/faithful INT8 against this frozen floor6 protocol. No training, architecture, threshold or MixVPR weight changes. All retired stair-dependent conclusions remain invalid for model decisions.

Historical context below.

# Experiment status — stair retired; floor6 selected

Active evaluation map: **floor6_v1** via `superpoint/configs/localization_map.json`. Checked1144 source images and64536 Point3Ds, dimensions, reciprocal tracks and20/20 native-camera internal PnP cases. RADIAL_FISHEYE geometry uses pycolmap; the old SIMPLE_RADIAL evaluator is blocked. This is internal sanity, not proof that the map is correct in the real world.128 images contain duplicate Point3D observations; the sanity test excludes ambiguous IDs.

**All stair-dependent PnP, Point3D descriptor and projected-coverage conclusions are invalid for model decisions.** In particular, previous detector-versus-descriptor priorities and coordinate-correction localization conclusions must be re-established on the replacement map. Map-independent training/ONNX/INT8 evidence remains separate. The interrupted spatial audit is not promoted.

Deleted stair_6_7_v1, its identical-geometry stair_template_v1 package, and both stair online map copies. Original source photographs and historical experiment evidence retained. Deletion identities: `superpoint/artifacts/map_replacement_floor6/deletion_record.json`.

ONE next gate: fresh teacher baseline and reference/query artifacts on floor6 with its native fisheye camera. No old stair IDs, descriptor banks or cached images may be reused. No additional training or MixVPR changes.

Historical entries below are superseded wherever they rely on stair geometry.

# Experiment status — student-keypoint donor test complete

Latest: [student-keypoint descriptor A/B](../superpoint/artifacts/student_keypoint_descriptor_pnp/REPORT.md), 2026-09-10. Current default wider_r2_cw FP32 keypoints; descriptor donor on both query and reference.

| Fixed student FP32 keypoints + donor | Pose success | Correct matches/query | Inliers/query |
|---|---:|---:|---:|
| A: teacher descriptor | 1/20 | 2.05 | 3.70 |
| B: student descriptor | 0/20 | 1.15 | 1.25 |

**Conclusion:** teacher descriptors do not rescue current student positions (1/20 versus0/20). Prioritize keypoint placement/coverage; this does not eliminate the separate descriptor deficit seen at improved corners.

**ONE next decision:** focus the next intervention on detector spatial localization/target alignment using the already completed audit; defer descriptor training and architecture changes. Do not repeat the failed NMS/threshold sweep.

Validation: regenerated keypoints identical;54 image coordinate/ID sets identical;40 deterministic PnP replays;20 student baseline replays exact. Default checkpoint weight equivalence verified. No training, INT8 experiment, map/MixVPR or default changes. Nearby DEV only.

Previous gates below are historical context.

# Experiment status — coordinate mapping A/B complete

Latest: [controlled mapping A/B](../superpoint/artifacts/coordinate_mapping_ab/REPORT.md), 2026-09-10. Both400 updates, selected step400; shared-valid excludes the same4 correspondences from both arms.

| Arm | Backend | Descriptor DEV top1 | Fixed teacher PnP | Fixed improved PnP | Student-keypoint pose |
|---|---|---:|---:|---:|---:|
| A | fp32 | 68.29% | 10/20 | 5/20 | 0/20 |
| A | int8 | 67.07% | 11/20 | 8/20 | 0/20 |
| B | fp32 | 67.46% | 10/20 | 6/20 | 0/20 |
| B | int8 | 67.85% | 7/20 | 3/20 | 0/20 |

**Conclusion:** pixel-center correction is geometrically valid, but this pilot shows no consistent quality gain and worse INT8 PnP. Own-keypoint localization remains0/20. A is the newly trained shared-valid control, not the prior unfiltered checkpoint.

**ONE next decision:** do not promote B or start full training from this result; retain the existing default checkpoint. Independent TEST and board qualification remain outstanding.

Validation: identical init/plan/calibration; finite800 updates; both ONNX parity, nncase compile/simulator PASS;80 additional PnP replays exact. Architecture, losses, detector supervision, evaluator and MixVPR unchanged.

Previous gates below are historical context.

# Experiment status — fixed improved-keypoint descriptor comparison

Latest: [controlled descriptor donor results](../superpoint/artifacts/improved_keypoint_descriptor_pnp/REPORT.md), canonical run_002, 2026-09-10.

| Descriptor donor | Pose success | Correct matches / query | Inliers / query | Mean inlier ratio |
|---|---:|---:|---:|---:|
| teacher_fp32 | 15/20 | 14.70 | 13.75 | 26.85% |
| student_fp32 | 8/20 | 9.15 | 5.00 | 7.79% |
| student_int8 | 6/20 | 8.80 | 4.60 | 7.32% |

Same improved keypoints and frozen matcher/map/PnP. Teacher recovers seven additional queries versus FP32, with no reverse wins. Student matches and all40 pose results exactly replay the prior audit. Descriptor quality under this matcher is a material remaining limitation; quantization is not the sole cause.

**ONE next decision:** design a bounded descriptor-supervision pilot at deployment-relevant keypoint locations; do not repeat the failed additive cosine KD recipe by default. No new training was run or recipe promoted. Full training/QAT/MixVPR remain unchanged. Nearby DEV only; no independent TEST or board qualification.

Previous gates below are historical context.

# Experiment status — keypoint root-cause audit

Latest: [six-hypothesis audit and partial recovery](../superpoint/artifacts/keypoint_cause_audit/REPORT.md).

Student has160 keypoints but only2.7 FP32 /3.3 INT8 geometrically useful at4px, versus16.1 teacher. Mapping algebra passes; threshold/border/NMS/sampling changes alone do not recover pose. CPU Shi–Tomasi on288×512 plus masked subpixel refinement gives 8/20 FP32 and 6/20 INT8 (previously0/20). This is PARTIAL RECOVERY on exploratory DEV, not release-ready.

Model, training recipe, calibration, production map and MixVPR unchanged. Next: investigate descriptor/matching at fixed improved locations; no automatic training or promotion. PC-first / Sipeed Maix Bit target remains unchanged.

Previous gates below are historical context.

# Experiment status — descriptor KD A/B

Latest: [completed descriptor KD pilot](../superpoint/artifacts/descriptor_kd_ab/REPORT.md).

| Arm | Backend | Selected update | Natural descriptor top1 | Fixed-teacher-keypoint PnP | Own-keypoint PnP | Synthetic correct MNN |
|---|---|---:|---:|---:|---:|---:|
|A|fp32|400|67.97%|10/20|0/20|65.49|
|A|int8|400|67.76%|9/20|0/20|27.52|
|B|fp32|400|67.30%|8/20|0/20|68.28|
|B|int8|400|66.97%|11/20|0/20|28.32|

Decision: **DO_NOT_ADD_TO_FINAL_RECIPE_YET**. Keep A as the unchanged baseline; do not start full training or add KD to the final recipe from this pilot. Use the fixed-versus-own-keypoint evidence to resolve the remaining localization limitation before another training intervention.

No full training/QAT/architecture/MixVPR change. User preference remains PC preparation before Sipeed Maix Bit board trials. Public benchmark/untouched TEST and hardware qualification remain incomplete.

Previous gates below are historical context; their proposed descriptor-KD A/B has now been executed above.

# Experiment status — fixed-keypoint descriptor → PnP

Latest gate: [A/B/C report](../superpoint/artifacts/fixed_keypoint_pnp/run_001/REPORT.md), COMPLETE. Fixed teacher keypoints: teacher descriptors16/20 good poses, student FP32 10/20, faithful INT8 9/20. Known-track geometry sanity20/20; all60 PnP results replay exactly. No training or production change.

The descriptor gap affects pose in this diagnostic. Propose a bounded current-recipe vs descriptor-KD A/B; it is NOT executed. Full training, QAT and final recipe freeze remain premature. Teacher-keypoint hybrid scores are not deployable student scores. Public benchmark/untouched TEST remain incomplete. User preference remains PC preparation before Sipeed Maix Bit board trials.

The previous measurement gate is preserved below for context; its proposed fixed-keypoint experiment has now been executed by the latest gate above.


Deployment target and order: [Sipeed Maix Bit K210; complete PC preparation before board trials](DEPLOYMENT_TARGET.md), as requested by the user. Board I/O, latency, RAM and stability remain unmeasured until the later hardware phase.

Gate run: `superpoint/artifacts/measurement_gate/run_002/`. Diagnostic measurements complete; public benchmark and final system qualification incomplete. No new training, no MixVPR change, no production map update.

## 1. CURRENT CANDIDATE IDENTITY

`wider_r2_cw`, widths24/32/64/96, head128, descriptor256. Checkpoint SHA256 `c72da545c1b8a03c8db90bc52934b73a7cdaae07a268052a010b32b587086ff0`; kmodel SHA256 `fd8e7f96feb4df1a8189b2aa3265a27de45f19027af0c5082ef56fda771e2786`. Logical W184×H320, CW90 deployment. Source hashes accompany the dirty/untracked Git state; commit alone is not used as implementation identity.

## 2. DATA SPLIT / DEV / TEST STATUS

140 TRAIN /42 DEV source images for the fixed synthetic pilot. Additional natural-view diagnostic uses existing20DEV queries and21DEV support frames. **TEST = NOT YET ESTABLISHED.** Map reference/query roles are separate from network training roles. Inventory completed before any download; no dataset downloaded or deleted. Pilot evidence retained.

## 3. EVALUATOR VALIDATION RESULT

INCONCLUSIVE overall. Upstream decoder comparison on six views, metric contract checks and42-pair replay PASS. Public benchmark reproduction NOT_RUN. No existing frozen public evaluator was selected; these results retain engineering scope.

## 4–6. TEACHER FP32 / STUDENT FP32 / STUDENT INT8 REAL-NNCASE RESULTS

| Model | Keypoints | MNN | Correct MNN | Precision | Repeatability* | Zero-correct pairs |
|---|---:|---:|---:|---:|---:|---:|
| teacher_fp32 | 84.82 | 74.51 | 73.42 | 98.63% | 83.72% | 0/42 |
| student_fp32 | 146.90 | 94.71 | 65.49 | 68.57% | 59.82% | 0/42 |
| student_int8 | 158.80 | 36.74 | 27.52 | 74.31% | 56.59% | 0/42 |

Actual INT8 simulator outputs were reused after identity checks, not regenerated. *Custom pilot repeatability. Full distributions and pair-level failures are in JSON/CSV.

New natural-view known-track descriptor ranking:

| Model | Top-1 legacy | Top-1 −0.5 px | Top-5 legacy |
|---|---:|---:|---:|
| teacher_fp32 | 87.38% | 87.36% | 98.71% |
| student_fp32 | 68.06% | 67.97% | 92.14% |
| student_int8 | 67.84% | 67.76% | 91.92% |

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
