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
