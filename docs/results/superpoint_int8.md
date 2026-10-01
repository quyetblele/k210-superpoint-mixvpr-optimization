# Update500 faithful INT8 and NMS validation — policy decision pending

Checkpoint unchanged: `b83e2e99e0d225c4e8b0c895a11b41711c400dea0430f6026a9058ebfa5c9416`. Raw parity discrepancy explained and explicitly accepted by a checkpoint/graph-bound numerical amendment. Strict raw-unit FAIL remains recorded; no checkpoint rescaling, tolerance increase or training. All96 Conv local rounding checks and semantic/dense/sparse parity pass, including exact42DEV metrics. See [numerical report](parity_resolution/REPORT.md).

Faithful nncase1.8 compile/simulator now complete for the trained checkpoint:12KPUConv,0CPUConv, kmodel1,331,368bytes, same64 TRAIN calibration tensors. Original floor6 FP32 results exactly reproduced. No board timing/peak-memory claim.

| Metric | FP32 original | INT8 original | FP32 greedy | INT8 greedy |
|---|---:|---:|---:|---:|
| DEV correct matches | 80.34 | 54.71 | 81.17 | 81.62 |
| DEV precision (%) | 88.62 | 92.20 | 88.75 | 88.34 |
| DEV repeatability (%) | 71.08 | 76.54 | 71.34 | 70.24 |
| DEV keypoints | 131.16 | 155.35 | 132.31 | 136.17 |
| Floor6 pose | 20/20 | 19/20 | 20/20 | 20/20 |

## Gate outcomes retained, not relabeled

Original NMS: only the predeclared DEV correct-match retention condition fails (68.104% versus required90%). Precision improves; floor6 INT8 gives19/20 poses and95.63% of FP32 mean inliers. Its status remains DO_NOT_FREEZE.

A follow-up original-postprocess audit exactly replays DEV metrics and finds80.752% of INT8 keypoints have another equal-score keypoint within Chebyshev4px; FP32 has0%. A controlled existing-greedy-NMS regression then removes all such clusters, restores matching, and gives20/20 floor6 poses. Threshold, cap160, descriptor sampling, checkpoint, kmodel, data, matcher and PnP remain fixed.

The historical NMS regression gate nevertheless FAILS two conditions: INT8 precision88.34% is3.87percentage points below original INT8 precision92.20%, and repeatability70.24% is below76.54%. These outcomes are preserved. They are not silently excused as a PASS. Against FP32 greedy, INT8 retains100.56% correct matches, loses0.413precision points, gives20/20 poses and100% mean inlier retention: the original *quality thresholds relative to FP32* all pass. Against original FP32 the same limits also pass. This motivates, but does not retrospectively validate, a different engineering acceptance policy.

User decision requested: authorize an explicit DEV policy amendment using FP32 as the quality reference, keep the old regression FAIL on record, freeze greedy and then run the untouched TEST; OR retain the old gate and do not freeze. Pending that decision, no release/model/NMS default has changed, TEST remains sealed, and integration has not started.

## Ready work

The final TEST runner is implemented and syntax checked. Its deterministic pair generation was tested only on a TRAIN source. All1,000 reserved redkitchen/seq-06 images will be retained, including empty-feature cases; no corner-count filtering. It requires an authorized immutable freeze manifest before reading TEST pixels. Geometry is synthetic homography, so this is heldout feature/matching evidence, not a public benchmark reproduction or independent pose test. INT8 shards use the actual nncase simulator. No TEST operation has run.

Existing online PC reference was inspected read-only: its old320x240 preprocessing, local descriptor identity/bank and camera/PnP path still need adaptation to the frozen model and canonical floor6. No MixVPR changes made.

Validation:8checkpoint-bound pipeline stages,80floor6 PnP replays plus80NMS regression replays, original metric replays, and original/greedy ONNX42DEV parity PASS. Checkpoint and calibration hashes unchanged. See deployment_verification.json.
