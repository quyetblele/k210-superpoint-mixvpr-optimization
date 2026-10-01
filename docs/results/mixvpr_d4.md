# MixVPR D4 final candidate locked — TEST blocked

Candidate/config lock completed before any TEST inference. Release is NOT yet TEST-qualified.

| Evidence | FP32 | Faithful8bit | Heldout TEST |
|---|---:|---:|---|
| Indoor DEV R1 |83.92%|84.31%|NOT RUN|
| Project DEV R1 |88.89%|88.89%|NOT RUN|
| Export / compile / gencode / sim |ONNX parity PASS|PASS|No locked protocol|

Checkpoint, D4 architecture, RGB240 preprocessing,512D CPU L2,64TRAIN calibration and nncase config are in locked_config.json and freeze_manifest.json. Model weights in original checkpoint and selected pilot checkpoint are bitwise equal; serialization hashes differ. The exact pilot kmodel with full DEV evidence is selected, not silently swapped with the historical binary.

- Original checkpoint SHA256: `d788f6fd49d28fc7e3f171ac9cbddd20920dae0ec9d3c53351e0ada440512e42`.
- Selected ONNX SHA256: `5581553b197472337946dd48ab55175a2711b5d8c12ee43f8bd5583feeac138d`.
- Selected kmodel SHA256: `2a3e3eb6bab6c11faad06b29c86da740ed76219b50cacc2cce185a99d43c02a1`.
- Freeze manifest SHA256: `5d6969356728d0ec622229679999eb4b04823acd9f3a297295c5ade0b9260bdb`.

Verified40 locked files,1,686INT8 output hashes and exact FP32/INT8 DEV reaggregation without rerunning inference. Model/config/source identities and historical failures remain unchanged.

Existing independent_evaluation.json explicitly reports BLOCKED_MISSING_LOCKED_LABELED_DATA. Existing MixVPR dataset has only TRAIN/DEV. All1,000SuperPoint heldout source images are MixVPR DEV, so they cannot be reused as untouched MixVPR TEST. Floor6/Floor7 GT1 was previously inspected and is engineering-only. No independent TEST query/gallery/labels/acceptance protocol was found in the audited workflow.

Required before TEST: identify independently held-out query/gallery data and audit provenance, lock retrieval labels/exclusions/evaluator and numerical/absolute acceptance gates. DEV retention limits cannot silently become all TEST/application requirements. No new images downloaded, no TEST opened, no DB rebuilt or online integration started.

Next decision: authorize preparing a new independent TEST protocol from remaining available data, or supply the intended locked TEST protocol/data. Keep this candidate unchanged while resolving TEST; do not mark final release PASS.
