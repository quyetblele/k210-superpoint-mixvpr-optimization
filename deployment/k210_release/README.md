# K210 pre-board release package

This package contains the frozen, host-validated artifacts for the first board
I/O test. It is not a claim of board runtime qualification.

## Runtime order

1. Camera frame → grayscale SuperPoint preprocessing (the portrait/CW90
   contract in `contracts/superpoint_preprocessing.json`).
2. Run `models/superpoint_full_r2_cw.kmodel`.
3. Apply the frozen inverse rotation, detector permutation, greedy NMS,
   border mask, descriptor sampling and normalization from
   `contracts/superpoint_postprocess.json` on the CPU.
4. For global retrieval, convert the raw RGB frame using
   `contracts/mixvpr.json`, then run `models/mixvpr_d4_240x240.kmodel`.
5. L2-normalize its raw `[1,512]` output on the CPU, then compare with the
   offline map descriptor bank.

The KPU models end before CPU post-processing. Do not feed camera bytes
directly to either model: the input contracts are explicit and static.

## Host test vectors

`test_vectors/` contains three deterministic normalized-input vectors and
expected ONNX outputs for each model. They are for checking the board's
input/output wrapper and dequantization. Use the exact input ordering and
output layout; do not compare raw integer buffers before applying the
runtime's quantization metadata.

## Qualification boundary

ONNX checker/shape/ORT and nncase PTQ/compile/gencode have passed for both
artifacts. SuperPoint has a frozen faithful INT8 quality test PASS. MixVPR has
frozen DEV and faithful-INT8 evidence, but an independent held-out TEST
protocol is still blocked. Actual K210 SRAM peak, latency, camera I/O, power,
and sustained stability remain unknown until the board test.
