# Gate 2 nncase compiler summary

## Environment

- Python: 3.8.20
- nncase: 1.8.0.20220929
- _nncase: 1.8.0-55be52f
- ONNX: 1.14.1
- NumPy: 1.24.4
- Target: k210

## Shape-inference remediation

- Original import failure: `Can't find value info for conv1_out to parse its shape`
- Source model: `models/tiny_fp32.onnx` (its serialized `graph.value_info` was empty)
- Remediated model: `models/tiny_fp32_inferred.onnx`, produced by ONNX shape inference and checker validation
- Serialized intermediate value_info:
  - conv1_out: FLOAT [1, 8, 32, 32]
  - relu1_out: FLOAT [1, 8, 32, 32]
  - conv2_out: FLOAT [1, 8, 32, 32]

## Configuration

- Quantization: uint8 activations and uint8 weights (nncase v1 PTQ)
- Calibration: 10 FP32 samples, shape (10, 3, 32, 32), range [-1.0, 1.0], fixed seed 20260905
- Debug dumps: IR, ASM, and quantization error under `reports/gate2_nncase/`

## Results

- ONNX import: PASS
- PTQ: PASS
- Compile: PASS
- Kmodel generation: PASS
- Kmodel size: 3208 bytes
- Kmodel SHA256: `b8bdd051c3f7d7edcc33374e356702659518121493e809f4fc1c3138d5a673b1`

## Warnings / errors

- Synthetic calibration data is only for this compiler smoke test; real deployment requires representative calibration data.
- nncase emitted non-fatal code-generation warnings: `Cannot find a decompiler for section .rdata` (twice) and `Cannot find a decompiler for section .text` (twice).
