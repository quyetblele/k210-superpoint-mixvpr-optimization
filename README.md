# K210 deployment lab

## Gate 1: minimal static ONNX graph

Gate 1 proves that the pinned environment can create and inspect a simple,
static FP32 ONNX graph before any K210 `.kmodel` compilation. The environment
is Python 3.8.20, nncase 1.8.0.20220929 (`_nncase` 1.8.0-55be52f), and ONNX
1.14.1.

Create the model:

```bash
/home/quyet/miniconda3/bin/conda run -n k210 python scripts/create_tiny_onnx.py
```

Inspect and validate it:

```bash
/home/quyet/miniconda3/bin/conda run -n k210 python scripts/inspect_onnx.py
```

Expected PASS criteria: `models/tiny_fp32.onnx` exists; the ONNX checker
passes; input is `input` FP32 `[1, 3, 32, 32]`; output is `output` FP32
`[1, 8, 32, 32]`; the graph contains only two `Conv` and two `Relu` nodes;
and it has no dynamic shapes or custom operators.

## Gate 2: nncase K210 compiler path

Gate 2 imports the Gate 1 model, runs nncase v1 PTQ, and produces a K210
`.kmodel` on the PC. It does not run the model on hardware. The calibration
set is deterministic synthetic FP32 data for a compiler smoke test only; a
real model requires representative calibration data.

```bash
/home/quyet/miniconda3/bin/conda run -n k210 python scripts/infer_tiny_onnx_shapes.py
/home/quyet/miniconda3/bin/conda run -n k210 python scripts/compile_tiny_k210.py
```

The first command serializes ONNX shape-inference metadata to
`models/tiny_fp32_inferred.onnx`; the original Gate 1 model is unchanged.
The compiler imports that inferred model using uint8 PTQ, then writes
`artifacts/tiny.kmodel`. This workflow passed with a non-empty K210 kmodel;
`reports/gate2_summary.md` records its size and SHA256. Compiler dumps are
written to `reports/gate2_nncase/`.
