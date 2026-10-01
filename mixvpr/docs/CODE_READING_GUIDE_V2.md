# MixVPR V2 — Code Reading Guide

Read the new pipeline in this order.

## 1. `configs/v2_progressive.json`

Start here. It defines:
- what never changes,
- which dimensions are pruned,
- exact stage order,
- training/recovery budgets,
- teacher/data locations,
- K210 feasibility guardrails.

If the architecture changes, this is the first file to inspect.

## 2. `src/mixvpr_k210/config.py`

This converts JSON stages into `ModelSpec` objects and enforces invariants.

Important checks:
- every student is D4,
- every student is 512D,
- exactly one pruning axis changes between adjacent stages,
- capacity never increases after stage 00.

## 3. `src/mixvpr_k210/model.py`

This is the only new model implementation.

Read:
1. `ConvStage`
2. `FeatureMixer`
3. `K210MixVPR.raw()`
4. `K210MixVPR.forward()`

Dataflow:

```
RGB 240x240
 -> five ConvStage blocks
 -> static 10x10 grid
 -> 4 FeatureMixer blocks
 -> channel projection
 -> row projection
 -> raw 512D
 -> L2 normalize
```

## 4. `src/mixvpr_k210/losses.py`

Two learning signals:
- task retrieval loss,
- ranking KD.

The student never tries to numerically copy a 4096D teacher vector.
Instead it learns the teacher's pairwise similarity/ranking structure.

## 5. `src/mixvpr_k210/pruning.py`

This is the heart of progressive compression.

Read:
- `prune_w5`
- `prune_w4`
- `prune_hidden`

The code:
1. computes channel/neuron importance,
2. selects important indices,
3. slices matching producer/consumer weights consistently,
4. copies all unaffected tensors exactly.

No stage is reinitialized from scratch after pruning.

## 6. `src/mixvpr_k210/trainer.py`

This implements the single-student lineage.

Important functions:
- `master_plan`: one non-overlapping data schedule for all stages,
- `initialize_first_stage`: random init only once,
- `initialize_pruned_stage`: creates the next stage from previous weights,
- `train_stage`: task loss + teacher KD + optional previous-stage KD,
- `run_stage`: enforces lineage order.

## 7. `src/mixvpr_k210/export.py`

This is the PyTorch -> ONNX boundary.

The deployment graph emits raw 512D only. CPU L2 is intentionally outside
the KPU graph.

The exporter also checks ONNX Runtime parity.

## 8. `scripts/v2/nncase_compile.py`

This runs only in the pinned K210 nncase environment.

It performs:
- ONNX import,
- TRAIN calibration PTQ,
- compile,
- gencode,
- mapping audit,
- K210 simulator execution.

A valid graph currently requires:
- 20 KPUConv2D,
- 0 CPU Conv2D,
- simulator PASS.

## 9. `src/mixvpr_k210/feasibility.py`

High-level wrapper around ONNX export + the pinned nncase backend.

Use it after a training/recovery stage to answer:

> Can this exact architecture enter the K210 deployment region?

## 10. `tests/test_v2_lineage.py`

Read this after the implementation.

It proves:
- the final geometry matches frozen D4 params/MACs,
- the readable V2 refactor is numerically identical to the legacy D4,
- W5/W4/H pruning copies the intended weight slices,
- every student remains D4/512D.

## 11. `./mixvpr/run`

This is the only command interface new work should use.

Typical learning workflow:

```bash
./mixvpr/run check
./mixvpr/run status
./mixvpr/run smoke
./mixvpr/run train 00_large          # mandatory 1000-update probe
# inspect DEV trend; do not continue automatically
./mixvpr/run train 00_large --continue-after-probe
./mixvpr/run feasibility 00_large
# then progressively run the next stage
```

The parent probe is a hard workflow gate: stage00 must demonstrate a credible
capacity/training signal before the remaining 5000-update budget is allowed.

## Historical files

Do not use old pilots as the new workflow. Keep them only as evidence and
for reproducing earlier claims.

The goal of V2 is one readable implementation, one lineage, one CLI, and
explicit evidence at every gate.
