# MixVPR → K210: Canonical V2 Workflow

This directory now has two clearly separated parts.

## 1. Preserved historical evidence

Do not move or rewrite files referenced by frozen manifests/protocol hashes.

Important preserved evidence:
- `artifacts/d4_final_candidate/`
- `artifacts/depth_pilot/`
- `artifacts/stage1_frontier/`
- `pitts30k_benchmark.py`
- `controlled_depth_pilot.py`
- `graph_frontier.py`

The old D4 baseline remains the comparison floor. V2 does not delete or
silently replace it.

## 2. Canonical V2 implementation

Use only this path for new work:

```
configs/v2_progressive.json
src/mixvpr_k210/
scripts/v2/
tests/
docs/
run
```

Entry point:

```bash
./mixvpr/run status
./mixvpr/run check
./mixvpr/run plan
./mixvpr/run kd-diagnostic --batches 8
./mixvpr/run smoke
./mixvpr/run feasibility 06_prune_h
./mixvpr/run train 00_large
```

## V2 idea: one student lineage

There is one 512-D student. It is not recreated from random weights after
each pruning stage.

```
Frozen D4 512D
    |
    | function-preserving widening
    v
00_large  [16,24,64,128,192], H128, D4, 512D
    ^
    |
Teacher 4096D -- relational/ranking KD
    |
    | prune W5 + recovery KD
    v
01_prune_w5 [16,24,64,128,176], H128
    |
    | prune W4 + recovery KD
    v
02_prune_w4 [16,24,64,120,176], H128
    |
    | prune H + recovery KD
    v
03_prune_h  [16,24,64,120,176], H112
    |
    | prune W5
    v
04_prune_w5 [16,24,64,120,160], H112
    |
    | prune W4
    v
05_prune_w4 [16,24,64,112,160], H112
    |
    | prune H
    v
06_prune_h  [16,24,64,112,160], H96
```

Invariants across every student stage:
- input: RGB 240x240
- mixer depth: D4
- tokens: 100
- projection: 128
- output descriptor: 512D

Only W4, W5, and mixer hidden width H are progressively pruned.

## Training versus deployment

Training stays in PyTorch:

```
.pt -> KD -> prune -> recovery KD -> next .pt
```

ONNX is a deployment checkpoint, not a training format:

```
candidate .pt
  -> raw 512D ONNX
  -> ONNX Runtime parity
  -> nncase PTQ INT8
  -> compile/gencode
  -> K210 simulator
  -> .kmodel
```

The KPU graph emits raw 512D FP32. Final L2 normalization remains CPU-side,
matching the frozen deployment contract.

## Stop rule

Do not force the lineage to reach the smallest stage.

After a recovered stage:
1. measure DEV quality,
2. export ONNX,
3. run nncase/K210 feasibility,
4. stop when quality and hardware constraints are both satisfactory.

Independent TEST must not be used for model selection or tuning.

## Mandatory parent capacity probe

The first `./mixvpr/run train 00_large` invocation stops at 1000 updates and
records `PROBE_COMPLETE`; it does not silently continue to the 6000-update
budget. Review DEV evidence first. Only if the large student shows a credible
capacity/training signal should training continue with:

```bash
./mixvpr/run train 00_large --continue-after-probe
```

This gate exists because the earlier widened-parent pilot degraded after 250
and 500 updates; larger capacity alone is not evidence of a better student.

## Research protocol

All new experiments must follow `docs/EXPERIMENT_PROTOCOL.md`.
It defines frozen evidence, KD diagnostics, one-variable-at-a-time ablations,
pruning damage/recovery gates, hardware evidence, and stop/rollback rules.
