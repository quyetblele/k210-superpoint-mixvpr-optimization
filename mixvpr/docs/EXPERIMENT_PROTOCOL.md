# MixVPR → K210 Research Experiment Protocol

## Purpose

This file defines the canonical research workflow for the V2 student lineage.
The goal is not merely to make a smaller model, but to identify the strongest
512-D student that satisfies K210 deployment constraints with reproducible
evidence.

The workflow follows ideas used in modern VPR distillation, progressive
compression, and function-preserving network expansion:

- relational/ranking knowledge rather than raw-vector MSE across unequal
  descriptor dimensions;
- one controlled structural change per compression stage;
- validation before and after pruning/recovery;
- function-preserving warm-start instead of random restart;
- deployment evidence as part of model selection.

## Frozen invariants

- Teacher: frozen MixVPR ResNet50, 4096-D.
- Student descriptor: 512-D at every stage.
- Input: 240 × 240 RGB.
- Mixer depth: D=4.
- Projection: P=128.
- Tokens: 100.
- TEST is sealed until final model selection.

## Evidence hierarchy

Every experiment must separate four questions.

1. **Initialization/function gate**
   - Does widening preserve the frozen D4 function?
   - Record raw-output and normalized-output parity.

2. **Optimization/KD gate**
   - Record task loss, teacher-KD loss, parent-KD loss.
   - Record gradient norm.
   - Diagnose task-vs-KD gradient cosine and weighted gradient-norm ratio.
   - Compare teacher KD on clean versus augmented student inputs.

3. **Retrieval gate**
   - Evaluate Indoor, Project, and Overall R@1/R@5/R@10 plus margin.
   - Stage00: evaluate at 0/250/500/750/1000 during the capacity probe.
   - Pruned stages: evaluate immediately after pruning and during recovery.

4. **Hardware gate**
   - PyTorch → ONNX parity → PTQ INT8 → nncase compile → gencode → simulator.
   - A training scaffold is allowed to fail K210 gencode.
   - Stop compression at the strongest recovered stage that passes deployment.

## Stage00 capacity-proof rule

Stage00 starts from the frozen D4 checkpoint using function-preserving widening:

D4 [16,24,64,112,160], H96
→ large [16,24,64,128,192], H128.

The first run is limited to 1000 optimizer updates.
Do not continue to the 6000-update budget merely because training loss falls.

The capacity hypothesis is supported only if validation shows a credible gain
over the step-0 D4-equivalent initialization. If the best checkpoint remains
step 0, or validation degrades consistently, stop and diagnose the objective,
KD signal, sampling, or augmentation before any long run.

## Progressive compression rule

Use exactly one structural action per stage:

- 01: W5 192 → 176
- 02: W4 128 → 120
- 03: H 128 → 112
- 04: W5 176 → 160
- 05: W4 120 → 112
- 06: H 112 → 96

Weights must be inherited from the immediately previous best checkpoint.
Never random-reinitialize after pruning.

## Recovery signals

For stage00:
- retrieval task loss;
- frozen 4096-D teacher relational/ranking KD.

For every pruned stage:
- retrieval task loss;
- frozen 4096-D teacher relational/ranking KD;
- immediate 512-D parent relational KD.

Interpretation:
- the frozen teacher supplies global retrieval structure;
- the immediate parent preserves the local function that existed before pruning;
- the task objective keeps the student tied to the actual retrieval labels.

Loss weights are experimental parameters, not truths.
A nominal 1:1 weight does not imply equal gradient influence.

## Required controlled ablations after a failed capacity probe

Change one factor at a time:

A. current augmentation + current KD;
B. reduced/disabled augmentation + same KD;
C. same augmentation + adjusted teacher-KD weight;
D. same augmentation + altered KD temperature;
E. if needed, stronger teacher relational context/hard negatives.

Do not combine B+C+D in one run.

## Decision record for every experiment

Every experiment directory must contain enough evidence to answer:

- hypothesis;
- single controlled change;
- parent/checkpoint hash;
- data-plan hash;
- seed;
- training budget;
- loss configuration;
- step-0 metrics;
- intermediate/final DEV metrics;
- best checkpoint;
- deployment result when applicable;
- decision: continue / stop / revise objective.

A failed experiment is retained when it rules out a hypothesis.
Do not keep redundant binary artifacts merely because the experiment failed.

## Research references to study

1. Net2Net: function-preserving network expansion.
   https://arxiv.org/abs/1511.05641

2. DistilVPR (AAAI 2024): relational/manifold knowledge distillation for VPR.
   https://ojs.aaai.org/index.php/AAAI/article/view/28905

3. TSCM (ICRA 2024): cross-metric knowledge distillation for VPR.
   https://arxiv.org/abs/2404.01587

4. D2-VPR (AAAI 2026): staged VPR distillation and recovery/fine-tuning.
   https://ojs.aaai.org/index.php/AAAI/article/view/38303

5. Progressive pruning/distillation literature should be used for methodology:
   prune one controlled structure, measure immediate damage, recover, measure again.

These references motivate the protocol; their hyperparameters are not copied
blindly because architecture, data, and deployment constraints differ.
