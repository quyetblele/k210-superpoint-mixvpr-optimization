# Literature Notes for MixVPR V2 Compression

These notes capture research ideas that inform the project. They are not a license to copy paper hyperparameters blindly.

## Net2Net - function-preserving widening

Reference: Chen, Goodfellow, Shlens, "Net2Net: Accelerating Learning via Knowledge Transfer" (2015), arXiv:1511.05641.

Core lesson: when moving to a wider student, preserve the already learned function instead of restarting from random weights.

Project use:
- frozen D4 512-D student is the source function
- stage00 is widened only along W4/W5/H
- parity is checked before any KD update
- larger capacity must prove value during the 1000-update probe

We borrow the principle, not the exact Net2Wider construction.

## DistilVPR - relational knowledge matters in VPR

Reference: Wang et al., "DistilVPR: Cross-Modal Knowledge Distillation for Visual Place Recognition", AAAI 2024. Official code: https://github.com/sijieaaa/DistilVPR

Core lessons:
- VPR distillation benefits from feature relationships, not only pointwise feature regression
- multiple relation geometries/signals can be useful
- distillation weights are task/dataset dependent
- ablation is required to justify each loss term

Project use:
- teacher 4096-D and student 512-D are compared through similarity/ranking structure
- no raw vector MSE is required across unequal descriptor dimensions
- KD weights will be measured through loss and gradient diagnostics instead of copied from a paper

## TSCM - cross-metric teacher/student VPR distillation

Reference: Shen, Liu, Lu, Chen, "TSCM: A Teacher-Student Model for Vision Place Recognition Using Cross-Metric Knowledge Distillation", ICRA 2024, arXiv:2404.01587. Official code: https://github.com/nubot-nudt/TSCM

Core lessons:
- a strong teacher and lightweight deployment student can use different metric representations
- distillation should target retrieval behavior, not merely identical internal dimensions
- VPR evaluation and ablation are central evidence

Project use:
- fixed teacher remains the global semantic signal throughout the lineage
- student interface remains 512-D for deployment compatibility
- DEV retrieval evidence decides whether KD/recovery is useful

## D2-VPR - distillation then fine-tuning

Reference: Zhang et al., "D2-VPR: A Parameter-efficient Visual-foundation-model-based Visual Place Recognition Method via Knowledge Distillation and Deformable Aggregation", AAAI 2026. Official code: https://github.com/tony19980810/D2VPR

Core lessons:
- parameter-efficient VPR can use staged distillation and subsequent fine-tuning
- teacher/student feature-space gaps may need explicit recovery mechanisms
- deployment efficiency should be evaluated together with retrieval quality

Project use now:
- keep the current relational KD baseline simple and measurable
- do not add a new recovery module yet

Future controlled ablation:
- KD for most of recovery, followed by a final task-only polishing phase
- compare against KD-throughout under the same data/seed/stage

## What we deliberately do not copy yet

Do not import paper-specific loss weights, extra modules, pruning criteria, or schedules before the baseline is measured.

Current controlled baseline:
1. function-preserving stage00 initialization
2. retrieval task loss
3. fixed-teacher relational/ranking KD
4. immediate-parent relational KD after pruning
5. one-axis structured pruning using deterministic L2-based importance
6. DEV evaluation after every controlled change
7. K210 hardware feasibility after recovered checkpoints

Only one new research idea should be introduced per ablation family.

## Learning principle

Read papers to extract hypotheses, measurements, and experimental controls. Do not treat published hyperparameters as universal constants.
