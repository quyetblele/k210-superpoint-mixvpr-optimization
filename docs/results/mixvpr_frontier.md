# MixVPR Stage 1 — baseline audit and graph frontier

2026-09-12. Six untrained higher-capacity graphs plus the frozen trained baseline. No optimizer updates, KD, quality evaluation on new graphs, TEST access, model promotion or board measurements.

## Current baseline

- Source checkpoint: `/mnt/d/k210_gate2/seed_20260912/KD/best.pt`; SHA256 `d788f6fd49d28fc7e3f171ac9cbddd20920dae0ec9d3c53351e0ada440512e42`. Best +4750 /5750 cumulative updates; immutable copy reused. All14 identity dependency hashes verified.
- Architecture: custom Conv backbone [16,24,64,112,160], ten3x3 Conv; first stride2, three2x2 pools; final15x15 -> MaxPool6/stride1 ->10x10. Four token mixers (LayerNorm100,100→96→100,residual), channel160→128, row100→4,512-D descriptor.
- Preprocess: PIL RGB direct bicubic 240x240, /255, ImageNet mean [.485,.456,.406], std [.229,.224,.225]; FP32 pre-L2 output; CPU L2 eps1e-12
- Tensor path: [1,3,240,240] → stem conv[1,16,120,120] → pooled stages[1,16,60,60],[1,24,30,30],[1,64,15,15] → late[1,112,15,15],[1,160,15,15] → [1,160,100] → [1,100,128] → [1,128,4] → [1,512].
- Parameters731,300; Conv+Linear258,553,600MACs per image. Pool/LayerNorm/nonlinearity/L2 work excluded; MAC is not measured latency.
- Existing canonical ONNX reused and freshly checked against the actual checkpoint on3 TRAIN images. Recompiled baseline kmodel is byte-identical to the old trained artifact. Actual nncase1.8.0.20220929 K210 simulator passes3 TRAIN inputs. Both activations/weights use uint8 PTQ internally with FP32 I/O wrappers.
- Mapping:20 KPUConv2D =10 backbone +8 mixer +2 projections;0 CPUConv. CPU LayerNorm4, window reduction/grid adapter, transforms, quant/dequant and external L2 remain. KPU/CPU counts below count convolution ops, not all execution.

## Historical quality, not rerun in Stage1

| Protocol | FP32 R@1 / R@5 / R@10 | Faithful8bit R@1 / R@5 / R@10 |
|---|---|---|
| Indoor DEV, scene-macro |83.92 /94.41 /96.70%| Not recorded in this audit |
| Project DEV |88.89 /94.44 /94.44%| Not recorded in this audit |
| Floor6 engineering retrieval |15.00 /33.33 /38.33%|15.00 /31.67 /36.67%|
| Floor6+7 engineering retrieval |29.84 /41.94 /46.77%|29.03 /41.13 /45.97%|

Historical Floor6/Floor7 retrieval protocol is not the canonical20-query SuperPoint pose protocol. Old stair-map localization0/10 is invalid for model decisions and is not reused. Historical project-stair image retrieval metrics are separately labeled and do not validate that retired map. No untouched independent MixVPR TEST claim.

## Bottleneck classification

**FACT**
- Original normalized FP32 canonical MixVPR full graph and separate backbone/aggregator compile PASS but gencode OOM.
- Original stem PASS then first layer1 block OOM; aggregator flatten PASS then first mixer OOM. These identify failing prefixes, not a unique allocation cause.
- Historical M3 backbone passes at160; tested aggregator surrogates still fail.
- Export DOUBLE dtype failure is separate and historical; FP32 canonicalization passes import/compile.
- Current compact baseline trained kmodel generates and executes; 20 KPU Conv,0 CPU Conv,CPU LayerNorm/data movement plus external final L2.
- Historical Floor6 retrieval R1 FP32=15%,8bit=15%; combined R1 retention97.30%.

**INFERENCE**
- Current deployment-domain retrieval gap is more important than measured 8bit loss on that historical benchmark; it does not isolate capacity versus data/training/evaluator.
- Higher C/H tensors may exceed compiler allocation limits even with unchanged early width; new frontier probes test feasibility only.

**UNKNOWN**
- Current canonical Floor6 retrieval quality under a newly locked independent protocol.
- Whether backbone capacity, token mixer, training data, supervision or label protocol dominates current quality.
- CPU/KPU stage latency, real board peak RAM, KPU memory reserve and simultaneous full-system fit.
- Whether any untrained candidate will outperform current quality.

## Controlled graph frontier

Fixed:240x240 input, first four widths[16,24,64,112],100tokens,512D output, same64 hashed TRAIN calibration images, nncase settings and3 simulator inputs. New weights seeded20260912; no gradient or optimizer step. “Higher-capacity” is not evidence of stronger retrieval.

C=last channel width; H=token-MLP hidden; D=mixer depth; P=channel projection width; output rows=512/P. projection256 redistributes512D into256x2; it is not uniformly more expressive than128x4.

| Candidate | C/H/D/P | Params | MACs(M) | Export | Compile | Gencode | Sim | KPU/CPU Conv | TOTAL(B) | Kmodel(B) | Graph gate |
|---|---|---:|---:|---|---|---|---|---|---:|---:|---|
| baseline | 160/96/4/128 | 731,300 | 258.554 | PASS | PASS | PASS | PASS | 20/0 | 2947880 | 766632 | PASS |
| late192 | 192/96/4/128 | 869,092 | 291.488 | PASS | PASS | FAIL | NOT_RUN | Unavailable | — | — | FAIL |
| late224 | 224/96/4/128 | 1,025,316 | 328.570 | PASS | PASS | FAIL | NOT_RUN | Unavailable | — | — | FAIL |
| mixer128 | 160/128/4/128 | 757,028 | 262.650 | PASS | PASS | FAIL | NOT_RUN | Unavailable | — | — | FAIL |
| depth6 | 160/96/6/128 | 770,492 | 264.698 | PASS | PASS | PASS | PASS | 24/0 | 2997224 | 815976 | PASS |
| projection256 | 160/96/4/256 | 751,706 | 260.602 | PASS | PASS | FAIL | NOT_RUN | Unavailable | — | — | FAIL |
| late192_mixer128 | 192/128/4/128 | 894,820 | 296.403 | PASS | PASS | FAIL | NOT_RUN | Unavailable | — | — | FAIL |

All five failed candidates fail at gencode with `Allocator has ran out of memory`; simulator is NOT_RUN because no executable model exists. Their TOTAL and execution mapping are unavailable, not zero. No evidence identifies the exact failed memory pool, so do not infer one from kmodel size.

## Shortlist and stop

**Only depth6 qualifies for a short pilot; do not fill the shortlist with failed graphs.** Use baseline D4 as control. D6 preserves the other dimensions, adds39,192 parameters and6.144M MACs; compiled model/TOTAL increase49,344B. It tests mixer depth rather than wider simultaneous tensors.

Provisional shortlist screen: compiler TOTAL≤3MiB, declared before the probes. Baseline2,947,880B; D6=2,997,224B,148,504B below this screen. This is not measured SRAM peak, a board-fit guarantee or proof both models plus the application fit together. Real CPU/KPU timing and buffer lifecycle remain unknown.

One next decision: user chooses whether to run controlled short pilot D4 versus D6. No pilot has started; no full training/KD/TEST authorized by this stage.

## Verification and provenance

-7 ONNX/reference checks on3 TRAIN inputs each PASS; all7 compile PASS;2 gencode/simulator PASS;5 allocator FAIL recorded. Successful outputs finite, nonzero and[1,512]. Independent arithmetic MAC checks PASS.
-Baseline regenerated kmodel hash equals historical artifact. All9 protected checkpoint/graph/calibration/SuperPoint evidence hashes unchanged; historical evidence hashes unchanged;14 baseline identity dependencies and64 TRAIN image hashes verified.
-A metadata counting bug initially expected only10 KPU Conv and incorrectly flagged the baseline. Correct expectation includes mixer/projection:12+2D. Raw flags remain in results.json; corrected authoritative flags are in summary.json, with mapping_count_correction.json. No graph or acceptance tolerance changed.
-Evidence: baseline_audit.json, protocol.json, summary.json, each candidate export.json/compile.json/compile.log/dump. Runner: ../../graph_frontier.py. No changes to legacy models, MixVPR default pointer, training protocol or SuperPoint release.
