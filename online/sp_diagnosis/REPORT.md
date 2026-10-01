# SuperPoint / online diagnosis and laptop optimization

## Findings
The original SuperPoint network can localize against the existing map when query scale is sufficient. This does not establish a need for a new architecture. Two implementation/configuration issues were found: the wrapper cap did not reach net.config, and small landscape query input substantially differed from the portrait/high-resolution map extraction.
The cap bug was fixed (net.config plus score ordering). The controlled cap-only ablation did NOT recover pose quality; it is not claimed as the primary explanation. Small portrait184x320 also failed, so distortion alone does not explain the problem. Increasing spatial detail to portrait360x640 was the effective change under the tested map/matcher.

## Tuning (first 10 registered project DEV images)
| Configuration | Good poses /10 | Median correspondences | Median frame ms |
|---|---:|---:|---:|
| legacy_landscape | 0 | 2.0 | 197.2 |
| fixed_topk_landscape | 0 | 2.0 | 195.5 |
| fixed_topk_portrait | 0 | 1.0 | 155.7 |
| portrait_medium | 10 | 28.0 | 576.6 |
| legacy_scale_reference | 10 | 70.5 | 1572.1 |

## Confirmation (remaining 31 DEV images)
Legacy320x240: 0 good poses. Selected PyTorch360x640: 30 good poses. FP32 ONNX360x640: 31 poses returned, 30 good poses. Good pose means >=8 inliers, <=5deg and <=0.1 native reconstruction units; one query exceeds center-error guard.
Selection used only the first10; the remaining31 did not change resolution/thresholds. These images belong to the reconstruction and have previously been used in development; this is NOT an untouched independent test. Query-to-map self-overlap can make results optimistic.

## CPU backend optimization
Local-stage benchmark medians on five tuning images: {'torch1': 544.0350505000424, 'torch2': 334.44796649996533, 'onnx2': 204.55738399982692} ms. Warm-up and two repeats per image; ONNX keeps >=99% of exact keypoint positions and matched descriptor cosine>0.9999 under asserted parity.
Confirmation median frame processing: 213.10 ms; p95 258.16 ms. Median ratio vs PyTorch1: 2.57x. Timing excludes image read/decode, capture queues and board transport. Reciprocal median is not a measured camera throughput.
Detailed per-stage/per-frame records are in onnx_confirmation.json. Query cap160, active600, topK3, matcher threshold.8/margin.05 and PnP parameters unchanged. CPU two threads; no GPU or new training.

## Ready-to-run development replay
```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 /home/quyet/k210_lab/.venv-mixvpr-cuda/bin/python /home/quyet/k210_lab/online/sp_diagnosis/confirm_onnx.py
```
The script uses real development frames, per-image COLMAP calibration, the frozen MixVPR and actual SuperPoint inference. It reports reconstructed-pose consistency, not independent ground truth.

## Live video runner
```bash
python /home/quyet/k210_lab/online/run.py \
  --map /home/quyet/training_recovery/final_gate2/real_map \
  --model /home/quyet/training_recovery/final_gate2/candidate.pt \
  --config /home/quyet/k210_lab/online/config_laptop.json \
  --local-backend onnx \
  --local-model /home/quyet/k210_lab/online/sp_diagnosis/superpoint_fp32_360x640.onnx \
  --threads 2 --allow-legacy-local-map \
  --video DEVELOPMENT_VIDEO --calibration VERIFIED_CAMERA_JSON --output poses.jsonl
```
Use the CUDA-capable project Python environment above even though inference is CPU-only. Camera calibration must match actual video resolution. The explicit legacy-local-map flag labels the unverified map provenance; it does not disable global model identity checks. Do not relabel this map as independently validated.

## K210 / distillation decision
The selected high-resolution original SuperPoint FP32 ONNX is a laptop baseline, NOT a K210 deployment claim. Existing R0 reports show the FR12 L1-initialized small model has poor geometric matching; descriptor cosine close to1 alone did not imply useful descriptors. Do not plug it into the final system as a quality-proven model.
Before starting a new architecture search, the next K210 experiment should compare the existing compiler-proven small architecture under supervised/geometric training versus teacher KD at equal budget, using original SuperPoint as frozen teacher. Use TRAIN-only homographies/observations and separate DEV. Evaluate detector repeatability, geometrically verified matching precision and PnP, plus descriptor diversity to detect collapse. A cross-resolution/map compatibility control is needed because KD cannot guarantee recovery of detail removed by resizing.
This KD experiment was not started: current laptop quality/speed issue was resolved without changing weights. No new architectural or board-readiness claim is made.
