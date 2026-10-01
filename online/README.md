> Stair runtime maps were deleted by user request on2026-09-10. Historical stair commands below are retired. Floor6 is selected for SuperPoint evaluation; a validated floor6 runtime descriptor bank is not yet built.

# Bounded online localization: PC reference

This module leaves Gate 2 unchanged. It does not access GT1, score accuracy, use LightGlue, FAISS, or a GPU. All reported timings are PC timings. The smoke test uses explicit mock descriptors and synthetic geometry; it is not a trained-model test.

## Run the smoke test

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/quyet/k210_lab/.venv-mixvpr-cuda/bin/python /home/quyet/k210_lab/online/smoke.py
```

`smoke_report.json` records actual results. The test creates a temporary three-frame video and a tiny map, runs actual OpenCV decoding and PnP, verifies known pose, empty-feature failure, calibration resolution checks, bounded active sets, deduplication and retrieval reuse. Temporary artifacts are removed automatically.

## Dataflow and contracts

Original BGR uint8 camera frame -> two independent preprocessing branches:

* Global: RGB PIL bicubic 240x240 -> ImageNet normalization in model adapter -> pre-L2 float32 `[512]` -> CPU L2 -> streamed reference cosine Top-K -> bounded covisibility expansion -> bounded visible Point3D rows.
* Local: original frame -> grayscale at configurable SuperPoint size -> SuperPoint keypoints `[Q,2]` and descriptors `[Q,256]`. Map keypoint coordinates back to the calibrated original image with the resize pixel-center transform. Never feed the global 240x240 image into SuperPoint.

Local descriptors -> block cosine nearest and second-nearest Point3D -> threshold and optional absolute cosine gap -> one strongest query per Point3D ID -> strongest capped correspondences -> OpenCV PnP-RANSAC -> world-to-camera R,t and world camera center `-R.T @ t`. Distortion and K refer to the ORIGINAL camera frame size. Resolution mismatch fails explicitly.

`config.json` is an engineering starting configuration, not GT1-tuned. SuperPoint 320x240 is a replaceable PC engineering setting; this is not a claim that this resolution is the final deployed K210 choice. The CPU adapter preserves existing SuperPoint NMS=3, threshold=.005, border=4 and fix_sampling=False, with keypoint cap reduced/configurable. Query/map local descriptors must use a compatible extraction recipe.

## Map v1

A directory with `map.json` and uncompressed `.npy` arrays, opened using read-only memory maps:

| File | Shape / type | Meaning |
|---|---|---|
| global.npy | `[N,512]` float32 or float16 | reference descriptors |
| image_ids.npy | `[N]` int64 | external image IDs |
| point_ids.npy | `[P]` int64 | external Point3D IDs; preferably unique |
| xyz.npy | `[P,3]` float32 | world XYZ |
| local.npy | `[P,256]` float32 or float16 | representative SuperPoint descriptor per Point3D |
| vis_offsets.npy | `[N+1]` int64 | CSR offsets |
| vis_rows.npy | `[V]` int64 | point ROW indices, not external IDs |
| cov_offsets.npy | `[N+1]` int64 | CSR offsets |
| cov_rows.npy | `[E]` int64 | image ROW indices ordered by covisibility priority |

Metadata requires version=1, global_identity, local_identity. Also record coordinate_frame/units, map provenance, calibration and local descriptor aggregation recipe. Descriptor identities must match query backends; mismatches are refused. Replacing final global checkpoint requires recomputing reference global descriptors (not duplicating XYZ/local arrays). ONNX/board parity may later justify shared semantic identities only after verification; no automatic override is provided.

No existing large map is copied or converted by this implementation. Build the arrays once from the project's validated reconstruction and feature tracks. The full global descriptor matrix is scanned in blocks; local descriptors are read only for selected rows in blocks. Active rows are capped. CSR arrays remain disk-backed. For an embedded port, use the same little-endian payloads with an explicit reader/header contract or transcode the small NPY headers; Python/NumPy/OpenCV themselves are not K210 firmware.

## Model adapters and online runner

`backends.py`: CPU PyTorch student, optional ONNX Runtime pre-L2 adapter, existing SuperPoint CPU adapter, and injected board transport interface. The latter has NO simulated INT8 fallback: a real transport must implement verified input quantization and output dequantization. Official 4096D teacher descriptors cannot be plugged into a 512D student map.

```bash
python run.py --map MAP_DIRECTORY --model FROZEN_CHECKPOINT.pt \
  --backend fp32 --video DEVELOPMENT_VIDEO.avi \
  --calibration camera.json --output poses.jsonl --max-frames 100
```

`camera.json`: `{"size":[640,480],"K":[[fx,0,cx],[0,fy,cy],[0,0,1]],"distortion":[k1,k2,p1,p2,k3]}` with real numerical calibration values. Use `--camera 0` instead of `--video` for capture. A camera/video source and map must come from ordinary development data; no test tuning.

Per-frame JSONL contains timestamps, processing status, retrieval reuse, Top-K image IDs, active points, query features, raw/deduplicated matches, PnP inliers, pose, and stage timings. Skipped/reused stages are absent rather than charged with fictitious timings. Source/model/map initialization errors stop with an error; per-frame feature/PnP errors return structured failure states.

## Resource limits and remaining work

At default Q=160/block=64, the score block is ~40 KiB; local block float32 descriptors ~64 KiB; query descriptors ~160 KiB, before normalization temporaries. Limits do not include frame/model/runtime memory. mmap bounds application allocations but does NOT prove OS page-cache or physical SRAM limits. OpenCV PnP is a PC reference, not a proven K210 solver. Current nearest/second-nearest updates are intentionally straightforward for C portability, not speed-optimized.

Remaining integration: real map import and descriptor aggregation with provenance, final trained global checkpoint/INT8 weights, verified SuperPoint deployed configuration and postprocessing parity, actual K210 transport, portable PnP, measured board memory/latency, and independent labeled evaluation. The passing synthetic smoke validates software wiring only. Do not claim an operational board system or localization accuracy from it.

## Pre-MixVPR preparation completed

`maps/stair_local_v1` imports the existing stair development map (205 references, 7,770 points) without modifying its source. Approximately 8.66 MB of local payloads were written; no model run, frame extraction, pruning or quantization was needed. `prepare_map.py` streams NPZ payload extraction and builds visibility/covisibility from COLMAP tracks. Full covisibility is preserved for the baseline; runtime expansion remains capped.

`validate_local.py` exercises a real active subset (452 points for reference row 0) and compares streamed matching with a small dense reference. Query descriptors are map self-descriptors: this is an implementation check, NOT localization evaluation. Geometry/scale remains in the source COLMAP coordinate frame; do not report metric pose errors until scale is established.

The local metadata deliberately says `UNVERIFIED_LEGACY_SUPERPOINT`. The source package is a legacy Pi5 package; its included INT8 ONNX files are NOT accepted as K210 INT8 models. Verify source extraction weights/preprocessing/sampling compatibility or re-extract local map descriptors for the selected query backend. Do not hide that dependency by merely changing the metadata identity string.

When final student and local provenance are ready, bind all reference global descriptors in stable image order:

```bash
python bind_global.py --map maps/stair_local_v1 \
  --images ORIGINAL_REFERENCE_IMAGE_ROOT --model FINAL_FROZEN_STUDENT.pt \
  --backend fp32 --local-proof VERIFIED_LOCAL_PROVENANCE.json
```

The proof must contain the prepared map's `source_index_sha256`, `query_local_identity` matching the actual local backend, and `evidence` describing the verified extraction configuration. This command does not run training, load GT1, duplicate local map data, or overwrite an existing bound global database. It requires all reference paths to exist before inference. Outputs are FP32 descriptor storage regardless of network backend; descriptor database quantization is a separate later experiment.

Still pending: final backend weights, verified local descriptor compatibility and source calibration for the live camera. These cannot honestly be declared complete before integration. The infrastructure and local payload are ready, not a validated final on-board system.
