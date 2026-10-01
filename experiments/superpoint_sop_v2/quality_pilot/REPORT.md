# SuperPoint: matched four-configuration quality pilot

Completed400updates per configuration. Selected for the next engineering gate: **wider_r2_cw**. This is a pilot selection, not a final K210 release model.

All variants use the same140TRAIN/42DEV source images, seeded sample plan, deterministic teacher-based initialization strategy, optimizer budget and supervision. Same-width variants start from identical checkpoint hashes. DEV metrics use a common144x256 coordinate frame,3px geometry tolerance, top160 points and valid mutual overlap. Selection: equal-scene macro correct matches, then precision, then repeatability. GT1 and MixVPR were not used.

| Candidate | Best update | Correct matches | Precision | Repeatability | Coverage | Correct/teacher |
|---|---:|---:|---:|---:|---:|---:|
|fr12_r1|400|28.94|42.17%|48.99%|49.08%|51.04%|
|wider_r1|400|55.72|63.74%|56.08%|68.97%|98.28%|
|fr12_r2_cw|400|38.88|50.11%|53.21%|58.04%|52.95%|
|wider_r2_cw|400|66.07|67.12%|59.11%|72.07%|89.98%|

## Domain breakdown

| Candidate | Domain | Correct matches | Precision | Repeatability |
|---|---|---:|---:|---:|
|fr12_r1|indoor|29.71|43.36%|48.89%|
|fr12_r1|project|23.50|33.90%|49.75%|
|wider_r1|indoor|56.57|63.47%|56.53%|
|wider_r1|project|49.79|65.62%|52.97%|
|fr12_r2_cw|indoor|39.43|50.65%|52.87%|
|fr12_r2_cw|project|35.00|46.37%|55.60%|
|wider_r2_cw|indoor|67.54|68.25%|60.33%|
|wider_r2_cw|project|55.79|59.17%|50.58%|

Teacher anchors are in `r1/teacher_dev.json` and `r2/teacher_dev.json`; full per-scene/per-image outcomes, loss trends and DEV curves are in each candidate `report.json`.

![Learning curves](learning_curves.png)

## Limits and next gate

- One seed and400updates, not training to convergence
- Fixed synthetic homography pairs, no real-viewpoint or independent localization evaluation
- TRAIN/DEV project images from related capture sequences
- Teacher detector supervision is common recipe, this does not isolate KD benefit
- Equal update budget, not equal compute; wider/higher-resolution candidates cost more
- Random-weight compile feasibility is not trained-weight INT8 quality

trained-weight ONNX parity -> TRAIN-only PTQ -> real nncase simulator descriptor/keypoint/matching quality for quality/cost Pareto candidates; no full train automatically.

Training usedCPU2threads, workers0, physical batch2images and2-pair gradient accumulation; FP32 CPU means no training VRAM or GPU teacher residency. Atomic best/latest checkpoints andRAM/disk guards allow safe resumption. Process RSS is laptop process memory, not K210 SRAM.

Maximum reported process RSS: 914.9MiB.
