# Frozen D4 vs official MixVPR-512 — Pitts30k-test

Run from `/home/quyet/k210_lab`, using the existing CUDA environment. Dataset, official checkpoint, vendor source and results are symlinked to `/mnt/d/k210_benchmarks/pitts30k_test`. Do not replace D4 artifacts or recalibrate on this benchmark.

```bash
.venv-mixvpr-cuda/bin/python mixvpr/setup_pitts30k.py
.venv-mixvpr-cuda/bin/python mixvpr/run_pitts30k_benchmark.py
```

The setup command resumes downloads, verifies original server MD5s, extracts only the ordered official TEST images, and records SHA256. The runner validates the evaluator on synthetic examples and frozen DEV output; locks inputs/source/model hashes; runs official FP32, D4 FP32, and two faithful nncase worker shards; then scores all queries. It can resume completed descriptor blocks without changing the protocol. Do not edit locked evaluator code mid-run or use results for model/preprocessing selection.

Manual stages, if needed to resume an interrupted process:

```bash
.venv-mixvpr-cuda/bin/python mixvpr/pitts30k_benchmark.py selftest
.venv-mixvpr-cuda/bin/python mixvpr/pitts30k_benchmark.py lock
.venv-mixvpr-cuda/bin/python mixvpr/pitts30k_benchmark.py infer --model official
.venv-mixvpr-cuda/bin/python mixvpr/pitts30k_benchmark.py infer --model D4
.venv-mixvpr-cuda/bin/python mixvpr/pitts30k_benchmark.py infer --model INT8 --shard 0 --shards 2
.venv-mixvpr-cuda/bin/python mixvpr/pitts30k_benchmark.py infer --model INT8 --shard 1 --shards 2
.venv-mixvpr-cuda/bin/python mixvpr/pitts30k_benchmark.py score
```

The nncase worker uses `/home/quyet/miniconda3/envs/k210/bin/python`, nncase1.8.0.20220929/runtime1.8.0-55be52f, and the already frozen kmodel. Preprocessed FP32 input travels over a binary pipe; normalized512D output returns to the evaluator. No multi-GB preprocessed image cache is created. Selftest confirms bitwise equality with an existing faithful DEV artifact.

Protocol: original10000 database images,6816 queries; UTM distance≤25m positives; exact CPU FAISS IndexFlatL2 top10; standard full-query denominator. Every model uses its own descriptors for both reference and query. No reranking, query expansion or test calibration.

Official baseline: released ResNet50 cropped layer4 + MixVPR, H400/D4, channel projection256, row projection2,512D, RGB320 bilinear/ImageNet normalization. D4: frozen RGB240 bicubic/ImageNet normalization,512D CPU L2. Different training data and input sizes are disclosed; this is a quality/cost comparison, not a controlled architecture-training ablation. Params count all inference parameters; MACs count Conv+Linear only at batch1 and each model's actual input. Compiler TOTAL is not measured board peak RAM; official model has no claimed K210 executable. Outdoor benchmark results do not establish indoor application acceptance.

Primary sources:

- [Official MixVPR code and released512D checkpoint](https://github.com/amaralibey/MixVPR/tree/4043915cef24818003ece1a8112bc8a24e69abe0): README reports Pitts30k-test90.7/95.5/96.3 for512D. These published values are context, never substituted for this run's metrics.
- [Official dataset loader](https://github.com/amaralibey/MixVPR/blob/4043915cef24818003ece1a8112bc8a24e69abe0/dataloaders/PittsburgDataset.py) and [recall evaluator](https://github.com/amaralibey/MixVPR/blob/4043915cef24818003ece1a8112bc8a24e69abe0/utils/validation.py).
- [NetVLAD dataset specifications](https://www.di.ens.fr/willow/research/netvlad/) and [original Pittsburgh archive server/checksums](https://data.ciirc.cvut.cz/public/projects/2015netVLAD/Pittsburgh250k/).

Evidence lives in `mixvpr/artifacts/pitts30k_test/`: selftest.json, protocol.json, per-backend shard metadata/block hashes, *_predictions.npz, results.json and process logs. No final result exists until all three backends finish; D4 remains unchanged throughout.
