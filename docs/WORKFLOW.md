# Workflow SuperPoint → K210

**Trạng thái nghiên cứu hiện tại:** [EXPERIMENT_STATUS](EXPERIMENT_STATUS.md). Gate đo phân rã chạy bằng `./superpoint/run measure --output superpoint/artifacts/measurement_gate/NEW_RUN_NAME`. Giữ nguyên model, calibration, NMS và MixVPR; chỉ đề xuất một thí nghiệm tiếp theo. Bảng dưới là pipeline engineering/pilot đã có, không phải chỉ dẫn tự động train tiếp. Public benchmark reproduction và untouched TEST chưa hoàn tất; không dùng metric pilot làm số benchmark công khai.

Đường chạy chuẩn nằm ở `superpoint/`. Các script trong `experiments/` và phần lớn `scripts/` là lịch sử nghiên cứu, không phải những lựa chọn thay thế cho các command dưới đây.

Chạy từ `/home/quyet/k210_lab`. `superpoint/run` chỉ chọn script và đúng môi trường Python; thuật toán nằm trong `superpoint/src/spk210/`. Không cần tự activate/switch môi trường giữa PyTorch và nncase.

## Pipeline và trách nhiệm

| Bước | Command chuẩn | Input | Output/artifact | Code chịu trách nhiệm |
|---|---|---|---|---|
| Trạng thái | `./superpoint/run status` | metadata hiện có | trạng thái thực, không train | `scripts/status.py` |
| Kiểm tra | `./superpoint/run check` | frozen protocol, cache, initialization | `artifacts/quality_pilot/refactor_checks.json` | `scripts/check.py` |
| Prepare | `./superpoint/run prepare` | source TRAIN/DEV và original teacher đã có | cache pair, mask, correspondence, teacher logits; SHA256 | `src/spk210/data.py` |
| Train | `./superpoint/run train` | cache đã kiểm tra, cùng initialization/sample plan | 4 nhánh, `best.pt`, `latest.pt`, DEV curves, `summary.json` | `src/spk210/training.py`, `reporting.py` |
| Export | `./superpoint/run export` | checkpoint được chọn bằng DEV | `raw.onnx`, `canonical.onnx`, `export.json` | `src/spk210/onnx_export.py::export` |
| ONNX parity | `./superpoint/run validate` | trained checkpoint + canonical ONNX + 8 TRAIN probes + 42 DEV pairs | `onnx_parity.json`, numerical reference + postprocessing/metric checks; phải PASS trước compile | `onnx_export.py::validate` |
| Calibration | `./superpoint/run calibrate` | 4 TRAIN sources/scene, original + warped views | 64 tensors, `calibration_train.npy`, identity manifest | `onnx_export.py::calibrate` |
| PTQ → compile → gencode | `./superpoint/run compile` | ONNX parity PASS, TRAIN-only calibration | `model.kmodel`, compiler dumps, CPU/KPU mapping, `compile.json` | `src/spk210/nncase_backend.py::compile_model` |
| Faithful simulation | `./superpoint/run simulate` | actual generated kmodel, DEV image pairs | raw simulator outputs + `simulation.json` | `nncase_backend.py::simulate` |
| INT8 quality | `./superpoint/run evaluate` | actual simulator outputs + exact FP32 evaluator | `int8_quality.json`, retention and per-scene outcomes | `src/spk210/int8_evaluation.py` |
| Chẩn đoán INT8 nếu quality giảm | `./superpoint/run diagnose` | FP32 model + actual simulator raw heads | `head_diagnosis.json`, tách ảnh hưởng detector/descriptor | `src/spk210/quantization_diagnosis.py` |
| Laptop profiling | `./superpoint/run benchmark` | four trained checkpoints, preloaded images | `laptop_timing.json` | `scripts/benchmark.py` |
| Deployment package | `./superpoint/run package` | selected kmodel, identity, parity/quality reports | deployment package manifest; **không flash board** | `scripts/package_deployment.py` |
| Board qualification | `./superpoint/run board` | qualified firmware/transport + board | hiện BLOCKED, ghi rõ thiếu gì | `scripts/board.py` |

Các command export/validate/calibrate/compile/simulate/evaluate/package mặc định lấy candidate trong `summary.json`. Có thể truyền cùng `--candidate wider_r1` để kiểm tra một ứng viên khác; không đổi identity ngầm giữa các bước. Train có `--candidate` để resume một nhánh, nhưng luồng chuẩn là chạy tuần tự cả bốn.

## Dữ liệu và training đang khóa

`superpoint/configs/pilot400.protocol.json` là bản sao nguyên vẹn protocol đã khóa trước run v2. Source manifest gồm 112 TRAIN + 28 DEV từ 7-Scenes, thêm 28 TRAIN + 14 DEV từ project: tổng 140/42 ảnh. Project TRAIN/DEV thuộc các phiên chụp có liên quan; đây không phải independent test. Không dùng GT1 để chọn mô hình.

Gray `/255`, giữ tỉ lệ ảnh, letterbox chung 288×512. R1: logical W144×H256. R2: phần ảnh W180×H320, thêm 2 pixel ở mỗi bên thành W184×H320. Padding/biên và phần không nhìn thấy ở cả hai view được mask. Tất cả sai số hình học quy về hệ W144×H256. R2 deploy bằng CW90 tương đương, không train ảnh bị xoay khác nội dung.

Mỗi source có một cặp homography cố định mới. Hai độ phân giải dùng cùng correspondence theo thứ tự sau khi lấy giao các điểm hợp lệ. Đây là pilot tổng hợp hình học, chưa chứng minh robustness với viewpoint 3D, motion blur thực hoặc localization trên map.

Training: detector KL từ original SuperPoint + correspondence InfoNCE, 400 updates/candidate, physical batch 2 images, accumulation 2 pairs, AdamW và cosine schedule. CPU 2 threads, workers 0, không dùng VRAM; vì chạy CPU FP32 nên không bật AMP. Teacher logits là cache FP16 trên đĩa, được đưa về FP32 khi tính loss. Teacher không resident trong student process.

DEV selection: equal-scene macro số MNN đúng → precision → repeatability; giữ checkpoint sớm hơn khi hòa. Báo thêm per-domain, per-scene, coverage, loss và learning curve. Không gọi một pilot là train đến hội tụ, cũng không suy ra kiến trúc tối ưu tuyệt đối từ một seed.

## Artifact và tính tái lập

- Dữ liệu/cache, checkpoint và bảng so sánh: `superpoint/artifacts/quality_pilot/`.
- Artifact deployment riêng từng candidate: `superpoint/artifacts/deployment/<candidate>/`.
- Log vận hành/refactor: `superpoint/results/`.
- Nguồn dataset, teacher, MixVPR và map giữ nguyên tại vị trí đã audit; không nhân bản dataset vào thư mục mới.
- Checkpoint gắn SHA protocol; export gắn SHA checkpoint; calibration, ONNX và kmodel đều có SHA. Simulator outputs gắn SHA input và kmodel.
- Refactor có manifest riêng và kiểm tra output/gradient/postprocessing chính xác. Không sửa protocol cũ để giấu thay đổi implementation.
- `latest.pt` cho phép resume, `best.pt` phục vụ đánh giá. Command train bỏ qua nhánh đã COMPLETE; không âm thầm mở một full train mới.

## Gate tiếp theo, không đi tắt

FP32 pilot → trained-weight ONNX parity → actual PTQ/simulator quality → model-compatible local map → real-map matching/PnP → qualified board runtime → đo latency/RAM thực. Compile PASS không chứng minh chất lượng, và simulator không thay thế benchmark board.

Nếu INT8 mất nhiều correspondence, dừng trước khi quảng bá candidate hoặc rebuild production map. Dùng `diagnose` để tách ảnh hưởng hai head; hybrid tensors chỉ là chẩn đoán, không được báo như kết quả deploy INT8. Sau đó mới thiết kế một thí nghiệm training/quantization có đối chứng nhằm vào nguyên nhân đã đo.

### Detector-only recovery A/B

`./superpoint/run recover-detector` chạy hai nhánh400updates từ cùng wider/R2 pilot checkpoint: control KL hiện tại và KL cân bằng foreground theo teacher. Chỉ convPa/convPb train; backbone và descriptor giữ nguyên từng tensor. Data/cache, sample plan, optimizer và FP32 DEV selection giống nhau. Đây chưa phải QAT.

`./superpoint/run evaluate-recovery` đưa best checkpoint từng nhánh qua export→validate→TRAIN calibration→nncase compile→simulator→quality evaluator chuẩn. Artifact ở `superpoint/artifacts/detector_recovery/`; deployment ở từng candidate riêng. Registry `configs/candidates.json` gắn candidate với checkpoint và training protocol, không ghi đè pilot.

Gate được khóa trước run: treatment có INT8 correct-match count≥control×1.10, FP32 count≥control×0.95 và INT8 precision≥control−0.02. Đây là signal gate của một pilot, không phải ngưỡng release. Đọc `src/spk210/detector_recovery.py::detector_loss` để hiểu trọng số foreground `[N,H/8,W/8]` áp lên KL của logits `[N,65,H/8,W/8]`.

Khi cần giải thích detector instability, chạy `./superpoint/run diagnose-peaks --candidate CANDIDATE`. Input là checkpoint + actual simulator raw outputs đã có; output `peak_stability.json` đo local-peak margin, winner survival, spatial agreement và repeated scores. Không đổi threshold hoặc train. Local-peak audit dùng valid-content mask từng view, không thay thế paired homography evaluator.

QAT chưa có trong canonical implementation. Chỉ xem xét khi bằng chứng trained-weight PTQ xác nhận bottleneck lượng tử hóa, với một thí nghiệm riêng có đối chứng. Không thêm QAT/pruning/training mới chỉ để làm sơ đồ dài hơn.

### Ghi chú parity đã gặp với trained wider/R2

So raw descriptor giữa hai implementation FP32 vượt ngưỡng ở một số phần tử do accumulation rounding. Không đổi weights hoặc nới `atol=rtol=2e-4`. Report giữ kết quả raw strict FAIL, rồi kiểm tra ONNX FP32 với cùng weights tính bằng float64 và round output về FP32, xác nhận rotation bằng float64, kiểm tra keypoint/normalized descriptor trên TRAIN và yêu cầu toàn bộ metric42DEV giống hệt original FP32. Candidate hiện tại qua các kiểm tra này. Đây là numerical qualification có ghi nhận, không phải tuyên bố raw outputs bitwise equal.

## Reuse cho model khác

### Cập nhật chẩn đoán NMS INT8

**Literature audit mới nhất:** đọc `docs/SUPERPOINT_EVALUATION_LITERATURE_AUDIT.md`. Protocol hiện tại là engineering tùy chỉnh, chưa tái hiện baseline public. Tạm dừng suy luận chất lượng intrinsic/training mới; gate kế tiếp là upstream evaluator + coordinate/sampling contract reproduction. Tách network TRAIN/DEV/TEST khỏi map reference/query, không coi resampling old tracks là feature-specific triangulation, và không so teacher high-resolution với student low-resolution như equal-budget. Các kết quả cũ được giữ nguyên với phạm vi chẩn đoán.

**Đính chính phép thử0/41:** TRAIN-only reference chỉ phủ median13.74%query tracks. Sanity chia DEV xen kẽ21support/20query (không self-match, không independent test) cho teacher368x640 đạt20/20; teacher184x320 đạt9/20 với mean descriptor và16/20 khi match observation theo từng ảnh. StudentFP32/INT8 cùng greedy NMS vẫn0/20. Đọc `superpoint/artifacts/map_nms_gate/REFERENCE_FIX_REPORT.md` trước các kết luận cũ. Chốt nguyên tắc kiểm tra độ phủ reference và giữ liên kết image–observation–Point3D; không dùng0/41cũ để kết luận reconstruction sai. Gate tiếp theo là tách detector/descriptor student với teacher cùng độ phân giải trên protocol đã kiểm tra coverage; chưa train dài hay promote production.

Chẩn đoán0/41đã được mở rộng: known-correspondence COLMAP→PnP đạt41/41, reprojection median1.33px; chưa có bằng chứng cần dựng lại hình học. Exact-observation descriptor map tăng lên5981points nhưng student vẫn0/41; đổi strict matcher→MNN cũng không cứu pose. Teacher gốc ở184x320 vẫn0/41; ở368x640 đạt2/41normal,4/41oracle. Xem `superpoint/artifacts/map_nms_gate/ROOT_CAUSE_REPORT.md`. Gate kế tiếp là known-track natural-view descriptor/localization audit; chưa train dài/QAT hay promote. Oracle và known-correspondence chỉ là chẩn đoán, không phải performance deploy.

Đã hoàn tất real-map diagnostic148TRAIN/41DEV cho6điều kiện FP32/INT8 ×3NMS: tất cả0/41pose. Không quy lỗi này cho INT8 vì FP32 cũng thất bại. Audit cho thấy median track overlap của query với retrieved/expanded reference bằng0, kèm match sai hình học. Xem `superpoint/artifacts/map_nms_gate/REPORT.md`. Gate tiếp theo: geometry-assisted reference diagnostic trên DEV để tách retrieval/coverage khỏi local matching; phải ghi rõ oracle, không báo như kết quả deploy. Chưa train/QAT hoặc thay pipeline mặc định.

Đã kiểm tra greedy NMS trực tiếp trên output kmodel wider/R2, không train. Correct MNN trên DEV tăng27.52→62.60 (score-only) hoặc64.19 (corner chỉ phá exact ties); bản gốc FP32 đạt65.49. Audit84views xác nhận NMS gốc để89.99% điểm có hàng xóm trong suppression radius, greedy giảm về0. Tuy nhiên gate precision đã khóa vẫn FAIL; chưa promote hay sửa default. Xem `superpoint/artifacts/nms_diagnosis/wider_r2_cw/REPORT.md` và protocol/results cùng thư mục. Gate tiếp theo ưu tiên matching/PnP DEV với map/query cùng postprocessing identity trước khi quyết định QAT; không tự động train lại.

Reuse thứ tự gate, quản lý identity, data split, parity, TRAIN-only calibration, kiểm tra CPU/KPU mapping và cách đo tài nguyên. Thay phần model, loss, preprocessing/postprocessing và metric đúng bài toán. Với MixVPR, descriptor global và retrieval R@K thay cho SuperPoint detector/descriptor local và matching; không copy nguyên loss hay decoder65 channels. Chỉ tổng quát hóa code khi đã có hai implementation thực sự cần dùng chung, không dựng framework trước.
