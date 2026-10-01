# Đường đọc code: SuperPoint làm reference project Edge AI

Đọc theo thứ tự dưới đây. Mỗi lần hãy tự vẽ shape trước khi chạy command. Không cần đọc toàn bộ thí nghiệm cũ để hiểu pipeline chuẩn.

| Thứ tự / file | Input → output | Function/class cần hiểu | Đọc tiếp |
|---|---|---|---|
| 1. `superpoint/README.md`, `configs/pilot400.protocol.json` | bài toán, data split, budget → hợp đồng thí nghiệm | metric, selection, giới hạn suy luận | settings |
| 2. `superpoint/src/spk210/settings.py` | workspace + shortlist → paths, widths, logical resolutions | `WIDTHS`, `RES`, `parse_candidate` | model |
| 3. `superpoint/src/spk210/model.py` | `[N,1,H,W]` → hai dense raw heads | `Model.__init__`, `Model.forward` | data |
| 4. `superpoint/src/spk210/data.py` | ảnh gốc + teacher + manifest → pair/mask/points/logit cache | `prepare`, `match_correspondences` | protocol/training |
| 5. `superpoint/src/spk210/training.py` | 2 image pairs/update + disk teacher logits → best/latest checkpoint | `train`, KL, symmetric InfoNCE, accumulation, selection | metrics |
| 6. `superpoint/src/spk210/metrics.py` | dense logits/descriptor + masks → keypoints/matches/DEV metrics | `features`, `project`, `pair_metrics`, `evaluate`, `key` | ONNX |
| 7. `superpoint/src/spk210/onnx_export.py` | trained state_dict → static graph + parity + calibration | `rotate_model`, `physical_input`, `logical_outputs`, `export`, `validate`, `calibrate` | nncase |
| 8. `superpoint/src/spk210/nncase_backend.py` | graph + TRAIN tensors → kmodel → actual runtime outputs | `compile_model`, `simulate`, identity assertions | evaluation |
| 9. `superpoint/src/spk210/int8_evaluation.py` và `scripts/benchmark.py` | actual outputs → quality retention / laptop timing | dùng lại cùng evaluator; warmup; median/P90 | deployment |
| 10. `superpoint/deployment/k210/README.md`, `online/README.md` | thưa keypoint/descriptor → matching → 2D–3D → PnP | model identity, camera transform, giới hạn active points, CPU/KPU boundary | map/board gate |

## 1–3: Shape và dataflow — phải hiểu

Backbone gồm hai Conv3×3+ReLU mỗi stage, MaxPool2 sau stage 1/2/3. Với FR12 widths `[12,24,32,64]`, R1 batch 2:

```text
[2,1,256,144]
 → stage1 [2,12,256,144] → pool [2,12,128,72]
 → stage2 [2,24,128,72]  → pool [2,24,64,36]
 → stage3 [2,32,64,36]   → pool [2,32,32,18]
 → stage4 [2,64,32,18]
 → detector hidden128 → logits [2,65,32,18]
 → descriptor hidden128 → raw descriptors [2,256,32,18]
```

Wider chỉ đổi widths thành `[24,32,64,96]`; hai head vẫn như trên. 65 detector channels là 64 vị trí trong cell 8×8 cộng dustbin, không phải 65 loại vật thể. Descriptor 256 channels là embedding mỗi cell, chưa phải một vector global như MixVPR.

Tự kiểm tra: R2 logical `[1,1,320,184]` cho raw heads `[1,65,40,23]` và `[1,256,40,23]`. Graph deployment xoay tương đương dùng `[1,1,184,320]`, nên output spatial thành `[23,40]`.

## 4–5: Dữ liệu và loss — phải hiểu

`pair` có hai view của cùng ảnh; `points` có shape `[2,K,2]` theo `(x,y)`. Homography cho correspondence thật trong thí nghiệm tổng hợp. `cell` mask bỏ padding khi tính KL; mask pixel bỏ vùng không hợp lệ lúc đánh giá.

Detector học phân phối teacher bằng KL. Descriptor được sample tại correspondence, L2, rồi tạo ma trận similarity `[K,K]`. Đường chéo là positive; các cột khác là negative. Symmetric InfoNCE buộc hai chiều nhất quán. Hãy giải thích được vì sao đổi thứ tự point list giữa hai view làm target diagonal sai.

Accumulation chia loss cho số microbatch để effective update tương ứng trung bình. Gradient clipping, optimizer state và scheduler phải được resume cùng checkpoint. Teacher cache được đọc từ mmap; điều này giảm RAM ứng dụng nhưng không chứng minh không có OS page cache.

Initialization L1 transfer dùng helper cũ đã kiểm chứng trong `scripts/create_superpoint_k210_w0.py::transfer_conv`. Nó chọn output filter sau khi giới hạn input channels từ stage trước; phải giữ nguyên 65 class/dustbin và 256 descriptor coordinates. Helper này còn là dependency lịch sử nên được giữ, không nhân bản một bản mới.

## 6: Postprocessing và metric — phải hiểu

Softmax trên 65 channels → bỏ dustbin → decode cell8×8 thành heatmap `[N,H,W]` → mask → NMS → threshold → top160. Descriptor dense được L2, nội suy tại keypoint bằng convention original SuperPoint, rồi L2 output. Kết quả `(keypoints[K,2], descriptors[K,256])`.

MNN có thể nhiều nhưng sai. Precision có thể cao khi chỉ còn ít điểm. Đọc đồng thời correct count, precision, repeatability và spatial coverage. Metric ở đây là homography DEV; không đổi tên thành localization success.

## 7–8: ONNX, CPU/KPU và INT8 — phải hiểu

ONNX chỉ chứa Conv/ReLU/MaxPool. Softmax/NMS/top-k/interpolation/L2 nằm ngoài graph. Native graph được shape inference/checker; không thêm algebraic canonicalization nếu không cần.

CW90 cần xoay cả ảnh, kernel và permutation pixel channels của detector. Chỉ xoay ảnh là đổi bài toán. `validate` so logical PyTorch với physical ONNX đã khôi phục, bao gồm cả phép biến đổi này.

Đọc cả `direct_FP32_raw_strict` trong parity report. Raw descriptor chưa L2 có thể chịu sai khác accumulation giữa kernel FP32. Pipeline giữ lỗi đó, kiểm tra thêm cùng FP32 weights bằng float64 arithmetic với ngưỡng không đổi, rồi kiểm tra actual postprocessing/DEV behavior. Không chỉ đổi tolerance đến khi PASS, và không gọi numerical parity là bitwise identity.

nncase hiện dùng uint8 cho weights/activations, thường gọi chung INT8/8-bit PTQ. External I/O đã kiểm tra là FP32 wrapper. Không đoán scale/zero-point nếu runtime khác. Calibration chỉ lấy TRAIN, không dùng DEV/test để điều chỉnh range.

Đọc compiler `runtime_ops.txt`: bao nhiêu Conv chạy CPU, bao nhiêu KPU? Graph compile được nhưng CPU fallback có thể khiến RAM/latency không chấp nhận được. Compiler planned memory không phải peak SRAM đo trên board.

## 9–10: Benchmark và deployment — phải hiểu

Tách network latency, postprocessing, transfer, matching và PnP. CPU laptop/nncase simulator/K210 board là ba loại bằng chứng khác nhau. Muốn pose đúng, query và reference local descriptors phải cùng model/preprocessing/normalization identity; đổi model phải rebuild map descriptor tương ứng.

Nếu INT8 mất quality, đọc thêm `src/spk210/quantization_diagnosis.py`: ghép detector FP32 với descriptor INT8 và ngược lại để cô lập nhánh gây mất match. Hybrid chỉ là phép chẩn đoán; không phải mô hình deploy. “Nhánh detector” bao gồm tác động lượng tử hóa backbone truyền vào detector, chưa chứng minh chỉ convolution cuối có lỗi.

Board chưa có transport/firmware được qualification cho candidate mới. Package có kmodel và contract không có nghĩa là đã chạy board.

## Phần có thể để AI hỗ trợ

Parser/launcher, atomic JSON/checkpoint writer, logger, CSV/plot formatting, hash manifest và boilerplate unittest. Vẫn cần đọc lỗi, input/output và điều kiện PASS. Không giao phó mù quáng shape, coordinate convention, split protocol, loss target, quantization mapping hoặc cách diễn giải benchmark.
