> Current continuation: canonical workspace and measured pilot/deployment results are in `/home/quyet/k210_lab/superpoint/results/REPORT.md`. The report below preserves the earlier SOP round.

# SuperPoint × K210 — áp dụng SOP v2, vòng kỹ thuật đầu

## Quyết định hiện tại

Chưa khóa final student. Đã hoàn tất replication của recipe detector, định vị mất chất lượng INT8 và tạo shortlist cấu hình dựa trên mapping compiler thực tế. Mô hình MixVPR và GT1 không tham gia thay đổi/tuning trong campaign này.

## G0/G3 — Tín hiệu chất lượng và giới hạn

Ba lần lặp dùng cùng checkpoint SUP warm-start, thay thứ tự dữ liệu. Đây không phải ba initialization độc lập. Mỗi nhánh thêm 400 updates, cùng kiến trúc, data, geometry loss, optimizer/budget; nhánh thử thay detector CE bằng teacher KL, không thêm descriptor KD.

| Data-order seed | SUP continuation | Teacher detector | Chênh lệch |
| --- | --- | --- | --- |
| 20260914 | 44.93% | 49.65% | +4.72 pp |
| 20260915 | 43.54% | 49.15% | +5.62 pp |
| 20260916 | 45.67% | 49.96% | +4.29 pp |

Mean precision: 44.71% → 49.59%. Có tín hiệu tương đối ở cả ba lần, nhưng gate tuyệt đối 50% chưa đạt; không đổi ngưỡng sau khi xem kết quả. Evaluator chỉ có 28 DEV homography pairs, chưa chứng minh real-map pose success hoặc generalization. Chi tiết: ../superpoint_quality_first/replicates/summary.json.

## G1/G2 — Compiler PASS chưa đủ; đã sửa layout gây CPU fallback

Hai layout 184×320 ban đầu compile nhưng hai convolution đầu chạy CPU. Compiler báo .data lần lượt 6,359,040 và 12,718,080 bytes. Không giữ các layout đó trong shortlist hiện tại.

Đã kiểm phép biến đổi CW90 đồng bộ input và các convolution kernels, đồng thời hoán vị 64 detector pixel channels; dustbin giữ nguyên. Khôi phục output bằng phép biến đổi ngược. Trên 16 TRAIN inputs, sai khác FP32 raw heads tối đa 1.49e-8 cho các graph random được kiểm. Với layout 320×184, cả 12 convolution về KPU, .data còn354,200 bytes. Đây là bằng chứng compiler/FP32 graph, chưa phải phép đo SRAM hoặc latency board.

| Candidate | Input W×H vào graph | Params | MAC | Conv CPU/KPU | Compiler .data bytes | Quyết định |
| --- | --- | --- | --- | --- | --- | --- |
| fr12_r1 | 144×256 | 269,989 | 301.03 M | 0/12 | 221,760 | Giữ để đo quality |
| wider_r1 | 144×256 | 478,345 | 705.80 M | 0/12 | 221,760 | Giữ để đo quality |
| fr12_r2 | 184×320 | 269,989 | 480.81 M | 2/10 | 6,359,040 | Loại layout hiện tại |
| wider_r2 | 184×320 | 478,345 | 1127.32 M | 2/10 | 12,718,080 | Loại layout hiện tại |
| fr12_r2_cw | 320×184 | 269,989 | 480.81 M | 0/12 | 354,200 | Giữ để đo quality |
| wider_r2_cw | 320×184 | 478,345 | 1127.32 M | 0/12 | 354,200 | Giữ để đo quality |

`cw` giữ logical input portrait 184×320 rồi xoay CW90 đồng bộ với weights. Chỉ xoay ảnh mà không xoay kernels/head permutation không có cùng tính tương đương. Cả 6 kmodel trong bảng này dùng random weights: không dùng chúng làm student chất lượng hoặc báo điểm INT8.

## G4 — Nguyên nhân giảm correspondence nằm chủ yếu ở detector

Hybrid diagnostic với checkpoint teacher-detector: detector FP32 + descriptor INT8 giữ 36.79 match đúng so với 36.89 FP32; detector INT8 + descriptor FP32 còn 21.07. Số keypoint không giảm tương ứng; chọn vị trí sau logits/NMS thay đổi. Hybrids dùng để chẩn đoán, không được gọi là deployment model.

Đã chạy bốn cấu hình triển khai bằng actual nncase K210 Simulator, calibration chỉ TRAIN. A/B calibration giữ 56 samples cho cả control và augmented. Hai reparameterization được kiểm FP32 parity trước PTQ: trừ common mean ở detector logits và tách 64 point channels / 1 dustbin. Không có retraining hoặc đổi threshold để tạo các kết quả dưới.

| Variant | MNN precision | Match đúng/cặp | Repeatability@3px |
| --- | --- | --- | --- |
| FP32 | 49.65% | 36.89 | 44.96% |
| INT8 baseline | 49.73% | 21.54 | 42.57% |
| TRAIN augmented calibration | 50.23% | 20.68 | 40.69% |
| Centered logits | 50.45% | 21.43 | 42.73% |
| Split detector | 50.30% | 21.86 | 42.05% |

Các cách sửa rẻ chưa đạt gate 85% correspondence retention. Không chọn variant chỉ vì precision cao hơn. Chưa có bằng chứng đủ để chuyển những pilot này thành final localization student; calibration coverage/mean-centering/head-splitting không giải quyết được vấn đề trong thiết lập đã đo. Không suy từ đó rằng mọi calibration hay QAT đều vô ích.

## Resource và artifacts

Các stage nặng chạy tuần tự, CPU 2 threads, workers 0, teacher cache trên đĩa; không dùng VRAM. Training có guard khi host available memory dưới 1536 MiB, atomic best/latest checkpoints. Giữ các checkpoint/evidence trước đó.

- requirements.json: task/resource contract, ghi rõ application thresholds và board metrics còn thiếu.
- candidate_registry.csv: input/width, mapping CPU/KPU và compiler statistics.
- quality_evidence.csv; decision_evidence.png/pdf: số đo và hình đối chiếu.
- quantization_diagnosis*.json: head-swap và oracle-point diagnostics.
- calibration_ab/, centered_detector/, split_detector/: protocol, parity, compiled artifacts và actual simulator outputs.
- joint_feasibility/, orientation_probe/: graph, mapping, calibration TRAIN và parity/memory evidence.
- gate_status.json; next_gate.json: trạng thái và bước kế tiếp.

## One next gate

G2-QUALITY: pilot chất lượng có đối chứng cho 4 cấu hình shortlist, với cùng data/sampling/initialization strategy/budget; teacher làm anchor ở từng logical resolution. Giữ input portrait khi học/evaluate; chỉ áp dụng orientation transform đã kiểm khi export cần layout KPU. Chọn bằng quality guardrails và resource; không chọn cấu hình rộng nhất hoặc resolution nhỏ nhất theo cảm tính.

Trước một kết luận deployment cuối phải rebuild map descriptors đúng model/preprocessing, kiểm real-map DEV và đo board. Những việc đó còn mở; không gọi kết quả homography hoặc compiler TOTAL là bằng chứng thay thế.
