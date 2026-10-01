# SuperPoint reference implementation

Trạng thái hiện tại: [EXPERIMENT_STATUS](../docs/EXPERIMENT_STATUS.md). Gate đo phân rã trước khi quyết định train/KD/QAT; MixVPR và candidate được giữ nguyên. Một đường chạy chuẩn: [WORKFLOW](../docs/WORKFLOW.md). Một đường học: [CODE_READING_GUIDE](../docs/CODE_READING_GUIDE.md).

```bash
cd /home/quyet/k210_lab
./superpoint/run status
./superpoint/run check
./superpoint/run measure --help
```

Pilot hiện tại so sánh FR12 `[12,24,32,64]` và wider `[24,32,64,96]`, ở hai logical resolutions W144×H256 và W184×H320. Tất cả dùng detector65, descriptor256, stride8, head hidden128. R2 deployment dùng CW90 tương đương để tránh CPU fallback đã thấy ở graph portrait.

`src/spk210/model.py` định nghĩa graph. `data.py` tạo cache. `training.py` train một candidate. `metrics.py` giữ một implementation cho cả FP32 và simulator quality. `onnx_export.py` phụ trách graph/parity/calibration. `nncase_backend.py` chỉ chạy trong môi trường nncase, không import PyTorch.

Protocol pilot đã khóa tại `configs/pilot400.protocol.json`: 400updates/candidate, same initialization strategy/source plan/canonical correspondences. Đây là một quality screen, chưa phải final full training hay bằng chứng paper đa seed. Không sửa protocol cũ để chạy một thí nghiệm khác; tạo run/config mới và giữ identity rõ ràng.

Cache/checkpoints/DEV report: `artifacts/quality_pilot/`. Deployment output: `artifacts/deployment/<candidate>/`. Log: `results/`. Folder thí nghiệm cũ chỉ để truy nguồn; không dùng các file `*_v2.py` lịch sử làm entrypoint chính.
