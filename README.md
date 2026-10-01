# K210 Edge AI workspace

Đang làm **SuperPoint → trained FP32 → ONNX → 8-bit PTQ → K210**, với SuperPoint là reference implementation đầu tiên. Mục tiêu là chất lượng hữu ích trong giới hạn phần cứng, có bằng chứng ở từng gate.

- [Workflow và command chuẩn](docs/WORKFLOW.md)
- [Đường đọc code để học từ đầu đến cuối](docs/CODE_READING_GUIDE.md)
- [K210/toolchain/CPU–KPU notes](docs/K210_NOTES.md)
- [Audit và thay đổi workspace](docs/WORKSPACE_AUDIT.md)
- [SuperPoint](superpoint/README.md)

Bắt đầu tại đây:

```bash
cd /home/quyet/k210_lab
./superpoint/run status
./superpoint/run check
```

Ưu tiên hiện tại: đọc [trạng thái thực nghiệm](docs/EXPERIMENT_STATUS.md). Gate đo phân rã giữ nguyên student/MixVPR, chưa cho phép mở training mới. `check` PASS là kiểm tra implementation, không phải điều kiện đủ để train tiếp hoặc promote model.

Tái chạy gate vào thư mục mới: `./superpoint/run measure --output superpoint/artifacts/measurement_gate/NEW_RUN_NAME`. Các command pilot phía dưới được giữ để tái lập lịch sử; không dùng chúng thay cho benchmark công khai hoặc TEST độc lập.

```text
k210_lab/
├── README.md
├── docs/                     # workflow, code guide, audit, K210 notes
├── superpoint/
│   ├── run                   # launcher nhỏ, chọn đúng Python environment
│   ├── configs/              # frozen pilot protocol
│   ├── src/spk210/           # implementation chuẩn theo trách nhiệm
│   ├── scripts/              # một entrypoint cho mỗi bước
│   ├── deployment/k210/      # I/O contract và board gate
│   ├── artifacts/            # cache, checkpoints, ONNX, kmodel, evidence
│   └── results/              # log vận hành và kiểm tra
├── mixvpr/                   # frozen model pointer; chưa refactor/train lại
├── online/                   # real-map/localization PC integration đang giữ
├── datasets/                 # dữ liệu giữ nguyên
├── tools/                    # audit/environment utilities
├── archive/                  # tài liệu lịch sử đã chuyển
├── experiments/              # lịch sử nghiên cứu, archive tại chỗ
├── scripts/                  # legacy scripts và shared helper còn phụ thuộc
├── models/, reports/         # model/benchmark lịch sử giữ nguyên
└── .venv-mixvpr-cuda/         # môi trường đã có, không đưa vào Git
```

Output chuẩn: `superpoint/artifacts/quality_pilot/` và `superpoint/artifacts/deployment/<candidate>/`. Checkpoint ở từng candidate, không copy thêm sang một thư mục khác gây lệch identity.

MixVPR sẽ reuse workflow và cách quản lý identity/evidence sau khi SuperPoint hoàn thành; checkpoint frozen hiện tại vẫn được giữ. Không coi refactor workspace là lý do train lại MixVPR.

**Trạng thái board:** package/compile/simulator không đồng nghĩa đã chạy K210. Latency và peak SRAM thực phải đo trên board; xem `superpoint/deployment/k210/README.md`.

Vòng tiếp theo đã hoàn thành: [Detector recovery A/B và chẩn đoán INT8](superpoint/artifacts/detector_recovery/REPORT.md). Cả training recipe thử nghiệm và exact-score tie remedy đều không qua gate; chưa promotion model mới.
