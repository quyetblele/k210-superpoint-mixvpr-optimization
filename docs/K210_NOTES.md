# K210 engineering notes của workspace này

- Toolchain đã dùng: Python3.8, nncase `1.8.0.20220929`, `_nncase` `1.8.0-55be52f`; pinned wheel hash ở root. PyTorch/ONNX Runtime chạy trong môi trường riêng. Không nâng dependency ngầm.
- Compiler graph SuperPoint: 12 Conv, ReLU, 3 MaxPool. Hậu xử lý sparse chạy CPU; dense raw output gồm65+256 channels.
- R1 portrait H256×W144 đã compiler-map đủ12 Conv lên KPU. R2 portrait H320×W184 từng fallback2 Conv lên CPU; transform CW90 tương đương đưa physical input về H184×W320 và random graph map đủ12 Conv lên KPU. Trained weights vẫn phải export/parity/PTQ lại.
- MAC/parameter count là chi phí lý thuyết; không suy ra latency board bằng một hệ số tự đặt.
- nncase `TOTAL`/`.data` là planned allocator/artifact statistics. Không gọi đó là actual peak SRAM. Frame buffers, runtime, CPU stack, DMA, output buffers, sparse features/map working set còn phải được đo.
- Faithful simulator chạy blob kmodel thật. Fake quantization không được dùng làm evidence thay thế. Không có faithful runtime thì đánh dấu INT8 quality BLOCKED.
- Threshold/NMS/top-k, sampling convention, coordinate mapping và CPU L2 thuộc model contract, không chỉ checkpoint.
- Laptop benchmark hiện chỉ gồm network + local postprocessing với ảnh đã nạp; không gồm camera/decode, matching, PnP hay transfer. Không gọi đó là full-system FPS.
- MixVPR đang frozen. Không mở train hoặc đổi global model trong lúc chuẩn hóa SuperPoint. Local map/query identity phải được giữ khớp khi tích hợp về sau.
