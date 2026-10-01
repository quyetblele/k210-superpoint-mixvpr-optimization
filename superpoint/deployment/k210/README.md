# Deployment contract và board gate

`./superpoint/run package` tạo package của trained kmodel và manifest identity. Đây là bước chuẩn bị deploy, không flash firmware.

Physical input FP32 wrapper là grayscale `/255`, static batch1. R1 `[1,1,256,144]`. R2 `[1,1,184,320]` sau CW90. Graph xuất raw detector65 và descriptor256; runtime I/O dtype phải được xác nhận. CPU khôi phục orientation/permutation R2 rồi softmax, pixel decode, valid mask, NMS, threshold, top160, sample/L2 descriptor theo contract.

Để chạy board cần xác định board/firmware, adapter KPU runtime hỗ trợ đúng kmodel, format/scale I/O, transport/camera, bộ nhớ output và CPU postprocessing. Hiện chưa có adapter board đã qualification cho candidate mới trong canonical pipeline. `./superpoint/run board` trả BLOCKED có lý do; không phát sinh số latency/RAM giả.

Khi có adapter: xác nhận raw-output parity trên cùng ảnh → end-to-end sparse feature parity → timing từng stage → peak RAM thực gồm firmware/frame/DMA/output/working set → matching/PnP với map identity tương thích. Không dùng thời gian nncase Simulator trên laptop thay cho latency K210.
