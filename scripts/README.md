# Legacy scripts and shared dependencies

Canonical SuperPoint commands nằm ở `../superpoint/run`. Không bắt đầu từ các audit/probe/tiny-model scripts trong thư mục này.

Giữ `create_superpoint_k210_w0.py` và `superpoint_kpu_wrapper.py` vì pipeline chuẩn đang tái sử dụng original-teacher loading và initialization transfer đã kiểm chứng. Các script còn lại có giá trị provenance/toolchain/MixVPR và một số vẫn được experiment import bằng absolute path; chưa di chuyển để tránh phá dependency.
