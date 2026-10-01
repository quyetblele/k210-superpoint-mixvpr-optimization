# Workspace audit và refactor

## Phạm vi đã kiểm tra

Inventory trước refactor gồm **94.949 file**, metadata toàn bộ cây K210 ngoài `.git` và virtualenv; source index đọc/parse **125 file Python** để ghi module docstring, imports, function/class và SHA256. Nội dung dataset/binary không được nạp vào RAM để “đọc toàn bộ”; thay vào đó bảo toàn file, manifest và vai trò. Code của các bước SuperPoint được tái sử dụng đã được đọc trực tiếp và kiểm tra hành vi.

Evidence: `audit/inventory.json.gz`, `audit/source_index.json`, `audit/refactor_manifest.json`, `audit/cleanup.json`, `audit/environment.json`. Inventory lớn được nén; không copy dataset.

Audit lại bằng `python3 tools/audit_workspace.py`; output `current_*` riêng, không đè inventory trước refactor. Kiểm tra environment bằng `python3 tools/doctor.py`; cả hai command không cài package và không xóa dữ liệu.

## KEEP / ARCHIVE / DELETE

| Nhóm | Quyết định | Lý do |
|---|---|---|
| datasets, archive dataset gốc | KEEP | tài sản gốc, không xóa; dung lượng logic khoảng42,4GiB trước audit |
| models/checkpoints, calibration, benchmark reports | KEEP | provenance/đối chứng/hardware evidence; không xóa model trung gian chưa rõ giá trị |
| `online/` và map | KEEP | active real-map integration; không đổi behavior trong refactor này |
| original teacher helper/wrapper trong `scripts/` | KEEP | dependency được code chuẩn và thí nghiệm cũ dùng |
| SuperPoint implementation mới | KEEP, canonical | `superpoint/src/spk210/`, một entrypoint chuẩn mỗi stage |
| thí nghiệm cũ và các probe MixVPR | ARCHIVE tại chỗ | có absolute paths/imports/source hashes; di chuyển cơ học có thể phá tái lập |
| README root cũ | MOVE/ARCHIVE | lưu ở `archive/README_toolchain_smoke_original.md` |
| active v2 artifact folder | MOVE | chuyển sang `superpoint/artifacts/quality_pilot`; giữ compatibility symlink tại đường dẫn cũ |
| Python bytecode cache cũ | DELETE |88 file,645.303bytes, tự tái tạo được; ledger đầy đủ ở cleanup.json |
| file chưa xác định chắc vai trò | KEEP | không suy đoán rồi xóa |

Không xóa dataset, checkpoint quan trọng, source đang dùng hoặc benchmark quan trọng. Không tuyên bố đã giải phóng hàng GB: đợt này ưu tiên đường chạy rõ ràng và bảo toàn bằng chứng. Tổng dung lượng metadata là logical size, có thể khác dung lượng đĩa vật lý vì hardlink/sparse/symlink.

## Những gì thay đổi thực chất

1. Root README chuyển từ tiny-ONNX smoke sang SuperPoint reference workflow.
2. Tách model/data/training/metric/ONNX/nncase/evaluation ra module riêng; không còn dùng giant experiment script làm command chính.
3. Launcher nhỏ chọn PyTorch hoặc nncase environment đúng từng stage, không trộn dependency.
4. Các data/checkpoint/export/simulator identity được kiểm tra, thay vì dựa vào tên `best.pt`.
5. Bổ sung workflow, code reading guide, K210 notes, môi trường và cleanup ledger.
6. `.gitignore` bỏ virtualenv, dataset và generated heavy artifacts khỏi Git; không xóa chúng khỏi đĩa. Source và config vẫn đọc được/review được.
7. Lịch sử experiment được đánh dấu rõ; không tiếp tục nhân bản active implementation thành `new2/final_fixed`.

## Kiểm tra trước khi tiếp tục SuperPoint

`./superpoint/run check` so implementation trước/sau refactor trên cả4cấu hình: raw outputs, gradient và postprocessing phải **exactly equal**;182 cặp có canonical correspondence giống nhau. Frozen protocol và cache hashes được kiểm tra. Kết quả thực ở `superpoint/artifacts/quality_pilot/refactor_checks.json`.

Pilot v1 có một khác biệt do border rasterization làm vài correspondence khác nhau giữa resolutions; được giữ và đánh dấu SUPERSEDED, không dùng để xếp hạng. Pilot v2 dùng giao correspondence, rerun từ cùng original initialization, không warm-start từ kết quả v1.

MixVPR tiếp tục frozen; không train lại, không đổi kiến trúc hay teacher. Board execution cho candidate mới là gate riêng chưa được chứng minh; canonical command báo BLOCKED thay vì bịa kết quả.
