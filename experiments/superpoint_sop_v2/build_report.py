"""Summarize completed SOP gates without promoting compile success to readiness."""
from pathlib import Path
import json
import re
import csv
import hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent
PREVIOUS = ROOT.parent / 'superpoint_quality_first'

def read(path):
    return json.loads(path.read_text())

def compiler_stats(folder):
    text = (folder / 'compile.log').read_text()
    stats = {}
    for key in ['.input', '.output', '.data', 'MODEL', 'TOTAL']:
        match = re.search(re.escape(key) + r'[^\n]*\((\d+) B\)', text)
        stats[key] = int(match[1]) if match else None
    cpu_file = folder / 'compiler_dump/stackvm/main/runtime_ops.txt'
    cpu = cpu_file.read_text() if cpu_file.exists() else ''
    kpu = '\n'.join(p.read_text() for p in (folder / 'compiler_dump/k210').rglob('runtime_ops.txt'))
    stats.update(CPU_Conv2D=cpu.count('[Conv2D]'), KPU_Conv2D=kpu.count('[KPUConv2D]'))
    return stats

def main():
    replication = read(PREVIOUS / 'replicates/summary.json')
    assert replication['status'] == 'COMPLETE'
    diagnostic = read(ROOT / 'quantization_diagnosis.json')
    calibration = read(ROOT / 'calibration_ab/summary.json')
    centered = read(ROOT / 'centered_detector/summary.json')
    split = read(ROOT / 'split_detector/summary.json')
    orientation = read(ROOT / 'orientation_probe/summary.json')
    assert all(r['gate_pass'] for r in orientation['candidates'])
    base = read(PREVIOUS / 'detector_candidate/int8_quality.json')
    variants = {
        'FP32': base['fp32'], 'INT8 baseline': base['int8'],
        'TRAIN augmented calibration': calibration['results']['original_and_warped']['int8'],
        'Centered logits': centered['centered_int8'], 'Split detector': split['split_int8'],
    }
    sources = [(ROOT / 'joint_feasibility', read(ROOT / 'joint_feasibility/summary.json')['candidates']),
               (ROOT / 'orientation_probe', orientation['candidates'])]
    candidates = []
    for parent, rows in sources:
        for row in rows:
            folder = parent / row['id']
            stats = compiler_stats(folder)
            export = row['export']
            eligible = row['compile'].get('gencode') == 'PASS' and stats['CPU_Conv2D'] == 0 and stats['KPU_Conv2D'] == 12
            candidates.append({'id': row['id'], 'input_WH': export['input_WH'], 'parameters': export['parameters'],
                'MAC': export['conv_MAC'], 'kmodel_bytes': row['compile'].get('kmodel_bytes'),
                'compiler': stats, 'shortlist': eligible,
                'quality': 'NOT_EVALUATED_RANDOM_WEIGHTS', 'board_peak_and_latency': 'NOT_MEASURED'})
    with (ROOT / 'candidate_registry.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['candidate','input_W','input_H','parameters','MAC','kmodel_bytes','CPU_Conv2D','KPU_Conv2D','compiler_data_bytes','compiler_TOTAL_bytes_NOT_actual_peak','shortlist','quality'])
        for r in candidates:
            writer.writerow([r['id'], *r['input_WH'], r['parameters'], r['MAC'], r['kmodel_bytes'], r['compiler']['CPU_Conv2D'], r['compiler']['KPU_Conv2D'], r['compiler']['.data'], r['compiler']['TOTAL'], r['shortlist'], r['quality']])
    with (ROOT / 'quality_evidence.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['variant','MNN_precision','correct_MNN_per_pair','repeatability_3px'])
        for name, r in variants.items():
            writer.writerow([name, r['mnn_precision'], r['correct_mnn_count'], r['repeatability_3px']])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
    ids = np.arange(3)
    axes[0].plot(ids, [100*r['control_best']['mnn_precision'] for r in replication['rows']], 'o-', label='SUP continuation')
    axes[0].plot(ids, [100*r['teacher_detector_best']['mnn_precision'] for r in replication['rows']], 'o-', label='Teacher detector supervision')
    axes[0].axhline(50, color='gray', ls='--', label='Predeclared pilot quality gate')
    axes[0].set(xticks=ids, xticklabels=['20260914','20260915','20260916'], ylabel='DEV MNN precision (%)',
                ylim=(40, 55), title='Same warm start; three data-order seeds')
    axes[0].legend(fontsize=8, loc='lower right')
    axes[0].grid(alpha=.2)
    names = list(variants)
    values = [variants[n]['correct_mnn_count'] for n in names]
    bars = axes[1].bar(np.arange(len(names)), values, color=['#14666c'] + ['#8397aa']*4)
    axes[1].axhline(.85*values[0], color='#c06a36', ls='--', label='85% correspondence-retention gate')
    axes[1].set(xticks=np.arange(len(names)), xticklabels=['FP32','INT8\nbase','TRAIN\naugmented','Centered\nlogits','Split\nhead'],
                ylabel='Mean correct MNN matches per pair', ylim=(0, 43), title='Actual K210 kmodel; existing 28 DEV pairs')
    axes[1].bar_label(bars, fmt='%.1f', padding=3)
    axes[1].legend(fontsize=8, loc='upper right')
    fig.savefig(ROOT / 'decision_evidence.png', dpi=160)
    fig.savefig(ROOT / 'decision_evidence.pdf')
    plt.close(fig)
    seed_lines = []
    for r in replication['rows']:
        seed_lines.append('| %s | %.2f%% | %.2f%% | +%.2f pp |' % (r['seed'],100*r['control_best']['mnn_precision'],100*r['teacher_detector_best']['mnn_precision'],100*r['precision_delta']))
    hardware_lines = []
    for r in candidates:
        hardware_lines.append('| %s | %s | %s | %.2f M | %d/%d | %s | %s |' % (r['id'], '×'.join(map(str,r['input_WH'])), f"{r['parameters']:,}", r['MAC']/1e6, r['compiler']['CPU_Conv2D'],r['compiler']['KPU_Conv2D'],f"{r['compiler']['.data']:,}", 'Giữ để đo quality' if r['shortlist'] else 'Loại layout hiện tại'))
    quant_lines = ['| %s | %.2f%% | %.2f | %.2f%% |' % (name,100*r['mnn_precision'],r['correct_mnn_count'],100*r['repeatability_3px']) for name,r in variants.items()]
    report = '''# SuperPoint × K210 — áp dụng SOP v2, vòng kỹ thuật đầu

## Quyết định hiện tại

Chưa khóa final student. Đã hoàn tất replication của recipe detector, định vị mất chất lượng INT8 và tạo shortlist cấu hình dựa trên mapping compiler thực tế. Mô hình MixVPR và GT1 không tham gia thay đổi/tuning trong campaign này.

## G0/G3 — Tín hiệu chất lượng và giới hạn

Ba lần lặp dùng cùng checkpoint SUP warm-start, thay thứ tự dữ liệu. Đây không phải ba initialization độc lập. Mỗi nhánh thêm400 updates, cùng kiến trúc, data, geometry loss, optimizer/budget; nhánh thử thay detector CE bằng teacher KL, không thêm descriptor KD.

| Data-order seed | SUP continuation | Teacher detector | Chênh lệch |
| --- | --- | --- | --- |
''' + '\n'.join(seed_lines) + '''

Mean precision: %.2f%% → %.2f%%. Có tín hiệu tương đối ở cả ba lần, nhưng gate tuyệt đối50%% chưa đạt; không đổi ngưỡng sau khi xem kết quả. Evaluator chỉ có28 DEV homography pairs, chưa chứng minh real-map pose success hoặc generalization. Chi tiết: ../superpoint_quality_first/replicates/summary.json.

## G1/G2 — Compiler PASS chưa đủ; đã sửa layout gây CPU fallback

Hai layout184×320 ban đầu compile nhưng hai convolution đầu chạy CPU. Compiler báo .data lần lượt6,359,040 và12,718,080 bytes. Không giữ các layout đó trong shortlist hiện tại.

Đã kiểm phép biến đổi CW90 đồng bộ input và các convolution kernels, đồng thời hoán vị64 detector pixel channels; dustbin giữ nguyên. Khôi phục output bằng phép biến đổi ngược. Trên16 TRAIN inputs, sai khác FP32 raw heads tối đa1.49e-8 cho các graph random được kiểm. Với layout320×184, cả12 convolution về KPU, .data còn354,200 bytes. Đây là bằng chứng compiler/FP32 graph, chưa phải phép đo SRAM hoặc latency board.

| Candidate | Input W×H vào graph | Params | MAC | Conv CPU/KPU | Compiler .data bytes | Quyết định |
| --- | --- | --- | --- | --- | --- | --- |
''' % (100*replication['mean_control_precision'],100*replication['mean_teacher_detector_precision']) + '\n'.join(hardware_lines) + '''

`cw` giữ logical input portrait184×320 rồi xoay CW90 đồng bộ với weights. Chỉ xoay ảnh mà không xoay kernels/head permutation không có cùng tính tương đương. Cả6 kmodel trong bảng này dùng random weights: không dùng chúng làm student chất lượng hoặc báo điểm INT8.

## G4 — Nguyên nhân giảm correspondence nằm chủ yếu ở detector

Hybrid diagnostic với checkpoint teacher-detector: detector FP32 + descriptor INT8 giữ36.79 match đúng so với36.89 FP32; detector INT8 + descriptor FP32 còn21.07. Số keypoint không giảm tương ứng; chọn vị trí sau logits/NMS thay đổi. Hybrids dùng để chẩn đoán, không được gọi là deployment model.

Đã chạy bốn cấu hình triển khai bằng actual nncase K210 Simulator, calibration chỉ TRAIN. A/B calibration giữ56 samples cho cả control và augmented. Hai reparameterization được kiểm FP32 parity trước PTQ: trừ common mean ở detector logits và tách64 point channels/1 dustbin. Không có retraining hoặc đổi threshold để tạo các kết quả dưới.

| Variant | MNN precision | Match đúng/cặp | Repeatability@3px |
| --- | --- | --- | --- |
''' + '\n'.join(quant_lines) + '''

Các cách sửa rẻ chưa đạt gate85% correspondence retention. Không chọn variant chỉ vì precision cao hơn. Chưa có bằng chứng đủ để chuyển những pilot này thành final localization student; calibration coverage/mean-centering/head-splitting không giải quyết được vấn đề trong thiết lập đã đo. Không suy từ đó rằng mọi calibration hay QAT đều vô ích.

## Resource và artifacts

Các stage nặng chạy tuần tự, CPU2 threads, workers0, teacher cache trên đĩa; không dùng VRAM. Training có guard khi host available memory dưới1536 MiB, atomic best/latest checkpoints. Giữ các checkpoint/evidence trước đó.

- requirements.json: task/resource contract, ghi rõ application thresholds và board metrics còn thiếu.
- candidate_registry.csv: input/width, mapping CPU/KPU và compiler statistics.
- quality_evidence.csv; decision_evidence.png/pdf: số đo và hình đối chiếu.
- quantization_diagnosis*.json: head-swap và oracle-point diagnostics.
- calibration_ab/, centered_detector/, split_detector/: protocol, parity, compiled artifacts và actual simulator outputs.
- joint_feasibility/, orientation_probe/: graph, mapping, calibration TRAIN và parity/memory evidence.
- gate_status.json; next_gate.json: trạng thái và bước kế tiếp.

## One next gate

G2-QUALITY: pilot chất lượng có đối chứng cho4 cấu hình shortlist, với cùng data/sampling/initialization strategy/budget; teacher làm anchor ở từng logical resolution. Giữ input portrait khi học/evaluate; chỉ áp dụng orientation transform đã kiểm khi export cần layout KPU. Chọn bằng quality guardrails và resource; không chọn cấu hình rộng nhất hoặc resolution nhỏ nhất theo cảm tính.

Trước một kết luận deployment cuối phải rebuild map descriptors đúng model/preprocessing, kiểm real-map DEV và đo board. Những việc đó còn mở; không gọi kết quả homography hoặc compiler TOTAL là bằng chứng thay thế.
'''
    for old, new in {
        'thêm400': 'thêm 400', 'đối50%': 'đối 50%', 'có28': 'có 28',
        'layout184': 'layout 184', 'lượt6,': 'lượt 6,', 'và12,': 'và 12,',
        'vị64': 'vị 64', 'Trên16': 'Trên 16', 'đa1.49': 'đa 1.49',
        'layout320': 'layout 320', 'cả12': 'cả 12', 'portrait184': 'portrait 184',
        'Cả6': 'Cả 6', 'giữ36.': 'giữ 36.', 'với36.': 'với 36.', 'còn21.': 'còn 21.',
        'giữ56': 'giữ 56', 'tách64': 'tách 64', 'channels/1': 'channels / 1',
        'gate85%': 'gate 85%', 'CPU2': 'CPU 2', 'workers0': 'workers 0',
        'dưới1536': 'dưới 1536', 'cho4': 'cho 4',
    }.items():
        report = report.replace(old, new)
    (ROOT / 'REPORT.md').write_text(report)
    status = {'G0': 'PARTIAL: existing homography evaluator reproduced; application acceptance and independent test not established',
              'G1': 'COMPILER_PREFLIGHT_PASS for shortlist; board resource/latency not measured',
              'G2': 'FOUR_CANDIDATES_SHORTLISTED; quality comparison not yet run',
              'G3': 'RELATIVE_SIGNAL_REPLICATED; predeclared absolute pilot gate FAIL',
              'G4': 'FAITHFUL_QUALITY_MEASURED; correspondence retention gate FAIL',
              'G5': 'OPEN: model-compatible real-map and board qualification required',
              'G6': 'NOT_RELEASE_READY', 'training_running': False, 'final_architecture_selected': False}
    (ROOT / 'gate_status.json').write_text(json.dumps(status,indent=2)+'\n')
    next_gate = {'id': 'G2-QUALITY', 'status': 'PLANNED_NOT_STARTED', 'shortlist': [r['id'] for r in candidates if r['shortlist']],
                 'objective': 'Compare quality under matched training protocol, then qualified deployment cost',
                 'requirements': ['same source images and normalized geometry perturbations', 'same initialization strategy and budget',
                                  'teacher anchor per logical resolution', 'precision plus correct matches and spatial coverage',
                                  'common-coordinate geometry tolerance across resolutions', 'fresh DEV transformations with provenance',
                                  'freeze protocol before running', 'no GT1 tuning'],
                 'scope_limit': 'No final architecture or quality claim from random-weight compile probes'}
    (ROOT / 'next_gate.json').write_text(json.dumps(next_gate,indent=2)+'\n')
    (ROOT / 'candidate_registry.json').write_text(json.dumps(candidates,indent=2)+'\n')
    manifest = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.rglob('*') if p.is_file() and p.suffix in ['.py','.json','.md','.csv','.kmodel','.onnx'] and p.name != 'artifacts.sha256.json'}
    (ROOT / 'artifacts.sha256.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('REPORT_COMPLETE', len(candidates), 'candidates,', sum(r['shortlist'] for r in candidates), 'shortlisted')

if __name__ == '__main__':
    main()
