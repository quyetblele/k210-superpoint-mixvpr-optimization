"""Summarize measured canonical results; never changes checkpoints or training selection."""
import json,hashlib
from pathlib import Path
w=Path(__file__).resolve().parents[1];sp=w/'superpoint';run=sp/'artifacts/quality_pilot'
def read(p):return json.loads(p.read_text())
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def put(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
s=read(run/'summary.json');name=s['selected_for_next_gate'];dep=sp/'artifacts/deployment'/name
diagnosis=read(dep/'head_diagnosis.json');quality=read(dep/'int8_quality.json');comp=read(dep/'compile.json');parity=read(dep/'onnx_parity.json');timing=read(run/'laptop_timing.json');cleanup=read(w/'docs/audit/cleanup.json')
lines=['# SuperPoint reference workspace — execution report','',
       'Workspace đã chuẩn hóa và pipeline được chạy thực tế qua training, export, ONNX qualification, TRAIN-only PTQ, compile/gencode và faithful simulator quality. Đây chưa phải final board-qualified/localization release.','',
       f'Frozen pilot protocol SHA256: `{sha(run/"protocol.json")}`. Selected checkpoint: `{name}/best.pt`, SHA256 `{comp["checkpoint_sha256"]}`.','',
       '## Matched quality pilot','',
       '| Candidate | Best update | Correct matches/pair | MNN precision | Repeatability | Coverage |','|---|---:|---:|---:|---:|---:|']
for r in s['comparison']:lines.append(f"|{r['candidate']}|{r['best_step']}|{r['correct_mnn_count']:.2f}|{r['mnn_precision']:.2%}|{r['repeatability']:.2%}|{r['coverage']:.2%}|")
lines += ['', 'Đây là equal-scene macro synthetic-homography DEV; không phải pose success/R@1. 140TRAIN/42DEV sources, 400updates/candidate, same source plan/init strategy/canonical correspondences. Một seed, fixed homography pairs, project capture sequences có liên quan; chưa independent test hoặc train đến plateau. GT1/MixVPR không dùng.','',
          '## Trained-weight deployment quality','', '| Backend | Correct matches | Precision | Repeatability | Coverage |','|---|---:|---:|---:|---:|']
for label in ['fp32','int8']:
    m=quality[label]['macro'];lines.append(f"|{label}|{m['correct_mnn_count']:.2f}|{m['mnn_precision']:.2%}|{m['repeatability']:.2%}|{m['coverage']:.2%}|")
lines += ['',f"Correct-match retention INT8/FP32: **{quality['macro_retention']['correct_mnn_count']:.2%}**; dense descriptor cosine {quality['mean_valid_dense_descriptor_cosine']:.6f}. Đây chỉ là một chỉ số retention, không phải overall teacher-quality percentage.", '',
          '## Project-domain subset','', '| Backend | Correct matches | Precision | Repeatability |','|---|---:|---:|---:|']
for label in ['fp32','int8']:
    m=quality[label]['per_domain']['project'];lines.append(f"|{label}|{m['correct_mnn_count']:.2f}|{m['mnn_precision']:.2%}|{m['repeatability']:.2%}|")
lines += ['',f"PTQ/compile/gencode: PASS. kmodel **{comp['kmodel_bytes']:,}bytes**; CPU Conv={comp['CPU_Conv2D_count']}, KPU Conv={comp['KPU_Conv2D_count']}. kmodel SHA256 `{comp['kmodel_sha256']}`.",'',
          f"ONNX qualification: {parity['status']}. Direct raw FP32 strict comparison: `{parity['direct_FP32_raw_strict']}`. Lỗi accumulation rounding được giữ trong report; không nới tolerance. Same-weight float64 reference, rotation equivalence, normalized feature checks PASS; toàn bộ42DEV metric identical với original FP32.",'',
          '## Laptop timing','', '| Candidate | Network median ms | Network+post median ms | P90 ms |','|---|---:|---:|---:|']
for r in timing['results']:lines.append(f"|{r['candidate']}|{r['network_median_ms']:.2f}|{r['network_and_postprocess_median_ms']:.2f}|{r['network_and_postprocess_p90_ms']:.2f}|")
lines += ['',timing['scope'],'', f"Training peak process RSS cao nhất: {max(r['peak_RSS_MiB'] for r in s['comparison']):.1f}MiB; CPU2threads, workers0, physical2images, accumulation2pairs, teacher disk cache. Không dùng training VRAM.",'',
          '## Workspace cleanup và bảo toàn','',f"Xóa {len(cleanup['deleted'])} Python bytecode caches ({cleanup['bytes_removed']:,}bytes, có thể tái tạo). Không xóa dataset, checkpoint hay benchmark. README cũ archive; active artifact chuyển về canonical folder bằng move+compatibility symlink, không copy dataset. Thí nghiệm cũ archive tại chỗ vì dependency/provenance.",'',
          '## Chẩn đoán nhánh lượng tử hóa','', f"FP32 detector + INT8 descriptor: {diagnosis['hybrids']['FP32_detector_INT8_descriptor']['macro']['correct_mnn_count']:.2f} correct matches. INT8 detector + FP32 descriptor: {diagnosis['hybrids']['INT8_detector_FP32_descriptor']['macro']['correct_mnn_count']:.2f}. Mất chất lượng chủ yếu nằm ở nhánh detector/selection. Đây là hybrid diagnostic, không phải deploy result; chưa đủ kết luận chỉ riêng convPb gây lỗi.",'', '## Một gate tiếp theo','', '**DETECTOR QUANTIZATION RECOVERY:** thiết kế một A/B ngắn có đối chứng để tăng độ ổn định detector khi lượng tử hóa, rồi kiểm tra trained-weight nncase/PTQ thật. Giữ kiến trúc/descriptor path hiện tại trong bước này. Chưa promotion student, chưa rebuild production map, chưa mở full train mặc định.','',
          'Board gate hiện BLOCKED: chưa có qualified firmware/transport adapter và không có board timing/SRAM measurement. Compile planned memory hoặc simulator time không được gọi là số đo board. Package hiện là engineering package, chưa release.','',
          'Tài liệu học/chạy: `../../docs/WORKFLOW.md`, `../../docs/CODE_READING_GUIDE.md`. Chi tiết từng image/scene/loss/curve: `../artifacts/quality_pilot/`; ONNX/PTQ/simulator/package: `../artifacts/deployment/`.' ]
(sp/'results/REPORT.md').write_text('\n'.join(lines)+'\n')
put(sp/'results/completion.json',{'workspace_refactor':'COMPLETE','quality_pilot':'COMPLETE','trained_weight_PTQ_compile_gencode':'PASS','faithful_INT8_quality':'MEASURED','selected_candidate':name,'checkpoint_sha256':comp['checkpoint_sha256'],'kmodel_sha256':comp['kmodel_sha256'],'board':'BLOCKED_NO_QUALIFIED_ADAPTER','INT8_candidate_release_ready':False,'quantization_bottleneck':'detector_path','next_gate':'DETECTOR_QUANTIZATION_RECOVERY','real_map_candidate_validation':'DEFERRED_UNTIL_QUANTIZATION_QUALITY','training_running':False})
# Preserve original experiment identity; record final source versions separately.
files=sorted((sp/'src').rglob('*.py'))+sorted((sp/'scripts').glob('*.py'))+[sp/'run']
put(w/'docs/audit/final_sources.json',{str(f.relative_to(w)):sha(f) for f in files})
put(sp/'results/artifacts.sha256.json',{str(f.relative_to(sp)):sha(f) for f in sorted(sp.rglob('*')) if f.is_file() and f.suffix in ['.pt','.kmodel','.onnx','.json','.md','.csv','.png','.pdf'] and f.name!='artifacts.sha256.json' and '__pycache__' not in f.parts})
legacy=w/'experiments/superpoint_sop_v2';g=read(legacy/'gate_status.json');g.update(G2='MATCHED_QUALITY_PILOT_COMPLETE; canonical evidence in superpoint/artifacts/quality_pilot',G4='SELECTED_TRAINED_KMODEL_FAITHFUL_QUALITY_MEASURED; see canonical report',training_running=False,final_architecture_selected=False);put(legacy/'gate_status.json',g)
put(legacy/'next_gate.json',{'id':'DETECTOR_QUANTIZATION_RECOVERY','status':'NOT_STARTED','candidate':name,'canonical_report':str(sp/'results/REPORT.md'),'objective':'bounded controlled detector-quantization recovery experiment followed by actual trained-weight nncase evaluation; no full train or GT1 tuning','board_gate':'BLOCKED until qualified firmware/transport and actual measurements'})
report=legacy/'REPORT.md';text=report.read_text();notice='> Current continuation: canonical workspace and measured pilot/deployment results are in `/home/quyet/k210_lab/superpoint/results/REPORT.md`. The report below preserves the earlier SOP round.\n\n'
if not text.startswith('> Current continuation:'):report.write_text(notice+text)
print(sp/'results/REPORT.md')
