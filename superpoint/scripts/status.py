import _bootstrap
import json
from spk210.settings import ROOT, SUPERPOINT
print('SuperPoint canonical workspace:', SUPERPOINT)
print('RESEARCH_GATE: see docs/EXPERIMENT_STATUS.md; no automatic training or promotion')
print('PUBLIC_BENCHMARK: NOT_RUN; untouched TEST: NOT YET ESTABLISHED')
for name in ['run_status.json', 'summary.json']:
    path = ROOT / name
    if path.exists():
        d = json.loads(path.read_text())
        print(name, {k: d[k] for k in ['status', 'candidate', 'selected_for_next_gate'] if k in d})
    else:
        print(name, 'NOT_RUN')
print('Board latency / actual SRAM peak: NOT_MEASURED')
summary=ROOT/'summary.json'
if summary.exists():
    candidate=json.loads(summary.read_text())['selected_for_next_gate']
    folder=SUPERPOINT/'artifacts/deployment'/candidate
    for name in ['export.json','onnx_parity.json','compile.json','simulation.json','int8_quality.json']:
        path=folder/name
        if path.exists():
            data=json.loads(path.read_text())
            fields=['status','direct_FP32_raw_strict','all42_DEV_metrics','PTQ','compile','gencode','CPU_Conv2D_count','KPU_Conv2D_count','INT8_QUALITY']
            if name=='export.json':data['status']='EXPORTED; validation reported separately below'
            print(name,{k:data[k] for k in fields if k in data})
        else:print(name,'NOT_RUN')
board=SUPERPOINT/'results/board_status.json'
if board.exists():print('BOARD',json.loads(board.read_text())['status'])
completion=SUPERPOINT/'results/completion.json'
if completion.exists():
    current=json.loads(completion.read_text())
    print('HISTORICAL_ENGINEERING_GATE',current['next_gate'],'release_ready=',current['INT8_candidate_release_ready'])
recovery=SUPERPOINT/'artifacts/detector_recovery/status.json'
if recovery.exists():print('HISTORICAL_DETECTOR_RECOVERY',json.loads(recovery.read_text()))
