"""Position-only PC experiment with frozen SuperPoint and selected student."""
import argparse
import json
from pathlib import Path
import numpy as np
from backends import TorchStudent, OnnxStudent, identity
from frozen_superpoint import FrozenSuperPoint
from pipeline import MapStore, Pipeline, frames

def main():
    p=argparse.ArgumentParser()
    for name in ('map','model','calibration','config','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--candidate',choices=['A_C144_H112','B_C160_H96'])
    p.add_argument('--backend',choices=['fp32','onnx'],default='fp32')
    src=p.add_mutually_exclusive_group(required=True)
    src.add_argument('--video');src.add_argument('--camera',type=int)
    p.add_argument('--max-frames',type=int,default=100)
    a=p.parse_args()
    cfg=json.loads(a.config.read_text());cal=json.loads(a.calibration.read_text())
    if cal.get('model') not in ('PINHOLE','OPENCV'):
        raise ValueError('Explicit PINHOLE/OPENCV calibration required; fisheye needs verified rectification')
    if cal['size']!=cfg['camera_size']:raise ValueError('calibration resolution mismatch')
    g=TorchStudent(a.model,a.candidate) if a.backend=='fp32' else OnnxStudent(a.model)
    l=FrozenSuperPoint()
    store=MapStore(a.map,g.identity,l.identity)
    pipe=Pipeline(store,g,l,cfg,cal['K'],cal.get('distortion'))
    a.output.parent.mkdir(parents=True,exist_ok=True)
    counts={};latencies=[]
    # Exclusive output creation prevents overwriting a previous experiment.
    with a.output.open('x') as f:
        f.write(json.dumps({'type':'run','global_identity':g.identity,'local_identity':l.identity,
            'map_metadata_sha256':identity(a.map/'map.json','sha256'),
            'config':cfg,'calibration':cal,'platform':'PC CPU','GT1_tuning':False})+'\n')
        for i,t,bgr in frames(a.camera if a.camera is not None else a.video):
            if i>=a.max_frames:break
            result=pipe.frame(bgr,i,t)
            counts[result['status']]=counts.get(result['status'],0)+1
            latencies.append(result['timings_ms']['total'])
            f.write(json.dumps(result,allow_nan=False)+'\n');f.flush()
    report={'frames':sum(counts.values()),'statuses':counts,'platform':'PC CPU',
            'latency_ms_p50':float(np.median(latencies)) if latencies else None,
            'latency_ms_p95':float(np.percentile(latencies,95)) if latencies else None,
            'accuracy':'UNMEASURED: requires independent ground truth'}
    a.output.with_suffix('.summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))

if __name__=='__main__':main()
