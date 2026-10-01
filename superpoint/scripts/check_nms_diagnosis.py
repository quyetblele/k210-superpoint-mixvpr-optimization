"""Verify plateau mechanism and greedy spacing on real cached INT8 outputs."""
import _bootstrap
import time
import numpy as np
from spk210.runtime import torch
from spk210.settings import ROOT, SUPERPOINT, RES
from spk210.metrics import features
from spk210.nms_diagnosis import greedy
from spk210.io import put, sha

dep=SUPERPOINT/'artifacts/deployment/wider_r2_cw'
dest=SUPERPOINT/'artifacts/nms_diagnosis/wider_r2_cw'
stats={name:[] for name in ('original','greedy_score','greedy_corner')}
for index in range(42):
    with np.load(ROOT/'r2/dev'/f'{index:03d}.npz') as z:
        images=z['pair'].copy(); masks=z['masks'].copy()
    with np.load(dep/'sim_outputs'/f'{index:03d}.npz') as z:
        a,b=(torch.from_numpy(z[k].copy()) for k in ('logits','desc'))
    for name,fn in [('original',lambda:features(a,b,masks,'r2')),
                    ('greedy_score',lambda:greedy(a,b,masks,'r2',images)),
                    ('greedy_corner',lambda:greedy(a,b,masks,'r2',images,True))]:
        start=time.perf_counter(); out=fn(); elapsed=(time.perf_counter()-start)*1000/2
        for xy,d in out:
            distance=np.abs(xy[:,None]-xy[None,:]).max(axis=2)*RES['r2'][2]
            np.fill_diagonal(distance,np.inf)
            fraction=float((distance<=round(3*RES['r2'][2])+1e-6).any(axis=1).mean()) if len(xy) else 0.
            assert np.isfinite(d).all() and len(xy)<=160
            if name!='original': assert fraction==0
            stats[name].append((fraction,elapsed))
put(dest/'spacing_check.json',{
    'status':'PASS','source_sha256':sha(__file__),
    'scope':'84 single-view content masks; timing one CPU pass includes extraction, excludes inference; not board timing',
    'results':{k:{'fraction_points_with_neighbor_inside_NMS_radius':float(np.mean(np.array(v)[:,0])),
                  'median_extraction_ms':float(np.median(np.array(v)[:,1]))} for k,v in stats.items()}})
print((dest/'spacing_check.json').read_text())
