"""Real local-map structural integration, without fabricated global descriptors."""
import json
from pathlib import Path
import numpy as np
from pipeline import MapStore,match
p=Path(__file__).parent;root=p/'maps/stair_local_v1';cfg=json.loads((p/'config.json').read_text())
# Partial MapStore intentionally bypasses global loading; validate local path only.
s=MapStore.__new__(MapStore)
for name in ['point_ids','xyz','local','vis_offsets','vis_rows','cov_offsets','cov_rows']:
 setattr(s,name,np.load(root/(name+'.npy'),mmap_mode='r'))
active=s.active([0],cfg['max_active_images'],cfg['max_active_points'])
assert 0<len(active)<=cfg['max_active_points'];assert len(set(active.tolist()))==len(active)
q=np.asarray(s.local[active[:min(32,len(active))]],np.float32)
found,raw=match(q,s,active,cfg)
# Dense check only on small development fixture; production matcher stays streamed.
sim=(q/np.linalg.norm(q,axis=1,keepdims=True))@(s.local[active]/np.linalg.norm(s.local[active],axis=1,keepdims=True)).T
for i,row,score in found:assert row==active[sim[i].argmax()] and abs(score-sim[i].max())<1e-5
r={'status':'PASS','scope':'real-map active loading and streamed matcher vs small dense reference; self-descriptors are not localization accuracy','active_points':len(active),'query_descriptors':len(q),'accepted':len(found),'raw':raw,'global_model_used':False,'GT1_used':False}
(root/'integration_report.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
