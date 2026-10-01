"""Repair correspondence matching without regenerating identical teacher caches."""
import os,json
from pathlib import Path
import numpy as np
import quality_pilot_v2 as q
OLD=Path(__file__).parent/'quality_pilot'
q.ROOT.mkdir(exist_ok=True)
q.verify(); p=q.protocol()
def link(a,b):
    if not b.exists():os.link(a,b)
for name in q.WIDTHS:
    for suffix in ['_init.pt','_channel_map.json']:link(OLD/(name+suffix),q.ROOT/(name+suffix))
rows=[]
for res in q.RES:
    (q.ROOT/res).mkdir(exist_ok=True);link(OLD/res/'teacher_dev.json',q.ROOT/res/'teacher_dev.json')
    for split in ['train','dev']:
        dest=q.ROOT/res/split;dest.mkdir(exist_ok=True);link(OLD/res/split/'teacher_logits.npy',dest/'teacher_logits.npy')
for split,rr in p['rows'].items():
    for i in range(len(rr)):
        filename=f'{i:03d}.npz'
        with np.load(OLD/'r1'/split/filename) as z:a={k:z[k] for k in z.files}
        with np.load(OLD/'r2'/split/filename) as z:b={k:z[k] for k in z.files}
        bp=(b['points']-np.array([2,0]))/1.25
        dist=np.linalg.norm(a['points'][0,:,None]-bp[0,None],axis=2)
        keep=dist.min(1)<1e-4; common=a['points'][:,keep]
        assert len(common[0])>=8
        # Same canonical point list/order in both views and resolutions.
        a['points']=common.astype(np.float32);b['points']=(common*1.25+np.array([2,0])).astype(np.float32)
        np.savez_compressed(q.ROOT/'r1'/split/filename,**a);np.savez_compressed(q.ROOT/'r2'/split/filename,**b)
        assert np.allclose((b['points']-np.array([2,0]))/1.25,a['points'],atol=2e-5)
        rows.append({'split':split,'index':i,'points':int(keep.sum()),'removed_from_r1':int((~keep).sum())})
q.put(q.ROOT/'correspondence_audit.json',{'status':'PASS','canonical_coordinates_identical':True,'rows':rows,'source_protocol_sha256':q.sha(OLD/'protocol.json'),'preparation_code_sha256':q.sha(__file__)})
files=sorted(f for f in q.ROOT.rglob('*') if f.is_file() and f.suffix in ('.npy','.npz','.pt','.json') and f.name!='cache_done.json')
q.put(q.ROOT/'cache_done.json',{'protocol_sha256':q.sha(q.ROOT/'protocol.json'),'files':{str(f.relative_to(q.ROOT)):q.sha(f) for f in files}})
q.put(OLD/'SUPERSEDED.json',{'reason':'resolution-dependent border filtering changed some training correspondences','replacement':str(q.ROOT),'v1_results_used_for_ranking':False,'data_and_teacher_caches_reused':True})
print('V2_CACHE_READY',len(rows),'exactly matched point lists',flush=True)
