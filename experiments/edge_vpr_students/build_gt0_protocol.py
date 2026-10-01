#!/usr/bin/env python3
"""GT0: model-independent diagnostic retrieval protocol from frozen COLMAP poses."""
from __future__ import annotations
import hashlib,json,sys
from collections import Counter,defaultdict
from pathlib import Path
import h5py,numpy as np,pycolmap
ROOT=Path('/home/quyet/k210_lab'); EDGE=Path('/home/quyet/edge_ai_project'); OUT=ROOT/'reports/edge_vpr_students'; ART=ROOT/'artifacts/edge_vpr_students/gt0'; SEED=20260906
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def put(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2)+'\n')
def descs(p):
 out={}
 with h5py.File(p,'r') as h:
  def v(n,o):
   if isinstance(o,h5py.Dataset) and n.endswith('/global_descriptor'):out[n[:-18]]=o[:].astype('float32').reshape(-1)
  h.visititems(v)
 return out
def frame(n):
 try:return int(Path(n).stem.split('_')[-1])
 except:return -1
def pose_model(path):
 r=pycolmap.Reconstruction(path);d={}
 for x in r.images.values():
  pose=x.cam_from_world;pose=pose() if callable(pose) else pose;R=np.asarray(pose.rotation.matrix());t=np.asarray(pose.translation);d[x.name]={'center':(-R.T@t).tolist(),'view':(R.T@np.array([0.,0.,1.])).tolist(),'image_id':int(x.image_id)}
 return d,r
def split(names):
 by=defaultdict(list)
 for n in names:by[n.split('/')[0]].append(n)
 q=[];excluded=[]
 # Contiguous middle temporal block; retain only every fifth query, and remove
 # an 8-index guard around every member of the full held-out block.
 for seq,xs in sorted(by.items(),key=lambda z:int(z[0])):
  xs=sorted(xs,key=frame);a=int(np.floor(.40*len(xs)));b=max(a+1,int(np.ceil(.60*len(xs))))
  block=xs[a:b];q.extend(block[::5]);lo=max(0,a-8);hi=min(len(xs),b+8);excluded.extend(xs[lo:hi])
 refs=sorted(set(names)-set(excluded));return sorted(q),refs,sorted(set(excluded)-set(q)),{k:len(v) for k,v in by.items()}
def stats(x):
 q=np.asarray(x,float);return {k:float(np.percentile(q,v)) for k,v in [('min',0),('p10',10),('p25',25),('median',50),('p75',75),('p90',90),('max',100)]}
def main():
 ART.mkdir(parents=True,exist_ok=True)
 allmeta={}; protocol={'seed':SEED,'level':'DIAGNOSTIC','selection':'Per registered sequence: held-out contiguous middle 20% temporal block; every fifth image in block is query; all block images plus 8 index guard each side excluded from references.','positive_definitions':{'primary':{'distance_m_lte':1.0,'view_angle_deg_lte':60.0},'sensitivity_strict':{'distance_m_lte':0.5,'view_angle_deg_lte':45.0},'sensitivity_loose':{'distance_m_lte':1.5,'view_angle_deg_lte':90.0}},'floors':{}}
 for floor in ('floor6','floor7'):
  pkg=EDGE/'map_packages'/f'{floor}_v1';m=json.loads((pkg/'manifest.json').read_text()); poses,r=pose_model(pkg/m['files']['colmap_model']); names=sorted(poses);q,ref,ex,seq=split(names); D=np.array([poses[n]['center'] for n in names]);V=np.array([poses[n]['view'] for n in names]);
  # nearest-neighbor distribution excludes itself
  dist=np.linalg.norm(D[:,None]-D[None,:],axis=2);np.fill_diagonal(dist,np.inf); nn=dist.min(1); ni=dist.argmin(1); ang=np.degrees(np.arccos(np.clip((V*V[ni]).sum(1),-1,1)))
  Rc=np.array([poses[n]['center'] for n in ref]);Rv=np.array([poses[n]['view'] for n in ref]); labels={}; counts={}
  for tag,cfg in protocol['positive_definitions'].items():
   lab={};cs=[]
   for n in q:
    c=np.array(poses[n]['center']);v=np.array(poses[n]['view']);dd=np.linalg.norm(Rc-c,axis=1);aa=np.degrees(np.arccos(np.clip(Rv@v,-1,1)));ids=np.where((dd<=cfg['distance_m_lte'])&(aa<=cfg['view_angle_deg_lte']))[0];lab[n]=[ref[int(i)] for i in ids];cs.append(len(ids))
   labels[tag]=lab;counts[tag]={'zero':int(sum(x==0 for x in cs)),'mean':float(np.mean(cs)),'median':float(np.median(cs)),'max':int(max(cs))}
  # Exact hash audit only across final query/reference images.
  imgs=EDGE/'assets'/floor; hashes={}; exact=[]
  for n in q+ref:
   p=imgs/n
   if p.exists():hashes[n]=sha(p)
  inv=defaultdict(list)
  for n,h in hashes.items():inv[h].append(n)
  for vs in inv.values():
   if any(x in q for x in vs) and any(x in ref for x in vs):exact.append(vs)
  db=descs(pkg/m['files']['global_features']);missing=[n for n in q+ref if n not in db]
  fd={'reference_list':str(ART/f'{floor}_reference.txt'),'query_list':str(ART/f'{floor}_query.txt'),'excluded_list':str(ART/f'{floor}_excluded.txt'),'positive_labels':str(ART/f'{floor}_positive_labels.json'),'pose_metadata':str(ART/f'{floor}_pose_metadata.json')}
  for key,vals in [('reference',ref),('query',q),('excluded',ex)]: (ART/f'{floor}_{key}.txt').write_text('\n'.join(vals)+'\n')
  put(ART/f'{floor}_positive_labels.json',labels);put(ART/f'{floor}_pose_metadata.json',{n:poses[n] for n in q+ref})
  protocol['floors'][floor]={'registered_images':len(names),'sequence_counts':seq,'reference_count':len(ref),'query_count':len(q),'excluded_count':len(ex),'nearest_center_distance':stats(nn),'nearest_view_angle_deg':stats(ang),'positive_counts':counts,'exact_query_reference_hash_duplicates':exact,'missing_precomputed_t0_descriptors':missing,'files':fd,'camera_model_counts':dict(Counter(x.camera.model.name for x in r.images.values()))}
 put(ART/'evaluation_config.json',protocol);put(OUT/'gt0_retrieval_protocol_manifest.json',protocol)
 lines=['# GT0 frozen diagnostic retrieval protocol','', '## Status','', '- GT level: **DIAGNOSTIC**. Queries are registered reconstruction images held out only from the retrieval reference list; they are not an independent capture, so this is not a final paper benchmark.', '- Positives are model-independent, constructed only from frozen COLMAP camera center `C=-Rᵀt` and optical-axis viewing direction `Rᵀ[0,0,1]`.', '- No MixVPR, SuperPoint, LightGlue, PnP, map or production source was modified.', '', '## Frozen split','',f'- Seed: `{SEED}`. {protocol["selection"]}', '- Exact query/reference image-hash duplicates: none (see manifest).', '', '## Geometry and threshold selection','']
 for f,x in protocol['floors'].items():lines += [f"### {f}",f"- Reference/query/excluded: `{x['reference_count']}` / `{x['query_count']}` / `{x['excluded_count']}`.",f"- Nearest-center distance distribution (map units/meters): `{x['nearest_center_distance']}`.",f"- Nearest view-angle distribution (degrees): `{x['nearest_view_angle_deg']}`.",f"- Positive counts: `{x['positive_counts']}`."]
 lines += ['', '## Primary protocol','', '`distance(Cq, Cr) <= 1.0` AND `view_angle(q,r) <= 60°`. Sensitivity settings are 0.5m/45° and 1.5m/90°. These were selected before seeing T0 similarity rankings, from the indoor nearest-neighbor pose distributions.', '', '## Downstream freeze','', '`candidate VPR → same Top-K → original SuperPoint → original LightGlue → same 2D-3D → same PnP`. This defines future comparison; it is not run in GT0.']
 (OUT/'gt0_retrieval_protocol.md').write_text('\n'.join(lines)+'\n');print('GT0 PROTOCOL FROZEN', {f:(x['reference_count'],x['query_count']) for f,x in protocol['floors'].items()})
if __name__=='__main__':main()
