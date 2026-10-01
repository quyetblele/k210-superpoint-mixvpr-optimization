"""Prepare immutable local-map payload; global binding is a separate final stage."""
import argparse,json,sys,hashlib,collections,zipfile,shutil
from pathlib import Path
import numpy as np
sys.path.insert(0,'/home/quyet/edge_ai_project/third_party/hloc')
from hloc.utils.read_write_model import read_images_binary

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()

def prepare(source,out):
 if out.exists():raise ValueError('destination exists; refusing overwrite')
 meta=json.loads((source/'manifest.json').read_text());idx=source/meta['files']['point3d_index'];model=source/meta['files']['colmap_model']
 out.mkdir(parents=True)
 # Stream zip members instead of decompressing the entire archive into RAM.
 with zipfile.ZipFile(idx) as z:
  for member,target in [('descriptors.npy','local.npy'),('point_ids.npy','point_ids.npy'),('xyz.npy','xyz.npy')]:
   with z.open(member) as a,open(out/target,'wb') as b:shutil.copyfileobj(a,b,1048576)
 ids=np.load(out/'point_ids.npy',mmap_mode='r');xyz=np.load(out/'xyz.npy',mmap_mode='r');local=np.load(out/'local.npy',mmap_mode='r')
 if xyz.shape!=(len(ids),3) or local.shape!=(len(ids),256):raise ValueError('invalid source shapes')
 lookup={int(p):i for i,p in enumerate(ids)}
 if len(lookup)!=len(ids):raise ValueError('duplicate source Point3D IDs')
 for start in range(0,len(ids),256):
  d=local[start:start+256];norm=np.linalg.norm(d,axis=1)
  if not np.isfinite(d).all() or not np.isfinite(xyz[start:start+256]).all() or np.any(norm<1e-8):raise ValueError('invalid geometry/descriptor')
 images=read_images_binary(model/'images.bin');keys=sorted(images)
 vis=[];offset=[0];postings=collections.defaultdict(list)
 for row,k in enumerate(keys):
  points=sorted({lookup[int(p)] for p in images[k].point3D_ids if int(p) in lookup});vis.extend(points);offset.append(len(vis))
  for p in points:postings[p].append(row)
 cov=[];co=[0]
 for row in range(len(keys)):
  count=collections.Counter()
  for p in vis[offset[row]:offset[row+1]]:count.update(postings[p])
  count.pop(row,None)
  # Preserve full baseline covisibility, highest shared-track count first.
  cov.extend(k for k,v in sorted(count.items(),key=lambda x:(-x[1],x[0])));co.append(len(cov))
 for name,a in [('image_ids',keys),('vis_offsets',offset),('vis_rows',vis),('cov_offsets',co),('cov_rows',cov)]:np.save(out/(name+'.npy'),np.asarray(a,np.int64))
 (out/'reference_images.json').write_text(json.dumps([{'image_id':k,'name':images[k].name} for k in keys],indent=2)+'\n')
 camera=source/meta['files']['camera_config'];shutil.copyfile(camera,out/'source_camera.json')
 manifest={'version':1,'status':'LOCAL_MAP_READY_GLOBAL_UNBOUND','source_package':str(source),'source_index_sha256':sha(idx),'source_images_sha256':sha(model/'images.bin'),'global_identity':None,'local_identity':'UNVERIFIED_LEGACY_SUPERPOINT','local_aggregation':'source index: mean of normalized observation descriptors, normalized output; see source build_point3d_index.py','coordinate_frame':'source COLMAP world; scale/unit not assumed metric','num_images':len(keys),'num_points':len(ids),'global_reference_dim_required':512,'pruning':False,'descriptor_quantization':False,'covisibility':'all connected images ordered by shared valid Point3D count','pending':['verify local extraction configuration against query SuperPoint','bind final global descriptors in reference_images.json order','validate camera calibration for live source']}
 (out/'map.json').write_text(json.dumps(manifest,indent=2)+'\n')
 report={'status':'PASS_LOCAL_MAP_STRUCTURE','images':len(keys),'points':len(ids),'visibility_entries':len(vis),'covisibility_entries':len(cov),'bytes':{p.name:p.stat().st_size for p in out.glob('*.npy')},'gt1_access':False,'quality_evaluation':False}
 (out/'preparation_report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();prepare(a.source,a.output)
