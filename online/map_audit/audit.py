import os
os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
import sys,json,collections,time
from pathlib import Path
import numpy as np,h5py,cv2
cv2.setNumThreads(1)
sys.path.insert(0,'/home/quyet/edge_ai_project/third_party/hloc')
from hloc.utils.read_write_model import read_images_binary,read_cameras_binary,qvec2rotmat
ROOT=Path(__file__).parent;SRC=Path('/home/quyet/edge_ai_project/map_packages/stair_6_7_v1')
def norm(x):return x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-12)
def main():
 idx=np.load(SRC/'indexes/point3d-index-refined_model_v2.npz');ids=idx['point_ids'];xyz=idx['xyz'];errors=idx['errors'];tracks=idx['track_lengths'];lookup={int(p):i for i,p in enumerate(ids)}
 images=read_images_binary(SRC/'colmap/refined_model_v2/images.bin');cams=read_cameras_binary(SRC/'colmap/refined_model_v2/cameras.bin')
 data=json.load(open('/home/quyet/training_recovery/full/data.json'))
 names={ '/'.join(Path(r['path']).parts[-2:]) for r in data['rows'] if r['domain']=='project' and r['split']=='dev'}
 queries=[i for i in sorted(images) if images[i].name in names]
 assert queries,'no DEV query overlap'
 protocol={'scope':'internal reconstruction audit, NOT independent test; reconstructed query poses/geometry remain in map','query_ids':queries,'query_rule':'all existing project DEV images registered in stair map','query_features':'legacy stored SuperPoint; top160 by score including untracked features','self_leakage_control':'subtract query-image observation contributions from each descriptor centroid before matching; exclude zero remaining observations','retrieval':'full map scan in blocks; no oracle subset, not final active-map online performance','variants':['FP32','FP16','INT8_per_descriptor','PRUNE_track5_error1p75_FP32'],'selection':'provisional smallest storage among variants with no lost baseline good-pose query, median rotation increase<=0.1deg, median center increase<=0.01 native units','good_pose':'PnP>=8 inliers and rotation error<=5deg and center error<=0.1 native units','matcher':'cosine>=0.8; best-second>=0.05; Point3D dedupe; max120; block128','pnp':'EPNP 100 iterations 4px confidence.99 deterministic seed17','pixel_coordinates':'COLMAP stored observation xy; cached feature xy checked within 1.01px rounding tolerance','GT1':False,'network_quantization':False}
 (ROOT/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
 sums=np.zeros((len(ids),256),np.float32);counts=np.zeros(len(ids),np.int32)
 with h5py.File(SRC/'features/features-v2.h5','r') as f:
  for im in images.values():
   d=norm(f[im.name]['descriptors'][:].T.astype(np.float32));assert len(d)==len(im.point3D_ids)
   for j,p in enumerate(im.point3D_ids):
    if int(p) in lookup:r=lookup[int(p)];sums[r]+=d[j];counts[r]+=1
  parity=float(np.max(np.abs(norm(sums)-idx['descriptors'])))
  assert parity<1e-4,('source centroid mismatch',parity)
  results=[]
  for image_id in queries:
   im=images[image_id];g=f[im.name];d=norm(g['descriptors'][:].T.astype(np.float32));xy=g['keypoints'][:];scores=g['scores'][:];order=np.argsort(-scores,kind='stable')[:160];q=d[order];pix=xy[order].astype(np.float64)
   assert np.max(np.abs(xy-im.xys))<=1.01,'feature/reconstruction alignment exceeds pixel-rounding tolerance'
   pix=im.xys[order].astype(np.float64)
   held=sums.copy();cnt=counts.copy()
   for j,p in enumerate(im.point3D_ids):
    if int(p) in lookup:r=lookup[int(p)];held[r]-=d[j];cnt[r]-=1
   base=norm(held);camera=cams[im.camera_id];assert camera.model=='SIMPLE_RADIAL'
   focal,cx,cy,k1=camera.params;K=np.array([[focal,0,cx],[0,focal,cy],[0,0,1.]]);dist=np.array([k1,0,0,0,0.]);Rgt=qvec2rotmat(im.qvec);Cgt=-Rgt.T@im.tvec
   for variant in protocol['variants']:
    mask=cnt>0
    if variant.startswith('PRUNE'):mask &= (tracks>=5)&(errors<=1.75)
    rows=np.flatnonzero(mask);best=np.full(len(q),-np.inf);second=best.copy();winner=np.full(len(q),-1,int);start=time.perf_counter()
    for s in range(0,len(rows),128):
     rr=rows[s:s+128];b=base[rr]
     if variant=='FP16':b=b.astype(np.float16).astype(np.float32)
     if variant=='INT8_per_descriptor':
      scale=np.maximum(abs(b).max(1,keepdims=True)/127,1e-12);b=np.rint(b/scale).clip(-127,127).astype(np.int8).astype(np.float32)*scale
     sim=q@norm(b).T
     for j,r in enumerate(rr):
      val=sim[:,j];take=val>best;second=np.where(take,best,np.maximum(second,val));best=np.where(take,val,best);winner=np.where(take,r,winner)
    chosen=[j for j in range(len(q)) if best[j]>=.8 and best[j]-second[j]>=.05];chosen=sorted(chosen,key=lambda j:-best[j]);used=set();pairs=[]
    for j in chosen:
     if winner[j] not in used:used.add(winner[j]);pairs.append(j)
    pairs=pairs[:120];r={'image_id':image_id,'variant':variant,'map_points':len(rows),'matches':len(pairs),'inliers':0,'pose_good':False,'rotation_deg':None,'center_native':None,'matching_ms':(time.perf_counter()-start)*1000}
    if len(pairs)>=8:
     cv2.setRNGSeed(17);ok,rv,tv,ins=cv2.solvePnPRansac(xyz[winner[pairs]].astype(np.float64),pix[pairs],K,dist,iterationsCount=100,reprojectionError=4.,confidence=.99,flags=cv2.SOLVEPNP_EPNP)
     if ok and ins is not None:
      R=cv2.Rodrigues(rv)[0];angle=float(np.degrees(np.arccos(np.clip((np.trace(R@Rgt.T)-1)/2,-1,1))));ce=float(np.linalg.norm(-R.T@tv.ravel()-Cgt));r.update(inliers=len(ins),rotation_deg=angle,center_native=ce,pose_good=bool(len(ins)>=8 and angle<=5 and ce<=.1))
    results.append(r)
   print('query',image_id,'done',flush=True)
 (ROOT/'queries.json').write_text(json.dumps(results,indent=2)+'\n')
 summary={}
 for variant in protocol['variants']:
  rs=[r for r in results if r['variant']==variant];valid=[r for r in rs if r['rotation_deg'] is not None]
  summary[variant]={'queries':len(rs),'good_pose':sum(r['pose_good'] for r in rs),'median_rotation_deg':float(np.median([r['rotation_deg'] for r in valid])) if valid else None,'median_center_native':float(np.median([r['center_native'] for r in valid])) if valid else None,'median_matches':float(np.median([r['matches'] for r in rs]))}
 full=norm(sums);np.save(ROOT/'local_fp16.npy',full.astype(np.float16));scale=np.maximum(abs(full).max(1)/127,1e-12).astype(np.float32);np.save(ROOT/'local_int8.npy',np.rint(full/scale[:,None]).clip(-127,127).astype(np.int8));np.save(ROOT/'local_int8_scale.npy',scale);np.save(ROOT/'prune_candidate_rows.npy',np.flatnonzero((tracks>=5)&(errors<=1.75)).astype(np.uint32))
 report={'status':'COMPLETE_INTERNAL_AUDIT','geometry':{'points':len(ids),'error_px_quantiles':np.quantile(errors,[0,.5,.9,1]).tolist(),'track_quantiles':np.quantile(tracks,[0,.5,.9,1]).tolist(),'source_centroid_max_abs_difference':parity},'results':summary,'artifacts':'candidate descriptor payloads only; original map/runtime unchanged','limitations':protocol['scope'],'selection':'pending paired no-regression assessment; independent test required'}
 (ROOT/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
if __name__=='__main__':main()
