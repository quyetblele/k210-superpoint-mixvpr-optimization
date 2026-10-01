"""Synthetic geometry + mock descriptors: software validation, NOT accuracy."""
import json,tempfile
from pathlib import Path
import numpy as np
import cv2
from pipeline import normalize,MapStore,Pipeline,frames,match

def main():
 rng=np.random.default_rng(17);cfg=json.loads(Path(__file__).with_name('config.json').read_text());K=np.array([[500,0,320],[0,500,240],[0,0,1]],float)
 xyz=np.column_stack([rng.uniform(-1,1,80),rng.uniform(-.8,.8,80),rng.uniform(4,7,80)]).astype(np.float32)
 desc=normalize(rng.normal(size=(80,256)));g=normalize(rng.normal(size=512));uv=(xyz[:,:2]/xyz[:,2:])*500+[320,240]
 class Global:
  identity='synthetic-global'
  def infer(self,rgb):assert rgb.shape==(240,240,3);return g
 class Local:
  identity='synthetic-local'
  def infer(self,gray):assert gray.shape==(240,320);return ((uv+.5)/2-.5).astype(np.float32),desc
 with tempfile.TemporaryDirectory(prefix='k210_online_') as tmp:
  root=Path(tmp);arrays={'global':np.stack([g,-g]),'image_ids':np.array([10,20],np.int64),'point_ids':np.arange(80,dtype=np.int64)+100,'xyz':xyz,'local':desc,'vis_offsets':np.array([0,40,80],np.int64),'vis_rows':np.arange(80,dtype=np.int64),'cov_offsets':np.array([0,1,2],np.int64),'cov_rows':np.array([1,0],np.int64)}
  for name,a in arrays.items():np.save(root/(name+'.npy'),a)
  (root/'map.json').write_text(json.dumps({'version':1,'global_identity':Global.identity,'local_identity':Local.identity,'coordinate_frame':'synthetic world; XYZ meters','local_aggregation':'synthetic unit descriptor per Point3D'}))
  store=MapStore(root,Global.identity,Local.identity)
  assert len(store.active([0],1,15))==15
  # Streamed matcher must agree with dense reference, including runner-up across blocks.
  out,raw=match(desc,store,np.arange(80),cfg);assert raw==80 and all(i==p for i,p,s in out)
  duplicate=np.stack([desc[0],desc[0]]);out,raw=match(duplicate,store,np.arange(80),cfg);assert raw==2 and len(out)==1
  pipe=Pipeline(store,Global(),Local(),cfg,K)
  video=root/'sample.avi';writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'MJPG'),10,(640,480));assert writer.isOpened()
  for _ in range(3):writer.write(np.zeros((480,640,3),np.uint8))
  writer.release();results=[pipe.frame(bgr,i,t) for i,t,bgr in frames(str(video))]
  assert len(results)==3 and all(r['status']=='pose' and r['pnp_inliers']==80 for r in results)
  assert results[1]['retrieval_reused'];assert np.linalg.norm(results[0]['camera_center_world'])<1e-4
  class Empty(Local):
   def infer(self,gray):return np.empty((0,2),np.float32),np.empty((0,256),np.float32)
  assert Pipeline(store,Global(),Empty(),cfg,K).frame(np.zeros((480,640,3),np.uint8),0,0)['status']=='insufficient_matches'
  assert pipe.frame(np.zeros((100,100,3),np.uint8),3,0)['status']=='error'
  try:MapStore(root,'wrong-model',Local.identity)
  except ValueError:pass
  else:raise AssertionError('identity guard failed')
  target=Path(__file__).with_name('smoke_report.json');target.write_text(json.dumps({'status':'PASS','scope':'synthetic software smoke only; mock global/local backends; real video decoding, retrieval, map loading, matcher, OpenCV PnP','frames':results,'checks':['bounded active map','streamed nearest match','Point3D dedupe','retrieval reuse','known-pose recovery','empty features','bad calibration size','model-map identity mismatch'],'GPU_used':False,'GT1_used':False},indent=2)+'\n')
  print('PASS:',target)
if __name__=='__main__':main()
