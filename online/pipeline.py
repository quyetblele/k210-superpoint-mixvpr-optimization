"""Bounded-memory PC reference implementation; no board-performance claims."""
import os
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import json,time
from pathlib import Path
from typing import Protocol
import numpy as np
import cv2
from PIL import Image
cv2.setNumThreads(1)

def normalize(x):
 x=np.asarray(x,dtype=np.float32)
 if not np.isfinite(x).all():raise ValueError('nonfinite descriptor')
 n=np.linalg.norm(x,axis=-1,keepdims=True)
 if np.any(n<1e-12):raise ValueError('zero descriptor')
 return x/n

class GlobalBackend(Protocol):
 identity:str
 def infer(self,rgb240:np.ndarray)->np.ndarray: ... # raw pre-L2 [512]
class LocalBackend(Protocol):
 identity:str
 def infer(self,gray:np.ndarray)->tuple: ... # pixel xy [Q,2], descriptors [Q,256]

class MapStore:
 def __init__(self,path,global_identity,local_identity):
  root=Path(path);self.meta=json.loads((root/'map.json').read_text())
  if self.meta['version']!=1:raise ValueError('unsupported map version')
  if self.meta['global_identity']!=global_identity or self.meta['local_identity']!=local_identity:raise ValueError('map/model identity mismatch: rebuild descriptors for selected backend')
  for name in ['global','image_ids','point_ids','xyz','local','vis_offsets','vis_rows','cov_offsets','cov_rows']:
   setattr(self,name,np.load(root/(name+'.npy'),mmap_mode='r',allow_pickle=False))
  n=len(self.image_ids);p=len(self.point_ids)
  if self.global_shape()!=(n,512) or self.local.shape!=(p,256) or self.xyz.shape!=(p,3):raise ValueError('map shape mismatch')
  for name,limit in [('vis',p),('cov',n)]:
   offsets=getattr(self,name+'_offsets');rows=getattr(self,name+'_rows')
   if offsets.shape!=(n+1,) or offsets[0]!=0 or offsets[-1]!=len(rows) or np.any(np.diff(offsets)<0):raise ValueError('invalid CSR offsets')
   for start in range(0,len(rows),4096):
    b=rows[start:start+4096]
    if np.any(b<0) or np.any(b>=limit):raise ValueError('invalid CSR row')
 def global_shape(self):return getattr(self,'global').shape
 def retrieve(self,q,k,block):
  best=[];g=getattr(self,'global')
  for start in range(0,len(g),block):
   score=normalize(g[start:start+block])@q
   best.extend((float(s),start+j) for j,s in enumerate(score));best=sorted(best,key=lambda x:(-x[0],x[1]))[:k]
  return [i for s,i in best]
 def active(self,refs,max_images,max_points):
  images=list(dict.fromkeys(refs))[:max_images]
  for r in list(images):
   for v in self.cov_rows[self.cov_offsets[r]:self.cov_offsets[r+1]]:
    if len(images)>=max_images:break
    if int(v) not in images:images.append(int(v))
  seen=set();chosen=[]
  for r in images:
   for p in self.vis_rows[self.vis_offsets[r]:self.vis_offsets[r+1]]:
    p=int(p)
    if p not in seen:seen.add(p);chosen.append(p)
    if len(chosen)>=max_points:return np.asarray(chosen,dtype=np.int64)
  return np.asarray(chosen,dtype=np.int64)

def match(query,store,active,cfg):
 q=normalize(query);count=len(q)
 if count==0 or len(active)==0:return [],0
 best=np.full(count,-np.inf,np.float32);second=best.copy();winner=np.full(count,-1,np.int64)
 for start in range(0,len(active),cfg['match_block']):
  rows=active[start:start+cfg['match_block']];scores=q@normalize(store.local[rows]).T
  for j,row in enumerate(rows):
   s=scores[:,j];take=s>best;second=np.where(take,best,np.maximum(second,s));best=np.where(take,s,best);winner=np.where(take,row,winner)
 accepted=[]
 for i in range(count):
  if best[i]>=cfg['similarity_threshold'] and (cfg['similarity_margin']==0 or (np.isfinite(second[i]) and best[i]-second[i]>=cfg['similarity_margin'])):accepted.append((i,int(winner[i]),float(best[i])))
 raw=len(accepted);used=set();result=[]
 for item in sorted(accepted,key=lambda x:(-x[2],x[0])):
  pid=int(store.point_ids[item[1]])
  if pid not in used:result.append(item);used.add(pid)
 return result[:cfg['max_matches']],raw

class Pipeline:
 def __init__(self,store,global_backend,local_backend,cfg,K,dist=None):
  self.store=store;self.g=global_backend;self.l=local_backend;self.c=cfg;self.K=np.asarray(K,np.float64);self.dist=np.asarray(dist if dist is not None else np.zeros(5),np.float64);self.refs=None;self.active_rows=None;self.last_retrieval=-100000
  for key in ['process_every','retrieval_every','top_k','max_active_images','max_active_points','max_keypoints','max_matches','match_block','retrieval_block']:
   if cfg[key]<1:raise ValueError(key)
  if self.K.shape!=(3,3) or not np.isfinite(self.K).all() or self.K[0,0]<=0 or self.K[1,1]<=0:raise ValueError('camera intrinsics')
 def frame(self,bgr,index,timestamp):
  start=time.perf_counter();r={'frame':index,'timestamp_s':timestamp,'timing_platform':'PC CPU','status':'skipped','timings_ms':{},'top_k':[],'active_points':0,'query_keypoints':0,'raw_matches':0,'correspondences':0,'pnp_inliers':0}
  def timed(name,fn):
   t=time.perf_counter()
   try:return fn()
   finally:r['timings_ms'][name]=(time.perf_counter()-t)*1000
  try:
   if index%self.c['process_every']:return r
   if bgr is None or bgr.ndim!=3 or bgr.shape[2]!=3:raise ValueError('expected original BGR frame HxWx3')
   h,w=bgr.shape[:2]
   if [w,h]!=self.c['camera_size']:raise ValueError('camera calibration resolution mismatch')
   if self.refs is None or index-self.last_retrieval>=self.c['retrieval_every']:
    rgb=timed('global_preprocess',lambda:np.asarray(Image.fromarray(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)).resize((240,240),Image.Resampling.BICUBIC)))
    raw=timed('mixvpr',lambda:self.g.infer(rgb));q=timed('cpu_l2',lambda:normalize(np.asarray(raw).reshape(-1)))
    if q.shape!=(512,):raise ValueError('global descriptor must be 512D')
    self.refs=timed('retrieval',lambda:self.store.retrieve(q,self.c['top_k'],self.c['retrieval_block']))
    self.active_rows=timed('active_map',lambda:self.store.active(self.refs,self.c['max_active_images'],self.c['max_active_points']));self.last_retrieval=index
    r['retrieval_reused']=False
   else:r['retrieval_reused']=True
   r['top_k']=[int(self.store.image_ids[i]) for i in self.refs];r['active_points']=len(self.active_rows)
   if hasattr(self.l,'infer_frame'):
    sw,sh=w,h
    xy,desc=timed('superpoint',lambda:self.l.infer_frame(bgr))
   else:
    sw,sh=self.c['superpoint_size']
    gray=timed('local_preprocess',lambda:cv2.resize(cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY),(sw,sh),interpolation=cv2.INTER_AREA))
    xy,desc=timed('superpoint',lambda:self.l.infer(gray))
   xy=np.asarray(xy,np.float32);desc=np.asarray(desc,np.float32)
   if xy.ndim!=2 or xy.shape[1]!=2 or desc.shape!=(len(xy),256) or not np.isfinite(xy).all():raise ValueError('local feature shape/values')
   if np.any(xy<0) or np.any(xy[:,0]>=sw) or np.any(xy[:,1]>=sh):raise ValueError('keypoints outside local input')
   xy=xy[:self.c['max_keypoints']];desc=desc[:len(xy)];r['query_keypoints']=len(xy)
   xy=(xy+.5)*np.array([w/sw,h/sh],np.float32)-.5
   matches,raw=timed('matching',lambda:match(desc,self.store,self.active_rows,self.c));r['raw_matches']=raw;r['correspondences']=len(matches)
   if len(matches)<self.c['min_inliers']:r['status']='insufficient_matches';return r
   points=np.asarray([self.store.xyz[p] for i,p,s in matches],np.float64);pixels=np.asarray([xy[i] for i,p,s in matches],np.float64)
   if not np.isfinite(points).all():raise ValueError('nonfinite XYZ')
   ok,rv,tv,inliers=timed('pnp',lambda:cv2.solvePnPRansac(points,pixels,self.K,self.dist,iterationsCount=100,reprojectionError=self.c['pnp_error_px'],confidence=.99,flags=cv2.SOLVEPNP_EPNP))
   n=0 if inliers is None else len(inliers);r['pnp_inliers']=n
   if not ok or n<self.c['min_inliers']:r['status']='pnp_failed';return r
   R=cv2.Rodrigues(rv)[0];depth=(points[inliers.ravel()]@R.T+tv.reshape(3))[:,2]
   if not np.isfinite(tv).all() or np.any(depth<=0):r['status']='invalid_pose';return r
   r.update(status='pose',world_to_camera_R=R.tolist(),world_to_camera_t=tv.reshape(3).tolist(),camera_center_world=(-R.T@tv).reshape(3).tolist())
  except (ValueError,RuntimeError,cv2.error,IndexError) as ex:r.update(status='error',error=str(ex))
  finally:r['timings_ms']['total']=(time.perf_counter()-start)*1000
  return r

def frames(source):
 cap=cv2.VideoCapture(source)
 if not cap.isOpened():raise RuntimeError('cannot open camera/video')
 index=0;origin=time.monotonic()
 try:
  while True:
   ok,bgr=cap.read()
   if not ok:break
   timestamp=time.monotonic()-origin if isinstance(source,int) else cap.get(cv2.CAP_PROP_POS_MSEC)/1000
   yield index,timestamp,bgr;index+=1
 finally:cap.release()
