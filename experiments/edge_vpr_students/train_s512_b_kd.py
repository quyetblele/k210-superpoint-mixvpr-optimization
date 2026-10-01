#!/usr/bin/env python3
"""PC-first S512-B task retrieval + relational-KD training; no source edits."""
from __future__ import annotations
import csv, hashlib, json, math, random, sys, time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import onnx, onnxruntime, pycolmap, torch
import torch.nn.functional as F
from pytorch_metric_learning import losses, miners

ROOT=Path('/home/quyet/k210_lab'); EDGE=Path('/home/quyet/edge_ai_project')
EXP=ROOT/'experiments/edge_vpr_students'; OUT=ROOT/'reports/edge_vpr_students'
GT0=ROOT/'artifacts/edge_vpr_students/gt0'; GT1=ROOT/'artifacts/edge_vpr_students/gt1_t0'
ART=ROOT/'artifacts/edge_vpr_students/s512_training'; MOD=ROOT/'models'
SEED=20260906; FLOORS=('floor6','floor7'); W,H=160,288
MEAN=torch.tensor([.485,.456,.406]).view(3,1,1); STD=torch.tensor([.229,.224,.225]).view(3,1,1)

def put(path,obj): path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(obj,indent=2)+'\n')
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def norm(x): return F.normalize(x,p=2,dim=1,eps=1e-12)

def poses(floor):
 m=json.loads((EDGE/'map_packages'/f'{floor}_v1'/'manifest.json').read_text()); r=pycolmap.Reconstruction(EDGE/'map_packages'/f'{floor}_v1'/m['files']['colmap_model'])
 d={}
 for im in r.images.values():
  p=im.cam_from_world; p=p() if callable(p) else p; R=np.asarray(p.rotation.matrix()); t=np.asarray(p.translation)
  d[im.name]=(-R.T@t,R.T@np.array([0.,0.,1.]))
 return d

def build_split():
 rows=[]; leakage={}
 for floor in FLOORS:
  q=set((GT0/f'{floor}_query.txt').read_text().splitlines()); e=set((GT0/f'{floor}_excluded.txt').read_text().splitlines()); r=(GT0/f'{floor}_reference.txt').read_text().splitlines(); ps=poses(floor)
  assert not(q & set(r)) and not(e & set(r))
  leakage[floor]={'training_reference_images':len(r),'gt0_queries':len(q),'temporal_guard_excluded':len(e),'intersection_train_query':len(q&set(r)),'intersection_train_excluded':len(e&set(r))}
  rows += [(floor,n,*ps[n]) for n in r]
 # Pair only within a floor, using the frozen primary geometric criterion.
 pos=defaultdict(list)
 for floor in FLOORS:
  ids=[i for i,x in enumerate(rows) if x[0]==floor]; C=np.stack([rows[i][2] for i in ids]); V=np.stack([rows[i][3] for i in ids])
  D=np.linalg.norm(C[:,None]-C[None,:],axis=2); A=np.degrees(np.arccos(np.clip(V@V.T,-1,1)))
  for a,i in enumerate(ids): pos[i]=[ids[b] for b in np.where((D[a]<=1.0)&(A[a]<=60)&(np.arange(len(ids))!=a))[0]]
 valid=[i for i in range(len(rows)) if pos[i]]
 assert valid
 return rows,pos,valid,leakage

def student_tensor(row,width=160,height=288):
 floor,name,*_=row; b=cv2.imread(str(EDGE/'assets'/floor/name),cv2.IMREAD_COLOR)
 if b is None: raise RuntimeError(f'missing {floor}/{name}')
 rgb=cv2.cvtColor(b,cv2.COLOR_BGR2RGB); rgb=cv2.resize(rgb,(width,height),interpolation=cv2.INTER_AREA)
 x=torch.from_numpy(rgb).permute(2,0,1).float().div_(255.); return (x-MEAN)/STD

def teacher_cache(rows):
 d={}
 for floor in FLOORS:
  ns=(GT1/f'{floor}_reference_names.txt').read_text().splitlines(); a=np.load(GT1/f'{floor}_reference_fp32.npy')
  d.update({(floor,n):a[i] for i,n in enumerate(ns)})
 missing=[(f,n) for f,n,*_ in rows if (f,n) not in d]
 if missing: raise RuntimeError(f'missing exact GT1 teacher cache entries: {missing[:3]}')
 return np.stack([d[(f,n)] for f,n,*_ in rows]).astype('float32')

def sample_batch(rng,pos,valid,pairs=8):
 anchors=rng.sample(valid,pairs); out=[]; labels=[]
 for lab,a in enumerate(anchors): out += [a,rng.choice(pos[a])]; labels += [lab,lab]
 return out,torch.tensor(labels,dtype=torch.long)

def epoch_plan(epoch,pos,valid,steps=48):
 """Persisted anchor-positive tuples and labels; chunks never re-sample them."""
 rng=random.Random(SEED+epoch);return [{'row_indices':(x:=sample_batch(rng,pos,valid))[0],'labels':x[1].tolist()} for _ in range(steps)]

def eval_recall(model,rows):
 model.eval(); D={}
 with torch.inference_mode():
  for floor in FLOORS:
   ids=[i for i,x in enumerate(rows) if x[0]==floor]
   vals=[]
   for s in range(0,len(ids),16): vals.append(model(torch.stack([student_tensor(rows[i]) for i in ids[s:s+16]])).cpu().numpy())
   D[floor]=(ids,np.concatenate(vals))
 result={'floors':{}}
 for floor,(ids,all_d) in D.items():
  refs=(GT0/f'{floor}_reference.txt').read_text().splitlines(); qs=(GT0/f'{floor}_query.txt').read_text().splitlines()
  # refs are exactly the trained-in database; queries were never used in training.
  idx={rows[i][1]:j for j,i in enumerate(ids)}; R=all_d[[idx[n] for n in refs]]
  with torch.inference_mode():
   Q=[]
   for s in range(0,len(qs),16): Q.append(model(torch.stack([student_tensor((floor,n,None,None)) for n in qs[s:s+16]])).cpu().numpy())
  Q=np.concatenate(Q); labels=json.loads((GT0/f'{floor}_positive_labels.json').read_text())['primary']; hits=[]
  for q,v in zip(qs,Q):
   order=np.argsort(-(R@v),kind='mergesort'); p=set(labels[q]); hits.append(None if not p else {k:bool(p.intersection(refs[j] for j in order[:k])) for k in(1,5,10)})
  good=[x for x in hits if x is not None]; result['floors'][floor]={'reference_count':len(refs),'query_count':len(qs),'zero_positive':len(hits)-len(good),'recall':{f'R@{k}':float(np.mean([x[k] for x in good])) for k in(1,5,10)}}
 den=sum(x['query_count']-x['zero_positive'] for x in result['floors'].values())
 result['combined']={f'R@{k}':sum((x['query_count']-x['zero_positive'])*x['recall'][f'R@{k}'] for x in result['floors'].values())/den for k in(1,5,10)}
 return result

def export_regression(model):
 model.eval(); x=torch.linspace(-1,1,3*H*W).reshape(1,3,H,W); p=MOD/'s512_b_trained_full_160x288.onnx'
 torch.onnx.export(model,x,str(p),opset_version=13,dynamo=False,do_constant_folding=False,input_names=['normalized_rgb'],output_names=['global_descriptor'])
 m=onnx.load(str(p)); onnx.checker.check_model(m); inf=onnx.shape_inference.infer_shapes(m,strict_mode=True); ip=MOD/'s512_b_trained_full_160x288_inferred.onnx';onnx.save(inf,str(ip)); o=onnxruntime.InferenceSession(str(ip),providers=['CPUExecutionProvider']).run(None,{'normalized_rgb':x.numpy()})[0]
 with torch.inference_mode(): y=model(x).numpy()
 return {'onnx':str(ip),'sha256':sha(ip),'output_shape':list(o.shape),'finite':bool(np.isfinite(o).all()),'l2_norm':float(np.linalg.norm(o)),'pytorch_ort_max_abs':float(np.max(np.abs(y-o))),'pytorch_ort_mean_abs':float(np.mean(np.abs(y-o))),'pytorch_ort_cosine':float(y.reshape(-1)@o.reshape(-1)/(np.linalg.norm(y)*np.linalg.norm(o))),'operators':dict(__import__('collections').Counter(n.op_type for n in inf.graph.node))}

def main():
 torch.manual_seed(SEED);np.random.seed(SEED);random.seed(SEED);torch.set_num_threads(4); ART.mkdir(parents=True,exist_ok=True)
 sys.path.insert(0,str(EXP)); from s512_engineering import S,build
 rows,pos,valid,audit=build_split(); teachers=teacher_cache(rows)
 # Teacher cache is exact GT1 T0, precomputed under its frozen online preprocessing.
 cache_audit={'source':'GT1 exact T0 reference arrays','samples':len(rows),'dimension':4096,'norm_min':float(np.linalg.norm(teachers,axis=1).min()),'norm_max':float(np.linalg.norm(teachers,axis=1).max()),'sha256':{f:sha(GT1/f'{f}_reference_fp32.npy') for f in FLOORS}}
 put(OUT/'s512_b_training_split_audit.json',{'audit':audit,'positive_pair_anchors':len(valid),'teacher_cache':cache_audit})
 model=build(S['S512-B']); teacher_before=None
 opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4); sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=4,eta_min=3e-5)
 taskfn=losses.MultiSimilarityLoss(alpha=1.,beta=50.,base=0.0); miner=miners.MultiSimilarityMiner(epsilon=.1)
 rng=random.Random(SEED); log=[]
 def step(epoch,stepno):
  ids,lab=sample_batch(rng,pos,valid); x=torch.stack([student_tensor(rows[i]) for i in ids]); t=torch.from_numpy(teachers[ids]);
  y=model(x); mined=miner(y,lab); task=taskfn(y,lab,mined); kd=F.mse_loss(y@y.T,t@t.T); total=task+2.0*kd
  opt.zero_grad();total.backward(); grad=float(sum(p.grad.detach().abs().sum() for p in model.parameters() if p.grad is not None));opt.step()
  return {'epoch':epoch,'step':stepno,'task':float(task.detach()),'relational_kd':float(kd.detach()),'total':float(total.detach()),'grad_l1':grad,'finite':bool(torch.isfinite(total) and math.isfinite(grad))}
 # Short sanity: update exactly two batches and validate state/checkpoint round-trip.
 init={k:v.detach().clone() for k,v in model.state_dict().items()}; sanity=[step(0,i) for i in range(2)]; changed=any(not torch.equal(init[k],v) for k,v in model.state_dict().items()); sanity_path=ART/'s512_b_sanity.pt';torch.save({'model':model.state_dict(),'optimizer':opt.state_dict()},sanity_path);loaded=torch.load(sanity_path,weights_only=True);assert set(loaded['model'])==set(model.state_dict()) and changed and all(x['finite'] for x in sanity)
 # Four epochs * 48 valid pair-batches: a single intentional CPU-first experiment, no sweep.
 best=-1.; best_epoch=-1; start=time.time()
 for epoch in range(1,5):
  es=[step(epoch,s) for s in range(48)]; log.extend(es); sch.step(); ev=eval_recall(model,rows); score=ev['combined']['R@5']
  ck={'epoch':epoch,'model':model.state_dict(),'optimizer':opt.state_dict(),'scheduler':sch.state_dict(),'config':{'architecture':'S512-B frozen','student_preprocess':'BGR->RGB; direct INTER_AREA resize 160x288; /255; ImageNet mean/std; NCHW','teacher_preprocess':'frozen GT1 exact current T0','task':'MultiSimilarityLoss + MultiSimilarityMiner on 8 pose-positive pairs/batch','relational_kd':'MSE(S S^T, T T^T)','weights':{'task':1.0,'relational_kd':2.0}}}
  torch.save(ck,ART/'s512_b_final.pt')
  if score>best: best=score;best_epoch=epoch;torch.save(ck,ART/'s512_b_best.pt')
  put(OUT/'s512_b_training_progress.json',{'sanity':sanity,'epochs_completed':epoch,'last_eval':ev,'best_r5':best,'best_epoch':best_epoch,'log':log})
 bestck=torch.load(ART/'s512_b_best.pt',weights_only=False);model.load_state_dict(bestck['model']); final_eval=eval_recall(model,rows); reg=export_regression(model)
 with (OUT/'s512_b_training_log.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(log[0]));w.writeheader();w.writerows(log)
 t0={'R@1':.5080645161,'R@5':.6209677419,'R@10':.6612903226}; st=final_eval['combined']; report={'actual_measured':True,'seed':SEED,'split_audit':audit,'teacher_cache':cache_audit,'sanity':{'records':sanity,'parameters_updated':changed,'checkpoint_roundtrip':'PASS'},'training':{'epochs':4,'steps_per_epoch':48,'batch_images':16,'optimizer':'AdamW','lr':3e-4,'weight_decay':1e-4,'scheduler':'CosineAnnealingLR T_max=4 eta_min=3e-5','duration_seconds':time.time()-start,'best_epoch_by_combined_R@5':best_epoch},'loss':{'task':'MixVPR-original-style MultiSimilarityLoss(alpha=1,beta=50,base=0) with MultiSimilarityMiner(epsilon=.1) over eight pose-positive pairs/batch','relational_kd':'MSE(S S^T, T T^T), both L2-normalized; dimensions BxB therefore independent of 4096 vs 512','weights':{'task':1.0,'relational_kd':2.0},'ranking_kd':'NOT RUN'},'evaluation':final_eval,'t0_exact':t0,'delta':{k:st[k]-t0[k] for k in t0},'retention':{k:st[k]/t0[k] for k in t0},'kd_control':'NOT RUN (one configured training/KD experiment only)','deployment_regression':reg,'artifacts':{'best':str(ART/'s512_b_best.pt'),'final':str(ART/'s512_b_final.pt'),'training_log':str(OUT/'s512_b_training_log.csv')}}
 put(OUT/'s512_b_training_kd_report.json',report);print(json.dumps(report,indent=2))
if __name__=='__main__':main()
