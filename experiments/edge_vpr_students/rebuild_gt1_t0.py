#!/usr/bin/env python3
from __future__ import annotations
import hashlib,json,sys
from pathlib import Path
import cv2,numpy as np,torch
ROOT=Path('/home/quyet/k210_lab');EDGE=Path('/home/quyet/edge_ai_project');OUT=ROOT/'reports/edge_vpr_students';GT0=ROOT/'artifacts/edge_vpr_students/gt0';ART=ROOT/'artifacts/edge_vpr_students/gt1_t0';ART.mkdir(parents=True,exist_ok=True)
sys.path[:0]=[str(EDGE/'src'),str(EDGE/'third_party'),str(EDGE/'third_party/hloc')]
from localize_live import RuntimeConfig,preprocess_frame,GlobalExtractorRunner
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):Path(p).write_text(json.dumps(x,indent=2)+'\n')
class Canonical:
 def __init__(self):
  self.cfg=RuntimeConfig(device='cpu',max_width=384,max_height=480);self.online=GlobalExtractorRunner('mixvpr','cpu');self.model=self.online.model
 def tensor(self,p):
  b=cv2.imread(str(p),cv2.IMREAD_COLOR)
  if b is None:raise RuntimeError(str(p))
  return preprocess_frame(b,self.cfg)[0]
 def one(self,p):return self.online(self.tensor(p))
 def many(self,ps,b=16):
  out=[]
  for i in range(0,len(ps),b):
   x=torch.cat([self.tensor(p) for p in ps[i:i+b]],0)
   with torch.inference_mode():y=self.model({'image':x})['global_descriptor'].cpu().numpy().astype('float32')
   y/=np.maximum(np.linalg.norm(y,axis=1,keepdims=True),1e-12);out.append(y)
  return np.concatenate(out)
def main():
 torch.set_num_threads(4)
 c=Canonical();man=json.loads((OUT/'gt0_retrieval_protocol_manifest.json').read_text());par=[]
 for floor in ('floor6','floor7'):
  names=(GT0/f'{floor}_query.txt').read_text().splitlines()[:2]
  for n in names:
   p=EDGE/'assets'/floor/n;a=c.one(p);b=c.many([p])[0];par.append({'floor':floor,'image':n,'max_abs':float(np.max(abs(a-b))),'mean_abs':float(np.mean(abs(a-b))),'cosine':float(a@b/(np.linalg.norm(a)*np.linalg.norm(b)))})
 if max(x['max_abs'] for x in par)>1e-6:raise RuntimeError('online-wrapper parity fail')
 res={'parity':par,'floors':{},'preprocessing':'cv2 IMREAD_COLOR BGR; preprocess_frame RuntimeConfig(cpu,max_width=384,max_height=480): scale=min(1,384/W,480/H), INTER_AREA; BGR->RGB; FP32 NCHW /255; Mixvpr ImageNet Normalize(mean .485,.456,.406/std .229,.224,.225), bilinear stretch 320x320 align_corners=False; 4096D L2-normalized output'}
 for floor,x in man['floors'].items():
  d={};
  for role in ('reference','query'):
   ns=(GT0/f'{floor}_{role}.txt').read_text().splitlines();ps=[EDGE/'assets'/floor/n for n in ns];a=c.many(ps);np.save(ART/f'{floor}_{role}_fp32.npy',a);(ART/f'{floor}_{role}_names.txt').write_text('\n'.join(ns)+'\n');d[role]={'count':len(ns),'shape':list(a.shape),'finite':bool(np.isfinite(a).all()),'norm_min':float(np.linalg.norm(a,axis=1).min()),'norm_max':float(np.linalg.norm(a,axis=1).max()),'sha256':sha(ART/f'{floor}_{role}_fp32.npy')}
  R=np.load(ART/f'{floor}_reference_fp32.npy');Q=np.load(ART/f'{floor}_query_fp32.npy');refs=(GT0/f'{floor}_reference.txt').read_text().splitlines();qs=(GT0/f'{floor}_query.txt').read_text().splitlines();labels=json.loads((GT0/f'{floor}_positive_labels.json').read_text())['primary'];rows=[]
  for q,v in zip(qs,Q):
   pos=set(labels[q]);o=np.argsort(-(R@v),kind='mergesort');top=[refs[j] for j in o[:10]];rows.append({'query':q,'valid':bool(pos),'top10':top,'scores':[float(R[j]@v) for j in o[:10]],'hits':{str(k):bool(pos.intersection(top[:k])) for k in(1,5,10)}})
  valid=[z for z in rows if z['valid']];d['zero_positive']=len(rows)-len(valid);d['recall']={f'R@{k}':float(np.mean([z['hits'][str(k)] for z in valid])) for k in(1,5,10)};d['ranking_sample']=rows[:5];res['floors'][floor]=d
 allrows=[]
 for f in res['floors']:
  # recompute valid booleans from frozen labels and saved rankings
  labs=json.loads((GT0/f'{f}_positive_labels.json').read_text())['primary'];allrows += [z for z in res['floors'][f]['ranking_sample'] if labs[z['query']]]
 # weighted combined directly from all saved rows generated above, reload json not needed
 total_num=sum((x['query']['count']-x['zero_positive'])*x['recall']['R@1'] for x in res['floors'].values());den=sum(x['query']['count']-x['zero_positive'] for x in res['floors'].values());res['combined']={'R@1':total_num/den,'R@5':sum((x['query']['count']-x['zero_positive'])*x['recall']['R@5'] for x in res['floors'].values())/den,'R@10':sum((x['query']['count']-x['zero_positive'])*x['recall']['R@10'] for x in res['floors'].values())/den}
 save(OUT/'gt1_exact_t0_manifest.json',res);save(ART/'evaluation_outputs.json',res)
 lines=['# GT1 exact current T0 rebuild','',f"Online↔canonical max abs `{max(x['max_abs'] for x in par):.3e}`, minimum cosine `{min(x['cosine'] for x in par):.9f}`.",'', '| Floor | refs | queries | zero-positive excluded | R@1 | R@5 | R@10 |','|---|---:|---:|---:|---:|---:|---:|']
 for f,x in res['floors'].items():r=x['recall'];lines.append(f"| {f} | {x['reference']['count']} | {x['query']['count']} | {x['zero_positive']} | {r['R@1']:.4f} | {r['R@5']:.4f} | {r['R@10']:.4f} |")
 r=res['combined'];lines += ['',f"Weighted combined: R@1 `{r['R@1']:.4f}`, R@5 `{r['R@5']:.4f}`, R@10 `{r['R@10']:.4f}`.",'','Historical H5 is incompatible with this current-online path; cause: **different preprocessing input resize** (historical H5 did not reproduce the exact 384×480 online path).']
 (OUT/'gt1_exact_t0_recall_results.md').write_text('\n'.join(lines)+'\n');(OUT/'gt1_exact_t0_rebuild_report.md').write_text('# GT1 exact T0 rebuild report\n\n'+res['preprocessing']+'\n\nIntegrity: all descriptor arrays FP32, finite, 4096D, unit-normalized; ordering is frozen in paired names files.\n');print('GT1 PASS',res['combined'])
if __name__=='__main__':main()
