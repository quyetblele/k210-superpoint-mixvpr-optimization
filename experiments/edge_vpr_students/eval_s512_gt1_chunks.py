"""One atomic GT1 descriptor chunk per invocation; no training/evaluation-rule edits."""
from __future__ import annotations
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np,torch
R=Path('/home/quyet/k210_lab');E=R/'experiments/edge_vpr_students';sys.path.insert(0,str(E));import train_s512_b_kd as z
A=R/'artifacts/edge_vpr_students/s512_training';C=A/'gt1_eval';G=R/'artifacts/edge_vpr_students/gt0'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('tag',choices=['best','final']);p.add_argument('floor',choices=['floor6','floor7']);p.add_argument('role',choices=['reference','query']);a=p.parse_args();torch.set_num_threads(4)
 ck=A/('s512_b_best.pt' if a.tag=='best' else 's512_b_epoch5_next0.pt');x=torch.load(ck,weights_only=False);from s512_engineering import S,build;m=build(S['S512-B']);m.load_state_dict(x['model_state_dict'] if 'model_state_dict'in x else x['model']);m.eval();names=(G/f'{a.floor}_{a.role}.txt').read_text().splitlines();d=C/a.tag/f'{a.floor}_{a.role}s';d.mkdir(parents=True,exist_ok=True);man=d/'manifest.json';meta={'checkpoint':str(ck),'sha256':sha(ck),'names':names,'shape':[len(names),512],'preprocess':'BGR->RGB; INTER_AREA direct 160x288; /255; ImageNet mean/std; NCHW','chunk_size':32};
 if not man.exists():man.write_text(json.dumps(meta,indent=2)+'\n')
 for ci,s in enumerate(range(0,len(names),32)):
  f=d/f'chunk_{ci:03d}.npz';e=min(s+32,len(names))
  if f.exists():
   q=np.load(f,allow_pickle=False);ok=q['desc'].shape==(e-s,512) and q['ids'].tolist()==names[s:e] and np.isfinite(q['desc']).all() and np.allclose(np.linalg.norm(q['desc'],axis=1),1,atol=1e-4)
   if ok:continue
  rows=[(a.floor,n,None,None) for n in names[s:e]]
  with torch.no_grad():v=m(torch.stack([z.student_tensor(r) for r in rows])).numpy().astype('float32')
  assert v.shape==(e-s,512) and np.isfinite(v).all() and np.allclose(np.linalg.norm(v,axis=1),1,atol=1e-4)
  t=Path(str(f)+'.tmp');np.savez_compressed(t,ids=np.array(names[s:e]),desc=v,checkpoint_sha256=meta['sha256'],start=s,end=e);Path(str(t)+'.npz').replace(f);print(a.tag,a.floor,a.role,ci,s,e);return
 print('COMPLETE',a.tag,a.floor,a.role)
if __name__=='__main__':main()
