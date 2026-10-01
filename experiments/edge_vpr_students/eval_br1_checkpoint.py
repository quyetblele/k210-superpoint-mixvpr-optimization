import sys,torch,numpy as np,json,argparse
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('epoch');a=p.parse_args();R=Path('/home/quyet/k210_lab');sys.path.insert(0,str(R/'experiments/edge_vpr_students'));import train_s512_b_kd as z
from s512_br1_engineering import BR1
A=R/'artifacts/edge_vpr_students/s512_br1_official';G=R/'artifacts/edge_vpr_students/gt0';c=torch.load(A/f'epoch{int(a.epoch)+1}_next0.pt',weights_only=False);m=BR1().eval();m.load_state_dict(c['model_state_dict']);out={}
for f in ['floor6','floor7']:
 d={}
 for role in ['reference','query']:
  ns=(G/f'{f}_{role}.txt').read_text().splitlines();ys=[]
  with torch.no_grad():
   for i in range(0,len(ns),32):ys.append(m(torch.stack([z.student_tensor((f,n,None,None),192,352) for n in ns[i:i+32]])).numpy())
  d[role]=(ns,np.concatenate(ys))
 refs,X=d['reference'];qs,Q=d['query'];lab=json.load(open(G/f'{f}_positive_labels.json'))['primary'];h=[]
 for q,v in zip(qs,Q):
  p0=set(lab[q]);o=np.argsort(-(X@v),kind='mergesort');h.append(None if not p0 else {k:bool(p0.intersection(refs[j] for j in o[:k])) for k in [1,5,10]})
 q=[x for x in h if x];out[f]={f'R@{k}':float(np.mean([x[k] for x in q])) for k in [1,5,10]}
out['combined']={f'R@{k}':((60*out['floor6'][f'R@{k}'])+(64*out['floor7'][f'R@{k}']))/124 for k in [1,5,10]};print(json.dumps(out));(R/f'reports/edge_vpr_students/s512_br1_epoch{a.epoch}_gt1.json').write_text(json.dumps(out,indent=2)+'\n')
