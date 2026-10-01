#!/usr/bin/env python3
"""Run exactly one resumed S512-B KD epoch; used to make the CPU job recoverable."""
from __future__ import annotations
import argparse,csv,json,random,sys,time
from pathlib import Path
import torch
from pytorch_metric_learning import losses,miners
ROOT=Path('/home/quyet/k210_lab');EXP=ROOT/'experiments/edge_vpr_students';OUT=ROOT/'reports/edge_vpr_students';ART=ROOT/'artifacts/edge_vpr_students/s512_training'
sys.path.insert(0,str(EXP));import train_s512_b_kd as z
def main():
 ap=argparse.ArgumentParser();ap.add_argument('epoch',type=int,choices=(2,3,4));a=ap.parse_args();torch.set_num_threads(4);torch.manual_seed(z.SEED);random.seed(z.SEED)
 from s512_engineering import S,build
 rows,pos,valid,audit=z.build_split();teachers=z.teacher_cache(rows);m=build(S['S512-B']);o=torch.optim.AdamW(m.parameters(),lr=3e-4,weight_decay=1e-4);sch=torch.optim.lr_scheduler.CosineAnnealingLR(o,T_max=4,eta_min=3e-5)
 ck=torch.load(ART/'s512_b_final.pt',weights_only=False); assert ck['epoch']==a.epoch-1,(ck['epoch'],a.epoch);m.load_state_dict(ck['model']);o.load_state_dict(ck['optimizer']);sch.load_state_dict(ck['scheduler']);tf=losses.MultiSimilarityLoss(alpha=1.,beta=50.,base=0.);mi=miners.MultiSimilarityMiner(epsilon=.1);rng=random.Random(z.SEED+a.epoch);log=[];t0=time.time()
 for s in range(48):
  ids,lab=z.sample_batch(rng,pos,valid);x=torch.stack([z.student_tensor(rows[i]) for i in ids]);t=torch.from_numpy(teachers[ids]);y=m(x);task=tf(y,lab,mi(y,lab));kd=torch.nn.functional.mse_loss(y@y.T,t@t.T);tot=task+2*kd;o.zero_grad();tot.backward();g=float(sum(p.grad.detach().abs().sum() for p in m.parameters() if p.grad is not None));o.step();log.append({'epoch':a.epoch,'step':s,'task':float(task.detach()),'relational_kd':float(kd.detach()),'total':float(tot.detach()),'grad_l1':g,'finite':bool(torch.isfinite(tot))})
 sch.step();ev=z.eval_recall(m,rows);new={'epoch':a.epoch,'model':m.state_dict(),'optimizer':o.state_dict(),'scheduler':sch.state_dict(),'config':ck['config']};torch.save(new,ART/'s512_b_final.pt');p=json.loads((OUT/'s512_b_training_progress.json').read_text());p['epochs_completed']=a.epoch;p['last_eval']=ev;p['log'].extend(log);p['epoch_duration_seconds']=time.time()-t0
 if ev['combined']['R@5']>p['best_r5']:p['best_r5']=ev['combined']['R@5'];p['best_epoch']=a.epoch;torch.save(new,ART/'s512_b_best.pt')
 z.put(OUT/'s512_b_training_progress.json',p);print(json.dumps({'epoch':a.epoch,'eval':ev,'best':p['best_epoch']},indent=2))
if __name__=='__main__':main()
