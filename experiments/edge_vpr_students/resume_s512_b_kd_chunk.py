#!/usr/bin/env python3
"""V2 chunk cursor: current_epoch, next_batch_index is first unprocessed."""
from __future__ import annotations
import argparse,json,sys,torch
from pathlib import Path
from pytorch_metric_learning import losses,miners
ROOT=Path('/home/quyet/k210_lab');EXP=ROOT/'experiments/edge_vpr_students';OUT=ROOT/'reports/edge_vpr_students';ART=ROOT/'artifacts/edge_vpr_students/s512_training';sys.path.insert(0,str(EXP))
import train_s512_b_kd as z
from s512_checkpoint_v2 import save,restore_rng_state
def main():
 p=argparse.ArgumentParser();p.add_argument('epoch',type=int);p.add_argument('start',type=int);a=p.parse_args();assert a.start in range(0,48,4);torch.set_num_threads(4)
 from s512_engineering import S,build
 rows,pos,valid,_=z.build_split();ts=z.teacher_cache(rows);m=build(S['S512-B']);o=torch.optim.AdamW(m.parameters(),lr=3e-4,weight_decay=1e-4);sc=torch.optim.lr_scheduler.CosineAnnealingLR(o,T_max=4,eta_min=3e-5)
 src=ART/('s512_b_epoch1_valid_v2.pt' if a.epoch==2 and a.start==0 else f's512_b_epoch{a.epoch}_next{a.start}.pt');c=torch.load(src,weights_only=False);assert c['schema_version']==2 and (c['current_epoch'],c['next_batch_index'])==(a.epoch,a.start);m.load_state_dict(c['model_state_dict']);o.load_state_dict(c['optimizer_state_dict']);sc.load_state_dict(c['scheduler_state_dict']);restore_rng_state(c['rng_state']);plan=c['epoch_plan'];tf=losses.MultiSimilarityLoss(alpha=1.,beta=50.,base=0.);mi=miners.MultiSimilarityMiner(epsilon=.1);lines=[]
 for b in range(a.start,a.start+4):
  e=plan[b];ids=e['row_indices'];lab=torch.tensor(e['labels']);x=torch.stack([z.student_tensor(rows[i]) for i in ids]);t=torch.from_numpy(ts[ids]);lr=o.param_groups[0]['lr'];y=m(x);task=tf(y,lab,mi(y,lab));kd=torch.nn.functional.mse_loss(y@y.T,t@t.T);tot=task+2*kd;o.zero_grad();tot.backward();o.step();lines.append({'current_epoch':a.epoch,'batch_index':b,'global_step':c['global_step']+b-a.start+1,'pair_row_indices':ids,'sample_ids':[f'{rows[i][0]}/{rows[i][1]}' for i in ids],'task_loss':float(task.detach()),'kd_loss':float(kd.detach()),'total_loss':float(tot.detach()),'learning_rate':lr})
 n=a.start+4
 # Epoch completion is after batch 47: one scheduler step, then persist a new plan.
 if n==48:
  sc.step();ne=a.epoch+1;np=z.epoch_plan(ne,pos,valid);dst=ART/f's512_b_epoch{ne}_next0.pt';ce,nb,ep=ne,0,np
 else: dst=ART/f's512_b_epoch{a.epoch}_next{n}.pt';ce,nb,ep=a.epoch,n,plan
 save(dst,model=m,optimizer=o,scheduler=sc,current_epoch=ce,next_batch_index=nb,global_step=c['global_step']+4,training_config=c['training_config'],epoch_plan=ep,best_metric_name=c['best_metric_name'],best_metric_value=c['best_metric_value'],best_checkpoint_path=c['best_checkpoint_path'])
 with (OUT/'s512_b_training_trace.jsonl').open('a') as f:
  for x in lines:f.write(json.dumps(x)+'\n')
 print(json.dumps({'batch_range':[a.start,n-1],'global_steps':[c['global_step']+1,c['global_step']+4],'checkpoint':str(dst),'trace':lines},indent=2))
if __name__=='__main__':main()
