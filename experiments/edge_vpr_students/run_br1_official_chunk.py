import sys,torch,json
from pathlib import Path
from pytorch_metric_learning import losses,miners
R=Path('/home/quyet/k210_lab');E=R/'experiments/edge_vpr_students';sys.path.insert(0,str(E));import train_s512_b_kd as z
from s512_br1_engineering import BR1
from s512_checkpoint_v2 import save,restore_rng_state
A=R/'artifacts/edge_vpr_students/s512_br1_official';c=torch.load(A/'epoch1_next0.pt',weights_only=False);assert(c['current_epoch'],c['next_batch_index'],c['global_step'])==(1,0,0);rows,pos,valid,_=z.build_split();t=z.teacher_cache(rows);m=BR1();o=torch.optim.AdamW(m.parameters(),lr=3e-4,weight_decay=1e-4);s=torch.optim.lr_scheduler.CosineAnnealingLR(o,T_max=4,eta_min=3e-5);m.load_state_dict(c['model_state_dict']);o.load_state_dict(c['optimizer_state_dict']);s.load_state_dict(c['scheduler_state_dict']);restore_rng_state(c['rng_state']);f=losses.MultiSimilarityLoss(alpha=1.,beta=50.,base=0.);mi=miners.MultiSimilarityMiner(epsilon=.1);tr=[]
for b in range(4):
 e=c['epoch_plan'][b];ids=e['row_indices'];lab=torch.tensor(e['labels']);x=torch.stack([z.student_tensor(rows[i],192,352) for i in ids]);y=m(x);tt=torch.from_numpy(t[ids]);u=f(y,lab,mi(y,lab));k=torch.nn.functional.mse_loss(y@y.T,tt@tt.T);v=u+2*k;o.zero_grad();v.backward();o.step();tr.append({'epoch':1,'batch_index':b,'global_step':b+1,'row_indices':ids,'labels':e['labels'],'image_ids':[f'{rows[i][0]}/{rows[i][1]}' for i in ids],'task_loss':float(u),'kd_loss':float(k),'total_loss':float(v),'lr':o.param_groups[0]['lr'],'finite':bool(torch.isfinite(v))})
save(A/'epoch1_next4.pt',model=m,optimizer=o,scheduler=s,current_epoch=1,next_batch_index=4,global_step=4,training_config=c['training_config'],epoch_plan=c['epoch_plan'],best_metric_name=c['best_metric_name'],best_metric_value=c['best_metric_value'],best_checkpoint_path=c['best_checkpoint_path']);open(A/'official_trace.jsonl','a').write(''.join(json.dumps(x)+'\n' for x in tr));print(json.dumps(tr))
