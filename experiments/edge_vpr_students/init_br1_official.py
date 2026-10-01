import sys,torch,random
from pathlib import Path
R=Path('/home/quyet/k210_lab');sys.path.insert(0,str(R/'experiments/edge_vpr_students'));import train_s512_b_kd as z
from s512_br1_engineering import BR1
from s512_checkpoint_v2 import save
torch.manual_seed(z.SEED);random.seed(z.SEED);rows,pos,valid,_=z.build_split();m=BR1();o=torch.optim.AdamW(m.parameters(),lr=3e-4,weight_decay=1e-4);s=torch.optim.lr_scheduler.CosineAnnealingLR(o,T_max=4,eta_min=3e-5);A=R/'artifacts/edge_vpr_students/s512_br1_official';save(A/'epoch1_next0.pt',model=m,optimizer=o,scheduler=s,current_epoch=1,next_batch_index=0,global_step=0,training_config={'student':'S512-BR1','input':[3,352,192],'task':'MultiSimilarity+relational KD','weights':{'task':1.,'kd':2.}},epoch_plan=z.epoch_plan(1,pos,valid),best_metric_name='combined_R@5',best_metric_value=float('-inf'),best_checkpoint_path=None)
