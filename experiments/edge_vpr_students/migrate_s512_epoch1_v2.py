from pathlib import Path
import sys,torch,json
R=Path('/home/quyet/k210_lab');sys.path.insert(0,str(R/'experiments/edge_vpr_students'))
import train_s512_b_kd as z
from s512_engineering import S,build
from s512_checkpoint_v2 import save
A=R/'artifacts/edge_vpr_students/s512_training';old=torch.load(A/'s512_b_final.pt',weights_only=False);rows,pos,valid,_=z.build_split();m=build(S['S512-B']);m.load_state_dict(old['model']);o=torch.optim.AdamW(m.parameters(),lr=3e-4,weight_decay=1e-4);o.load_state_dict(old['optimizer']);s=torch.optim.lr_scheduler.CosineAnnealingLR(o,T_max=4,eta_min=3e-5);s.load_state_dict(old['scheduler']);save(A/'s512_b_epoch1_valid_v2.pt',model=m,optimizer=o,scheduler=s,current_epoch=2,next_batch_index=0,global_step=48,training_config=old['config'],epoch_plan=z.epoch_plan(2,pos,valid),best_metric_name='combined_R@5',best_metric_value=float(json.load(open(R/'reports/edge_vpr_students/s512_b_training_progress.json'))['best_r5']),best_checkpoint_path=str(A/'s512_b_best.pt'))
(A/'s512_b_epoch2_step4.INVALID_FOR_FINAL_REPORT.txt').write_text('INVALID_FOR_FINAL_REPORT: ambiguous schema-v1 partial epoch2; never resume.\n')
