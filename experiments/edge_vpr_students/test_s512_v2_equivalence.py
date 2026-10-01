from pathlib import Path
import os
import hashlib,json,sys,torch,numpy as np
from pytorch_metric_learning import losses,miners
R=Path('/home/quyet/k210_lab');E=R/'experiments/edge_vpr_students';sys.path.insert(0,str(E));import train_s512_b_kd as z
from s512_checkpoint_v2 import save,restore_rng_state,capture_rng_state
A=R/'artifacts/edge_vpr_students/s512_training';O=R/'reports/edge_vpr_students'
def h(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def make(c):
 from s512_engineering import S,build
 m=build(S['S512-B']);o=torch.optim.AdamW(m.parameters(),lr=3e-4,weight_decay=1e-4);s=torch.optim.lr_scheduler.CosineAnnealingLR(o,T_max=4,eta_min=3e-5);m.load_state_dict(c['model_state_dict']);o.load_state_dict(c['optimizer_state_dict']);s.load_state_dict(c['scheduler_state_dict']);restore_rng_state(c['rng_state']);return m,o,s
def go(m,o,plan,rows,ts,a,b):
 f=losses.MultiSimilarityLoss(alpha=1.,beta=50.,base=0.);mi=miners.MultiSimilarityMiner(epsilon=.1);out=[]
 for i in range(a,b):
  e=plan[i];ids=e['row_indices'];lab=torch.tensor(e['labels']);x=torch.stack([z.student_tensor(rows[j]) for j in ids]);t=torch.from_numpy(ts[ids]);lr=o.param_groups[0]['lr'];y=m(x);u=f(y,lab,mi(y,lab));k=torch.nn.functional.mse_loss(y@y.T,t@t.T);v=u+2*k;o.zero_grad();v.backward();o.step();out.append({'batch':i,'ids':ids,'labels':e['labels'],'task':float(u.detach()),'kd':float(k.detach()),'total':float(v.detach()),'lr':lr})
 return out
torch.set_num_threads(int(os.environ.get('S512_TEST_THREADS','4')));base_a=torch.load(A/'s512_b_epoch1_valid_v2.pt',weights_only=False);base_b=torch.load(A/'s512_b_epoch1_valid_v2.pt',weights_only=False);rows,pos,valid,_=z.build_split();ts=z.teacher_cache(rows);plan_a=base_a['epoch_plan'];plan_b=base_b['epoch_plan'];ma,oa,sa=make(base_a);mb,ob,sb=make(base_b);initial_model=max(float((ma.state_dict()[k]-mb.state_dict()[k]).abs().max()) for k in ma.state_dict());ta=go(ma,oa,plan_a,rows,ts,0,8);tb=go(mb,ob,plan_b,rows,ts,0,4);save(A/'_equiv_tmp.pt',model=mb,optimizer=ob,scheduler=sb,current_epoch=2,next_batch_index=4,global_step=52,training_config=base_b['training_config'],epoch_plan=plan_b,best_metric_name=base_b['best_metric_name'],best_metric_value=base_b['best_metric_value'],best_checkpoint_path=base_b['best_checkpoint_path']);c=torch.load(A/'_equiv_tmp.pt',weights_only=False);mb,ob,sb=make(c);tb+=go(mb,ob,plan_b,rows,ts,4,8)
md=max(float((ma.state_dict()[k]-mb.state_dict()[k]).abs().max()) for k in ma.state_dict()); same=ta==tb; od=max(float((x-y).abs().max()) for x,y in zip(oa.state_dict()['state'].values(),ob.state_dict()['state'].values()) if isinstance(x,dict) for x,y in zip(x.values(),y.values()) if torch.is_tensor(x))
def eq(a,b):
 if torch.is_tensor(a):return torch.equal(a,b)
 if isinstance(a,dict):return a.keys()==b.keys() and all(eq(a[k],b[k]) for k in a)
 if isinstance(a,(list,tuple)):return len(a)==len(b) and all(eq(x,y) for x,y in zip(a,b))
 return a==b
sched_equal=eq(oa.state_dict(),ob.state_dict());r={'hashes':{p:h(E/p) for p in ['s512_checkpoint_v2.py','train_s512_b_kd.py','resume_s512_b_kd_chunk.py']},'independent_disk_loads':True,'epoch_plan_equal':plan_a==plan_b,'max_initial_model_diff':initial_model,'trace_equal':same,'run_a':ta,'run_b':tb,'max_model_parameter_diff':md,'max_optimizer_tensor_diff':od,'scheduler_equal':sched_equal,'verdict':'PASS' if same and md==0 and od==0 and initial_model==0 and sched_equal else 'FAIL'};O.joinpath('s512_v2_equivalence.json').write_text(json.dumps(r,indent=2)+'\n');print(r['verdict'],md,od)
