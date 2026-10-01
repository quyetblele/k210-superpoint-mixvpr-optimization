from __future__ import annotations
import random
from pathlib import Path
import numpy as np, torch
def capture_rng_state(): return {'python':random.getstate(),'numpy':np.random.get_state(),'torch_cpu':torch.get_rng_state(),'torch_cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}
def restore_rng_state(s):
 random.setstate(s['python']);np.random.set_state(s['numpy']);torch.set_rng_state(s['torch_cpu'])
 if torch.cuda.is_available() and s.get('torch_cuda') is not None: torch.cuda.set_rng_state_all(s['torch_cuda'])
def save(path, **k):
 m,o,sch=k.pop('model'),k.pop('optimizer'),k.pop('scheduler');k.update(schema_version=2,model_state_dict=m.state_dict(),optimizer_state_dict=o.state_dict(),scheduler_state_dict=sch.state_dict() if sch else None,rng_state=capture_rng_state());p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');torch.save(k,t);t.replace(p)
