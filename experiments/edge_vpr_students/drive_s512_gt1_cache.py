"""Time-budgeted wrapper: one evaluator process per missing chunk."""
import subprocess,sys,time
from pathlib import Path
R=Path('/home/quyet/k210_lab');P=R/'experiments/edge_vpr_students/eval_s512_gt1_chunks.py'
t=time.monotonic();budget=22
for tag in ('best','final'):
 for floor,role in [('floor6','reference'),('floor6','query'),('floor7','reference'),('floor7','query')]:
  while time.monotonic()-t<budget:
   r=subprocess.run([sys.executable,str(P),tag,floor,role],cwd=R,capture_output=True,text=True)
   print(r.stdout.strip());
   if r.returncode or 'COMPLETE' in r.stdout:break
