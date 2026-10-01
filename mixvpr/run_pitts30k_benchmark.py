"""Reproducible bounded-memory benchmark orchestration; all outputs on D:."""
from pathlib import Path
import subprocess,sys,json,time
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'mixvpr/artifacts/pitts30k_test';PY=ROOT/'.venv-mixvpr-cuda/bin/python';EVAL=ROOT/'mixvpr/pitts30k_benchmark.py'
def run(args):subprocess.run([str(PY),str(EVAL),*args],cwd=ROOT,check=True)
if __name__=='__main__':
 assert OUT.resolve().is_relative_to(Path('/mnt/d')),'Benchmark outputs must stay on D:'
 if '--wait-data' in sys.argv:
  deadline=time.monotonic()+7200
  while not (ROOT/'datasets/pitts30k_test/download_manifest.json').exists():
   assert time.monotonic()<deadline,'Dataset download did not complete within2h'
   time.sleep(10)
 run(['selftest']);run(['lock'])
 jobs=[]
 # Two CPU simulator workers and sequential GPU models bound RAM/CPU use.
 for shard in range(2):
  log=(OUT/f'INT8_{shard}.log').open('a');jobs.append((subprocess.Popen([str(PY),str(EVAL),'infer','--model','INT8','--shard',str(shard),'--shards','2'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT),log))
 try:
  for model in ['official','D4']:
   with (OUT/f'{model}.log').open('a') as log:subprocess.run([str(PY),str(EVAL),'infer','--model',model],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
  for process,log in jobs:assert process.wait()==0,'INT8 worker failed'
 finally:
  for process,log in jobs:
   if process.poll() is None:process.terminate();process.wait()
   log.close()
 run(['score'])
