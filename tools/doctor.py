"""Read-only environment audit, no installs or environment mutation."""
import json,subprocess,sys,shutil
from pathlib import Path
root=Path(__file__).resolve().parents[1]
code="""import json,sys,importlib.metadata as m
names=['torch','numpy','opencv-python','onnx','onnxruntime','onnxruntime-gpu','nncase','matplotlib','Pillow']
v={}
for n in names:
 try:v[n]=m.version(n)
 except m.PackageNotFoundError:pass
print(json.dumps({'python':sys.version,'executable':sys.executable,'packages':v}))
"""
result={}
for name,python in [('training',root/'.venv-mixvpr-cuda/bin/python'),('nncase',Path('/home/quyet/miniconda3/envs/k210/bin/python'))]:
    result[name]=json.loads(subprocess.check_output([str(python),'-c',code],text=True))
result['workspace_free_GiB']=shutil.disk_usage(root).free/1024**3
result['serial_by_id']=[str(x) for x in Path('/dev/serial/by-id').glob('*')]
output=root/'docs/audit/environment.json';output.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
