import os
os.environ.update(OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
import sys,json,time
from pathlib import Path
import numpy as np,torch,cv2
ROOT=Path(__file__).parent;sys.path.insert(0,str(ROOT.parent));sys.path.insert(0,'/home/quyet/k210_lab/scripts')
from backends import SuperPointCPU,identity
from superpoint_kpu_wrapper import SuperPointKPUWrapper
from superpoint_onnx import SuperPointOnnx
sp=SuperPointCPU([360,640],160);torch.set_num_threads(2);path=ROOT/'superpoint_fp32_360x640.onnx';wrapper=SuperPointKPUWrapper(sp.model).eval()
torch.onnx.export(wrapper,torch.zeros(1,1,640,360),str(path),opset_version=13,dynamo=False,input_names=['gray'],output_names=['detector_logits','descriptor_map'])
Path(str(path)+'.json').write_text(json.dumps({'size':[360,640],'onnx_identity':identity(path,'onnx'),'local_identity':sp.identity,'precision':'FP32'},indent=2)+'\n')
backend=SuperPointOnnx(path,[360,640],160,2);names=json.load(open('/home/quyet/k210_lab/online/maps/stair_local_v1/reference_images.json'));ids=set(json.load(open(ROOT/'protocol.json'))['tuning_ids'][:5]);files=[Path('/home/quyet/edge_ai_project/assets/stair_6_7')/r['name'] for r in names if r['image_id'] in ids];timings={'torch1':[],'torch2':[],'onnx2':[]};parity=[]
for p in files:
 gray=cv2.resize(cv2.imread(str(p),cv2.IMREAD_GRAYSCALE),(360,640),interpolation=cv2.INTER_AREA)
 torch.set_num_threads(2);a,d=sp.infer(gray);b,e=backend.infer(gray)
 # Keypoint sets and matched descriptors; ordering may change at score ties.
 common={tuple(x):j for j,x in enumerate(b)};pairs=[(i,common[tuple(x)]) for i,x in enumerate(a) if tuple(x) in common]
 assert len(pairs)>=.99*len(a)
 cosine=np.array([d[i]@e[j] for i,j in pairs]);assert cosine.min()>.9999
 parity.append({'shared_keypoints':len(pairs),'total':len(a),'min_descriptor_cosine':float(cosine.min())})
 for name,be,threads in [('torch1',sp,1),('torch2',sp,2),('onnx2',backend,2)]:
  torch.set_num_threads(threads);be.infer(gray)
  for _ in range(2):
   start=time.perf_counter();be.infer(gray);timings[name].append((time.perf_counter()-start)*1000)
report={'scope':'CPU laptop FP32 local stage only; five tuning images, warm-up plus two repeats each; no K210 timing claim','parity':parity,'median_ms':{k:float(np.median(v)) for k,v in timings.items()},'timings_ms':timings}
(ROOT/'backend_benchmark.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report['median_ms']))
