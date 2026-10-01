"""Frozen Pitts30k-test benchmark: official 512D vs D4 FP32/nncase.
No optimizer, calibration, model selection or benchmark-driven preprocessing.
"""
from __future__ import annotations
import os
os.environ.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
from pathlib import Path
import sys,json,hashlib,importlib.util,ast,argparse,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];DATA=ROOT/'datasets/pitts30k_test';OUT=ROOT/'mixvpr/artifacts/pitts30k_test';VENDOR=ROOT/'mixvpr/vendor/MixVPR'
FREEZE=ROOT/'mixvpr/artifacts/d4_final_candidate/freeze_manifest.json'

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def put(p,value):
 p=Path(p);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(p)
def module(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def split():
 # Execute the unmodified official parser definition without its hardcoded
 # dataset-root import-time check; benchmark data resides on D:.
 from scipy.io import loadmat
 from collections import namedtuple
 source=VENDOR/'dataloaders/PittsburgDataset.py';tree=ast.parse(source.read_text())
 nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='parse_dbStruct' or isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='dbStruct' for t in n.targets)]
 ns={'loadmat':loadmat,'namedtuple':namedtuple};exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source),'exec'),ns)
 s=ns['parse_dbStruct'](str(DATA/'pitts30k_test.mat'))
 assert s.whichSet=='test' and (s.numDb,s.numQ)==(10000,6816) and s.posDistThr==25
 from sklearn.neighbors import NearestNeighbors
 gt=NearestNeighbors(n_jobs=2).fit(s.utmDb).radius_neighbors(s.utmQ,radius=s.posDistThr,return_distance=False)
 names=s.dbImage+['queries_real/'+p for p in s.qImage]
 return s,names,gt

def model(which):
 import torch
 import torch.nn as nn
 if which=='D4':
  sys.path.insert(0,str(ROOT/'experiments/edge_vpr_students'))
  from s512_br1_sq240_m100_c160h96_model import S512BR1SQ240M100C160H96
  m=S512BR1SQ240M100C160H96();ck=json.loads(FREEZE.read_text())['deployment_checkpoint']
  m.load_state_dict(torch.load(ck,map_location='cpu',weights_only=False)['model_state_dict'],strict=True)
 else:
  ResNet=module('official_resnet',VENDOR/'models/backbones/resnet.py').ResNet
  MixVPR=module('official_aggregator',VENDOR/'models/aggregators/mixvpr.py').MixVPR
  class Official(nn.Module):
   def __init__(self):
    super().__init__();self.backbone=ResNet('resnet50',pretrained=False,layers_to_crop=[4]);self.aggregator=MixVPR(1024,20,20,256,4,1,2)
   def forward(self,x):return self.aggregator(self.backbone(x))
  m=Official();m.load_state_dict(torch.load(OUT/'official_mixvpr512.ckpt',map_location='cpu',weights_only=True),strict=True)
 return m.eval()

def image(path,which):
 from PIL import Image
 with Image.open(path) as im:
  im=im.convert('RGB');im=im.resize((320,320) if which=='official' else (240,240),Image.Resampling.BILINEAR if which=='official' else Image.Resampling.BICUBIC)
  x=np.array(im,dtype=np.float32).transpose(2,0,1)/np.float32(255)
 return np.ascontiguousarray((x-np.array([.485,.456,.406],np.float32)[:,None,None])/np.array([.229,.224,.225],np.float32)[:,None,None])

def recalls(r,q,gt):
 import faiss
 faiss.omp_set_num_threads(2);index=faiss.IndexFlatL2(r.shape[1]);index.add(np.ascontiguousarray(r,dtype=np.float32));dist,pred=index.search(np.ascontiguousarray(q,dtype=np.float32),10)
 hit=np.array([[bool(np.isin(p[:k],g).any()) for k in (1,5,10)] for p,g in zip(pred,gt)])
 return {'R@'+str(k):float(hit[:,j].mean()) for j,k in enumerate((1,5,10))},hit,pred

def verify_frozen():
 f=json.loads(FREEZE.read_text());assert all(sha(p)==h for p,h in f['files'].items());return f

def selftest():
 import torch
 torch.set_num_threads(2);s,names,gt=split();rng=np.random.default_rng(42)
 r=rng.normal(size=(20,512)).astype('float32');q=r[:3].copy();g=[np.array([0]),np.array([1]),np.array([],dtype=int)]
 own,hit,_=recalls(r,q,g)
 official=module('official_validation',VENDOR/'utils/validation.py').get_validation_recalls(r,q,[1,5,10],g,print_results=False)
 assert all(own['R@'+str(k)]==official[k]==2/3 for k in (1,5,10))
 from sklearn.neighbors import NearestNeighbors
 assert np.array_equal(NearestNeighbors().fit([[0.,0.]]).radius_neighbors([[25.,0.]],radius=25,return_distance=False)[0],[0])
 # Validate both transforms against their exact reference implementations on
 # synthetic pixels, not benchmark images.
 from PIL import Image
 import torchvision.transforms as T
 im=Image.fromarray(rng.integers(0,256,(47,61,3),dtype='uint8'))
 for which,size,interp in [('D4',240,T.InterpolationMode.BICUBIC),('official',320,T.InterpolationMode.BILINEAR)]:
  a=T.Compose([T.Resize((size,size),interpolation=interp),T.ToTensor(),T.Normalize([.485,.456,.406],[.229,.224,.225])])(im).numpy()
  x=np.array(im.resize((size,size),Image.Resampling.BICUBIC if which=='D4' else Image.Resampling.BILINEAR),np.float32).transpose(2,0,1)/np.float32(255)
  b=(x-np.array([.485,.456,.406],np.float32)[:,None,None])/np.array([.229,.224,.225],np.float32)[:,None,None]
  assert np.array_equal(a,b)
 profiles={}
 for which,size in [('D4',240),('official',320)]:
  m=model(which);mac=[0]
  def hook(l,args,out):mac[0]+=out.numel()*(l.in_channels//l.groups*l.kernel_size[0]*l.kernel_size[1] if isinstance(l,torch.nn.Conv2d) else l.in_features)
  handles=[l.register_forward_hook(hook) for l in m.modules() if isinstance(l,(torch.nn.Conv2d,torch.nn.Linear))]
  with torch.inference_mode():y=m(torch.zeros(1,3,size,size))
  for h in handles:h.remove()
  assert y.shape==(1,512) and torch.isfinite(y).all()
  profiles[which]={'params':sum(p.numel() for p in m.parameters()),'Conv_Linear_MACs':mac[0],'input':[size,size],'descriptor':512}
 import subprocess
 dev=ROOT/'mixvpr/artifacts/depth_pilot'
 x=np.array(np.load(dev/'dev_inputs.npy',mmap_mode='r')[:1])
 worker=subprocess.run(['/home/quyet/miniconda3/envs/k210/bin/python',str(Path(__file__).resolve()),'worker'],input=x.tobytes(),stdout=subprocess.PIPE,check=True)
 actual=np.frombuffer(worker.stdout,np.float32);expected=np.load(dev/'D4/int8_dev/0000.npy')
 assert np.array_equal(actual,expected),'nncase worker must reproduce existing faithful DEV artifact exactly'
 verify_frozen();put(OUT/'selftest.json',{'status':'PASS','runner_sha256':sha(__file__),'faithful_worker_DEV_parity':'EXACT','official_recall_parity':True,'preprocessing_exact_parity':True,'GT_boundary_inclusive':True,'numDb':s.numDb,'numQ':s.numQ,'queries_without_positive':sum(len(g)==0 for g in gt),'profiles':profiles});print(profiles)

def lock():
 f=verify_frozen();s,names,gt=split();d=json.loads((DATA/'download_manifest.json').read_text())
 assert [x['relative'] for x in d['images']]==names
 assert all(sha(DATA/'images'/x['relative'])==x['sha256'] for x in d['images'])
 assert json.loads((OUT/'selftest.json').read_text())['status']=='PASS'
 assert json.loads((OUT/'selftest.json').read_text())['runner_sha256']==sha(__file__)
 files=[FREEZE,DATA/'download_manifest.json',DATA/'pitts30k_test.mat',OUT/'official_mixvpr512.ckpt',Path(__file__),ROOT/'mixvpr/setup_pitts30k.py',ROOT/'mixvpr/run_pitts30k_benchmark.py']+list(VENDOR.glob('models/**/*.py'))+[VENDOR/'dataloaders/PittsburgDataset.py',VENDOR/'utils/validation.py']
 from importlib.metadata import version
 versions={n:version(n) for n in ['torch','torchvision','numpy','scipy','scikit-learn','pillow','faiss-cpu']}
 p={'versions':versions,'nncase':'1.8.0.20220929/runtime1.8.0-55be52f','benchmark':'Pitts30k-test','numDb':s.numDb,'numQ':s.numQ,'positive_radius_m':25,'retrieval':'Official CPU FAISS IndexFlatL2 exact, top10, full query denominator','R':[1,5,10],'files':{str(p):sha(p) for p in files},'models':['official','D4','INT8'],'preprocess':{'official':'RGB320 PIL bilinear/ImageNet, official convention','D4':'Frozen RGB240 PIL bicubic/ImageNet','INT8':'Exactly D4 input; frozen existing kmodel + CPU L2 eps1e-12'},'kmodel':f['kmodel'],'kmodel_sha256':f['kmodel_sha256'],'no_training_tuning_or_calibration':True,'claim':'Descriptive standard outdoor benchmark, not application-specific indoor acceptance; different training distributions and inputs disclosed. No post-hoc pass gate or checkpoint selection.'}
 if (OUT/'protocol.json').exists():assert json.loads((OUT/'protocol.json').read_text())==p
 else:put(OUT/'protocol.json',p)
 put(OUT/'GT.json',[g.tolist() for g in gt]);print('PROTOCOL_LOCKED',sha(OUT/'protocol.json'))

def infer(which,shard,shards):
 assert 0<=shard<shards
 p=json.loads((OUT/'protocol.json').read_text());assert all(sha(f)==h for f,h in p['files'].items());verify_frozen()
 s,names,gt=split();indices=list(range(shard,len(names),shards));folder=OUT/which;folder.mkdir(exist_ok=True)
 meta={'protocol_sha256':sha(OUT/'protocol.json'),'which':which,'shard':shard,'shards':shards,'indices':indices}
 mp=folder/f'shard{shard}.json'
 if mp.exists():assert json.loads(mp.read_text())==meta
 else:put(mp,meta)
 if which=='INT8':
  import subprocess
  worker=subprocess.Popen(['/home/quyet/miniconda3/envs/k210/bin/python',str(Path(__file__).resolve()),'worker'],stdin=subprocess.PIPE,stdout=subprocess.PIPE)
 else:
  import torch
  torch.set_num_threads(2);assert torch.cuda.is_available();torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
  m=model(which).cuda()
 for block in range(0,len(indices),64):
  ids=indices[block:block+64];dest=folder/f'shard{shard}_block{block:05d}.npy'
  if dest.exists():
   a=np.load(dest);assert a.shape==(len(ids),512) and np.isfinite(a).all();continue
  values=[]
  if which=='INT8':
   for i in ids:
    worker.stdin.write(image(DATA/'images'/names[i],'D4').tobytes());worker.stdin.flush()
    raw=worker.stdout.read(512*4);assert len(raw)==512*4,'nncase worker failed'
    v=np.frombuffer(raw,np.float32);assert np.isfinite(v).all() and np.linalg.norm(v)>1e-12;values.append(v.copy())
  else:
   with torch.inference_mode():
    for j in range(0,len(ids),4):
     x=np.stack([image(DATA/'images'/names[i],which) for i in ids[j:j+4]])
     if which=='D4':
      v=m.raw(torch.from_numpy(x).cuda()).float().cpu();v=torch.nn.functional.normalize(v,dim=1,eps=1e-12)
     else:v=m(torch.from_numpy(x).cuda()).float().cpu()
     values.extend(v.numpy())
  a=np.array(values,np.float32);assert a.shape==(len(ids),512) and np.isfinite(a).all()
  tmp=dest.with_suffix('.tmp')
  with tmp.open('wb') as f:np.save(f,a)
  tmp.replace(dest);print(which,shard,min(block+64,len(indices)),'/',len(indices),flush=True)
 if which=='INT8':
  worker.stdin.close();assert worker.wait(timeout=30)==0
 put(folder/f'complete{shard}.json',{'meta':meta,'hashes':{x.name:sha(x) for x in folder.glob(f'shard{shard}_block*.npy')}})

def nncase_worker():
 import nncase,_nncase
 from importlib.metadata import version
 assert version('nncase')=='1.8.0.20220929' and _nncase.__version__=='1.8.0-55be52f'
 p=json.loads(FREEZE.read_text());assert sha(p['kmodel'])==p['kmodel_sha256']
 sim=nncase.Simulator();sim.load_model(Path(p['kmodel']).read_bytes())
 assert sim.get_input_tensor(0).dtype==np.dtype('float32') and sim.get_output_tensor(0).dtype==np.dtype('float32')
 while True:
  raw=sys.stdin.buffer.read(3*240*240*4)
  if not raw:break
  assert len(raw)==3*240*240*4
  x=np.frombuffer(raw,np.float32).reshape(1,3,240,240).copy()
  sim.set_input_tensor(0,nncase.RuntimeTensor.from_numpy(x));sim.run()
  v=sim.get_output_tensor(0).to_numpy().astype(np.float32).reshape(512)
  assert np.isfinite(v).all() and np.linalg.norm(v)>1e-12
  v=v/max(np.linalg.norm(v),1e-12);sys.stdout.buffer.write(v.tobytes());sys.stdout.buffer.flush()

def score():
 verify_frozen();s,names,gt=split();p=json.loads((OUT/'protocol.json').read_text());assert all(sha(f)==h for f,h in p['files'].items());results={}
 for which in p['models']:
  folder=OUT/which;values=np.empty((len(names),512),np.float32);seen=set();completions=list(folder.glob('complete*.json'));assert completions
  for cf in completions:
   c=json.loads(cf.read_text());meta=c['meta'];assert meta['protocol_sha256']==sha(OUT/'protocol.json');assert len(completions)==meta['shards'];ids=meta['indices']
   for block in range(0,len(ids),64):
    name=f"shard{meta['shard']}_block{block:05d}.npy";assert sha(folder/name)==c['hashes'][name];ix=ids[block:block+64];assert not seen.intersection(ix);seen.update(ix);values[ix]=np.load(folder/name)
  assert len(seen)==len(names) and np.isfinite(values).all()
  metric,hit,pred=recalls(values[:s.numDb],values[s.numDb:],gt);np.savez(OUT/f'{which}_predictions.npz',predictions=pred,hits=hit)
  results[which]={'recalls':metric,'correct_counts':hit.sum(0).tolist(),'queries':s.numQ,'descriptor':512}
 put(OUT/'results.json',{'status':'BENCHMARK_COMPLETE','protocol_sha256':sha(OUT/'protocol.json'),'results':results,'profiles':json.loads((OUT/'selftest.json').read_text())['profiles'],'D4_resource_evidence':json.loads((ROOT/'mixvpr/artifacts/depth_pilot/D4/compile.json').read_text()),'checkpoint_changed':False});print(json.dumps(results))

if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('stage',choices=['selftest','lock','infer','score','worker']);a.add_argument('--model',choices=['D4','official','INT8']);a.add_argument('--shard',type=int,default=0);a.add_argument('--shards',type=int,default=1);args=a.parse_args()
 if args.stage=='worker':nncase_worker()
 elif args.stage=='selftest':selftest()
 elif args.stage=='lock':lock()
 elif args.stage=='infer':infer(args.model,args.shard,args.shards)
 else:score()
