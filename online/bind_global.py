"""Bind final checkpoint to references without copying local map arrays."""
import argparse,json,os
from pathlib import Path
import numpy as np
from PIL import Image
from backends import TorchStudent,OnnxStudent
from pipeline import normalize

def main():
 p=argparse.ArgumentParser();p.add_argument('--map',type=Path,required=True);p.add_argument('--images',type=Path,required=True);p.add_argument('--model',required=True);p.add_argument('--backend',choices=['fp32','onnx'],default='fp32');p.add_argument('--local-proof',type=Path,required=True,help='Reviewed provenance JSON, not inferred automatically');a=p.parse_args()
 meta=json.loads((a.map/'map.json').read_text());proof=json.loads(a.local_proof.read_text())
 if proof.get('source_index_sha256')!=meta['source_index_sha256'] or not proof.get('query_local_identity') or not proof.get('evidence'):raise ValueError('missing matching local-backend provenance evidence')
 if (a.map/'global.npy').exists():raise ValueError('global binding already exists; refusing overwrite')
 backend=(TorchStudent if a.backend=='fp32' else OnnxStudent)(a.model)
 refs=json.loads((a.map/'reference_images.json').read_text());paths=[]
 for r in refs:
  f=(a.images/r['name']).resolve()
  if not f.is_relative_to(a.images.resolve()) or not f.is_file():raise ValueError('missing/unsafe reference path: '+str(f))
  paths.append(f)
 tmp=a.map/'global.pending.npy';db=np.lib.format.open_memmap(tmp,mode='w+',dtype=np.float32,shape=(len(refs),512))
 for i,f in enumerate(paths):
  with Image.open(f) as im:rgb=np.asarray(im.convert('RGB').resize((240,240),Image.Resampling.BICUBIC))
  db[i]=normalize(backend.infer(rgb).reshape(512))
 db.flush();del db
 os.replace(tmp,a.map/'global.npy')
 meta.update(status='BOUND_PENDING_REAL_QUERY_VALIDATION',global_identity=backend.identity,local_identity=proof['query_local_identity'],local_proof=proof,pending=['real development-query integration validation','board parity and resources','independent quality evaluation'])
 t=a.map/'map.pending.json';t.write_text(json.dumps(meta,indent=2)+'\n');os.replace(t,a.map/'map.json')
 print('BOUND',a.map)
if __name__=='__main__':main()
