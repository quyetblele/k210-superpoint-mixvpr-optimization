"""Download original Pitts30k-test images only; verify official archive checksums."""
from pathlib import Path
import hashlib,json,tarfile,subprocess,concurrent.futures,threading,time
NETWORK=threading.Semaphore(4)
from scipy.io import loadmat
ROOT=Path(__file__).resolve().parents[1];DATA=ROOT/'datasets/pitts30k_test'
URL='https://data.ciirc.cvut.cz/public/projects/2015netVLAD/Pittsburgh250k/'
def digest(p,algo='sha256'):
 h=hashlib.new(algo)
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def fetch(name):
 target=DATA/name
 if name.endswith('.tar') and target.exists() and target.stat().st_size:
  return ranged_resume(name,target)
 with (DATA/(name+'.download.log')).open('a') as log:
  subprocess.run(['curl','-fL','--retry','5','--connect-timeout','20','-C','-',URL+name,'-o',str(target)],stdout=log,stderr=log,check=True)
 return target
def ranged_resume(name,target):
 import requests
 start=target.stat().st_size
 with requests.get(URL+name,headers={'Range':'bytes=0-0','Accept-Encoding':'identity'},timeout=(20,60),stream=True) as r:
  assert r.status_code==206 and r.headers['Content-Range'].startswith('bytes 0-0/')
  total=int(r.headers['Content-Range'].split('/')[-1])
 if start==total:return target
 assert start<total
 size=(total-start+3)//4
 ranges=[(a,min(a+size,total)-1) for a in range(start,total,size)]
 def part(bounds):
  a,b=bounds;dest=DATA/(name+f'.range_{a}_{b}')
  if dest.exists() and dest.stat().st_size==b-a+1:return dest
  with NETWORK:
   for attempt in range(4):
    try:
     offset=dest.stat().st_size if dest.exists() else 0
     if offset==b-a+1:return dest
     assert offset<b-a+1
     begin=a+offset
     with requests.get(URL+name,headers={'Range':f'bytes={begin}-{b}','Accept-Encoding':'identity'},timeout=(20,60),stream=True) as response:
      assert response.status_code==206 and response.headers['Content-Range']==f'bytes {begin}-{b}/{total}'
      with dest.open('ab') as f:
       for chunk in response.iter_content(1<<20):f.write(chunk)
     assert dest.stat().st_size==b-a+1
     print('RANGE_COMPLETE',name,a,b,flush=True);return dest
    except Exception:
     if attempt==3:raise
     time.sleep(2)
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:parts=list(pool.map(part,ranges))
 assert target.stat().st_size==start
 with target.open('ab') as f:
  for partfile in parts:
   with partfile.open('rb') as src:
    for chunk in iter(lambda:src.read(8<<20),b''):f.write(chunk)
 assert target.stat().st_size==total
 for partfile in parts:partfile.unlink()
 return target
if __name__=='__main__':
 DATA.mkdir(parents=True,exist_ok=True)
 if not (DATA/'netvlad_v100_datasets.tar.gz').exists():fetch('netvlad_v100_datasets.tar.gz')
 with tarfile.open(DATA/'netvlad_v100_datasets.tar.gz') as t:
  m=[m for m in t if m.name.endswith('/pitts30k_test.mat')];assert len(m)==1
  raw=t.extractfile(m[0]).read();p=DATA/'pitts30k_test.mat'
  if p.exists():assert p.read_bytes()==raw
  else:p.write_bytes(raw)
 specs=loadmat(p)['dbStruct'].item();db=[f[0].item() for f in specs[1]];q=['queries_real/'+f[0].item() for f in specs[3]]
 wanted=set(db+q);assert len(wanted)==16816
 checks=DATA/'md5sums.txt'
 if not checks.exists():fetch('md5sums.txt')
 md5={line.split()[1].lstrip('*./'):line.split()[0] for line in checks.read_text().splitlines() if len(line.split())==2}
 names=[k+'.tar' for k in sorted({n.split('/')[0] for n in wanted})]
 def one(name):
  archive=DATA/name
  if not archive.exists() or digest(archive,'md5')!=md5[name]:fetch(name)
  assert digest(archive,'md5')==md5[name],name
  count=0
  with tarfile.open(archive) as t:
   for m in t:
    name_in=m.name.removeprefix('./')
    if name_in not in wanted:continue
    assert m.isfile();dest=DATA/'images'/name_in;dest.parent.mkdir(parents=True,exist_ok=True)
    content=t.extractfile(m).read()
    if dest.exists():assert dest.read_bytes()==content
    else:dest.write_bytes(content)
    count+=1
  result={'archive':name,'url':URL+name,'md5':md5[name],'sha256':digest(archive),'images_extracted':count}
  (DATA/(name+'.verified.json')).write_text(json.dumps(result,indent=2)+'\n');print('VERIFIED',name,count,flush=True);return result
 with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:results=list(pool.map(one,names))
 assert all((DATA/'images'/n).exists() for n in wanted)
 manifest={'source':URL,'split_sha256':digest(p),'numDb':len(db),'numQ':len(q),'radius_m':float(specs[7].item()),'archives':results,'images':[{'relative':n,'sha256':digest(DATA/'images'/n)} for n in db+q]}
 (DATA/'download_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print('DATASET_READY',len(wanted),flush=True)
