import os
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
import sys,json,time
from pathlib import Path
import numpy as np,cv2
ROOT=Path(__file__).parent;sys.path.insert(0,str(ROOT.parent));sys.path.insert(0,'/home/quyet/edge_ai_project/third_party/hloc')
from pipeline import Pipeline,MapStore
from backends import TorchStudent,SuperPointCPU
from hloc.utils.read_write_model import read_images_binary,read_cameras_binary,qvec2rotmat
src=Path('/home/quyet/edge_ai_project/map_packages/stair_6_7_v1');frozen=Path('/home/quyet/training_recovery/final_gate2');images=read_images_binary(src/'colmap/refined_model_v2/images.bin');cams=read_cameras_binary(src/'colmap/refined_model_v2/cameras.bin')
names={'/'.join(Path(r['path']).parts[-2:]) for r in json.load(open('/mnt/d/k210_official_ab/data.json'))['rows'] if r['domain']=='project' and r['split']=='dev'}
ids=[i for i in sorted(images) if images[i].name in names];cfg=json.load(open(ROOT.parent/'config.json'))
variants=[('legacy_landscape',[320,240],160,False),('fixed_topk_landscape',[320,240],160,True),('fixed_topk_portrait',[184,320],160,True),('portrait_medium',[360,640],160,True),('legacy_scale_reference',[576,1024],160,True)]
protocol={'scope':'DEV engineering ablation only; source reconstruction contains query frames','tuning_ids':ids[:10],'confirmation_ids':ids[10:],'variants':variants,'fixed':['MixVPR checkpoint','map','threshold .8 and margin .05','max_active600','PnP100iter4px'],'selection':'on first10: most good poses then lowest median total_ms; good means >=8 inliers, <=5deg and <=0.1 native center error; confirmation not used to reselect','GT1':False,'architecture_change':False,'network_training':False}
(ROOT/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
g=TorchStudent(frozen/'candidate.pt');store=MapStore(frozen/'real_map',g.identity,'UNVERIFIED_LEGACY_SUPERPOINT');sp=SuperPointCPU([320,240],160)
class Legacy:
 identity='legacy-bug-replay'
 def infer(self,gray):
  import torch
  with torch.inference_mode():o=sp.model({'image':torch.from_numpy(gray.astype(np.float32)/255)[None,None]})
  return o['keypoints'][0].numpy(),o['descriptors'][0].numpy().T

def run(variant,selected,stage):
 name,size,cap,fixed=variant;sp.model.net.config['max_keypoints']=cap if fixed else 3072;c=dict(cfg,superpoint_size=size,max_keypoints=cap);results=[]
 for i in selected:
  im=images[i];cam=cams[im.camera_id];assert cam.model=='SIMPLE_RADIAL';f,cx,cy,k=cam.params;c['camera_size']=[cam.width,cam.height]
  pipe=Pipeline(store,g,sp if fixed else Legacy(),c,[[f,0,cx],[0,f,cy],[0,0,1]],[k,0,0,0,0]);cv2.setRNGSeed(17)
  r=pipe.frame(cv2.imread(str(Path('/home/quyet/edge_ai_project/assets/stair_6_7')/im.name)),0,0);r.update(image_id=i,variant=name,stage=stage,good_pose=False)
  if r['status']=='pose':
   R=np.array(r['world_to_camera_R']);Rgt=qvec2rotmat(im.qvec);angle=float(np.degrees(np.arccos(np.clip((np.trace(R@Rgt.T)-1)/2,-1,1))));error=float(np.linalg.norm(np.array(r['camera_center_world'])+Rgt.T@im.tvec));r.update(rotation_deg=angle,center_native=error,good_pose=angle<=5 and error<=.1)
  results.append(r)
 print(name,stage,sum(r['good_pose'] for r in results),len(results),flush=True);return results
allresults=[];summaries=[]
for variant in variants:
 rs=run(variant,ids[:10],'tuning');allresults+=rs;summaries.append({'variant':variant[0],'good_pose':sum(r['good_pose'] for r in rs),'poses':sum(r['status']=='pose' for r in rs),'median_ms':float(np.median([r['timings_ms']['total'] for r in rs])),'median_matches':float(np.median([r['correspondences'] for r in rs]))})
chosen=max(summaries,key=lambda x:(x['good_pose'],-x['median_ms']));v=next(v for v in variants if v[0]==chosen['variant'])
(ROOT/'selection_before_confirmation.json').write_text(json.dumps(chosen,indent=2)+'\n')
for variant in [variants[0],v]:allresults+=run(variant,ids[10:],'confirmation')
(ROOT/'results.json').write_text(json.dumps({'summary':summaries,'selected':chosen,'frames':allresults},indent=2)+'\n')
