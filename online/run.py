import argparse,json
from pathlib import Path
from pipeline import MapStore,Pipeline,frames
from backends import TorchStudent,OnnxStudent,SuperPointCPU

def main():
 p=argparse.ArgumentParser();p.add_argument('--map',required=True);p.add_argument('--model',required=True);p.add_argument('--backend',choices=['fp32','onnx'],default='fp32');p.add_argument('--video');p.add_argument('--camera',type=int);p.add_argument('--calibration',required=True);p.add_argument('--config',default=str(Path(__file__).with_name('config.json')));p.add_argument('--output',required=True);p.add_argument('--max-frames',type=int,default=100);p.add_argument('--local-backend',choices=['torch','onnx'],default='torch');p.add_argument('--local-model');p.add_argument('--threads',type=int,default=2);p.add_argument('--allow-legacy-local-map',action='store_true',help='Explicit development experiment; local map/query compatibility remains unverified');a=p.parse_args()
 if (a.video is None)==(a.camera is None):p.error('choose exactly one video/camera')
 cfg=json.loads(Path(a.config).read_text());cal=json.loads(Path(a.calibration).read_text())
 if cal['size']!=cfg['camera_size']:raise ValueError('calibration size mismatch')
 if cal.get('model','OPENCV') not in ('PINHOLE','OPENCV'):
  raise ValueError('camera model requires verified undistortion before OpenCV PnP: '+str(cal.get('model')))
 g=(TorchStudent if a.backend=='fp32' else OnnxStudent)(a.model)
 if a.local_backend=='onnx':
  if not a.local_model:p.error('--local-model required for ONNX')
  from superpoint_onnx import SuperPointOnnx
  l=SuperPointOnnx(a.local_model,cfg['superpoint_size'],cfg['max_keypoints'],a.threads)
 else:l=SuperPointCPU(cfg['superpoint_size'],cfg['max_keypoints'])
 import torch
 torch.set_num_threads(a.threads)
 expected_local=json.loads((Path(a.map)/'map.json').read_text())['local_identity'] if a.allow_legacy_local_map else l.identity
 store=MapStore(a.map,g.identity,expected_local);pipe=Pipeline(store,g,l,cfg,cal['K'],cal.get('distortion'))
 with open(a.output,'w') as f:
  for i,t,bgr in frames(a.camera if a.camera is not None else a.video):
   if i>=a.max_frames:break
   result=pipe.frame(bgr,i,t);result['local_map_compatibility']='unverified-development-override' if a.allow_legacy_local_map else 'identity-checked';f.write(json.dumps(result,allow_nan=False)+'\n');f.flush()
if __name__=='__main__':main()
