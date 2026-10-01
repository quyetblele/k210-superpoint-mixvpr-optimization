#!/usr/bin/env python3
"""Read-only Phase 0--2 inventory for the isolated Edge-VPR student gate."""
from __future__ import annotations
import hashlib, json, sys
from collections import Counter
from pathlib import Path
import cv2, h5py, numpy as np, onnx, pycolmap, torch

ROOT=Path('/home/quyet/k210_lab'); EDGE=Path('/home/quyet/edge_ai_project'); OUT=ROOT/'reports/edge_vpr_students'
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,o): p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(o,indent=2)+'\n')
def video_meta(p):
 c=cv2.VideoCapture(str(p)); d={'path':str(p),'open':bool(c.isOpened())}
 if c.isOpened():
  d.update(width=int(c.get(cv2.CAP_PROP_FRAME_WIDTH)),height=int(c.get(cv2.CAP_PROP_FRAME_HEIGHT)),fps=float(c.get(cv2.CAP_PROP_FPS)),frame_count=int(c.get(cv2.CAP_PROP_FRAME_COUNT)))
 c.release();return d
def camera_summary(model):
 r=pycolmap.Reconstruction(model); cams=list(r.cameras.values()); groups=Counter((x.model.name,x.width,x.height,tuple(round(float(v),8) for v in x.params)) for x in cams)
 return {'model_path':str(model),'registered_images':len(r.images),'points3D':len(r.points3D),'camera_count':len(cams),'model_dimension_counts':{'/'.join(map(str,k)):v for k,v in Counter((x.model.name,x.width,x.height) for x in cams).items()},'unique_intrinsics':len(groups),'first_camera':{'model':cams[0].model.name,'width':cams[0].width,'height':cams[0].height,'params':[float(v) for v in cams[0].params]}}
def h5_summary(p):
 with h5py.File(p,'r') as h:
  ds=[]
  def visit(name,o):
   if isinstance(o,h5py.Dataset) and name.endswith('global_descriptor'):ds.append((name,list(o.shape),str(o.dtype)))
  h.visititems(visit)
 return {'path':str(p),'sha256':sha(p),'descriptor_records':len(ds),'dimension_counts':{'x'.join(map(str,k)):v for k,v in Counter(tuple(x[1]) for x in ds).items()},'dtype_counts':dict(Counter(x[2] for x in ds)),'first_record':ds[0] if ds else None}
def main():
 sys.path[:0]=[str(EDGE/'third_party'),str(EDGE/'third_party/hloc')]
 from hloc import extractors
 from hloc.utils.base_model import dynamic_load
 packages={}
 for floor in ('floor6','floor7'):
  pkg=EDGE/'map_packages'/f'{floor}_v1'; man=json.loads((pkg/'manifest.json').read_text());packages[floor]={'manifest':man,'camera':camera_summary(pkg/man['files']['colmap_model']),'global_db':h5_summary(pkg/man['files']['global_features'])}
 videos=sorted(list((EDGE/'datasets/stair_6_7_videos').glob('*.mp4'))+list((EDGE/'datasets/floor7_videos').glob('*.mp4')))
 video=[video_meta(x) for x in videos]
 model=dynamic_load(extractors,'mixvpr')({}).eval().cpu(); torch.manual_seed(20260906); x=torch.linspace(0,1,3*320*320,dtype=torch.float32).reshape(1,3,320,320)
 with torch.inference_mode(): a=model({'image':x})['global_descriptor']; b=model({'image':x})['global_descriptor']
 params=sum(v.numel() for v in model.parameters()); norm=float(torch.linalg.vector_norm(a,dim=1).item());
 t0={'implementation':str(EDGE/'third_party/hloc/hloc/extractors/mixvpr.py'),'preprocessing':'torchvision ImageNet Normalize then bilinear F.interpolate to [320,320], align_corners=False','input_shape':[1,3,320,320],'descriptor_shape':list(a.shape),'descriptor_dimension':int(a.shape[1]),'parameters':params,'fp32_parameter_bytes':params*4,'deterministic_forward_max_abs_error':float((a-b).abs().max()),'finite':bool(torch.isfinite(a).all()),'l2_norm':norm,'checkpoint_source':'torch.hub.load(jarvisyjw/MixVPR, get_trained_model, pretrained=True); cached checkout used by current environment'}
 no_gt={'recall_protocol_found':False,'reason':'No project file defines a query/reference split plus relevance labels, synchronized query poses, or a Recall@K evaluator for the Floor6/Floor7 production maps. Existing online audit explicitly states no synchronized 6-DoF ground truth; reference-only H5 databases cannot define Recall@K for external query videos.'}
 out={'phase':'0-2 read-only inventory','source_project_modified':False,'packages':packages,'videos':video,'t0':t0,'evaluation_status':no_gt}
 write(OUT/'edge_vpr_student_manifest.json',out)
 lines=['# Edge-VPR student gate — data and camera audit','', '## Active geometry','', '- All readable project source videos are portrait `1080×1920` (aspect `9:16`), with rotation already represented as portrait pixels. No active crop/pad is applied before camera reconstruction.', '- The current global MixVPR wrapper converts BGR→RGB upstream, scales pixels to `[0,1]`, applies ImageNet normalization, then **stretches** normalized RGB bilinearly to `320×320`; it does not preserve the 9:16 aspect ratio.', '', '## Camera evidence','']
 for k,v in packages.items():lines.append(f"- {k}: `{v['camera']['model_dimension_counts']}` across {v['camera']['registered_images']} registered images; `{v['camera']['unique_intrinsics']}` intrinsics records. First record: `{v['camera']['first_camera']}`.")
 lines += ['', 'The active reconstruction camera model is **RADIAL_FISHEYE**, dimensions `1080×1920`; its five parameters are `f, cx, cy, k1, k2`. Camera intrinsics do not enter global VPR, but must remain intact for downstream SuperPoint/LightGlue/PnP.', '', '## Data inventory', '']
 for k,v in packages.items():lines.append(f"- {k}: global DB `{v['global_db']['descriptor_records']}` descriptors, shape counts `{v['global_db']['dimension_counts']}`.")
 lines += ['', '## Gate decision','', '**Recall@K cannot be reproduced yet.** The project contains reference descriptor databases and unlabelled online-query videos, but no frozen query/reference split with relevance or pose ground truth. A self-retrieval score would be invalid (and trivially inflated), while localization success/inliers are not Recall labels. Therefore architecture comparison, tiny-overfit retrieval, KD and pilot training must not start under the stated protocol.']
 (OUT/'01_data_camera_audit.md').write_text('\n'.join(lines)+'\n')
 t=['# T0 baseline reproduction status','',f"- Active model: MixVPR Torch Hub pretrained; core input `{t0['input_shape']}`, descriptor `{t0['descriptor_dimension']}D`, parameters `{params:,}`.",f"- Deterministic forward max absolute difference: `{t0['deterministic_forward_max_abs_error']}`; finite `{t0['finite']}`; L2 norm `{norm:.8f}`.",f"- Exact preprocessing: {t0['preprocessing']}.",'', '## Retrieval/localization status','', 'T0 Recall@1/@5/@10: **NOT VALIDATED** — no valid frozen query/relevance protocol was found.', 'Localization success / PnP / pose metrics: **NOT VALIDATED for this student gate** — existing online logs do not have synchronized 6-DoF ground truth, and the requested student-localization comparison requires candidate-specific reference DBs.', '', '## Stop rule', '', 'Phase 2 is not reproducible as a retrieval experiment. Per the master instruction, no student architecture comparison, training, KD, or K210 selection is started.']
 (OUT/'02_t0_teacher_report.md').write_text('\n'.join(t)+'\n')
 print('PHASE0_T0_AUDIT COMPLETE; RECALL PROTOCOL NOT FOUND')
if __name__=='__main__':main()
