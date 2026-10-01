import json,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pipeline import MapStore,match
p=Path(__file__).parent;base=p.parent/'maps/stair_local_v1';out=p.parent/'maps/stair_local_int8_candidate';out.mkdir(exist_ok=True)
r=json.loads((p/'report.json').read_text());queries=json.loads((p/'queries.json').read_text());b={x['image_id']:x for x in queries if x['variant']=='FP32'};v=[x for x in queries if x['variant']=='INT8_per_descriptor']
assert all(x['pose_good'] for x in v if b[x['image_id']]['pose_good'])
assert r['results']['INT8_per_descriptor']['median_rotation_deg']<=r['results']['FP32']['median_rotation_deg']+.1
assert r['results']['INT8_per_descriptor']['median_center_native']<=r['results']['FP32']['median_center_native']+.01
for f in base.glob('*.npy'):
 target=p/'local_int8.npy' if f.name=='local.npy' else f
 dest=out/f.name
 if not dest.exists():dest.symlink_to(target)
for name in ['reference_images.json','source_camera.json']:
 if not (out/name).exists():(out/name).symlink_to(base/name)
if not (out/'local_scale.npy').exists():(out/'local_scale.npy').symlink_to(p/'local_int8_scale.npy')
meta=json.loads((base/'map.json').read_text());meta.update(status='PROVISIONAL_INT8_DATABASE_LOCAL_ONLY',descriptor_quantization={'type':'symmetric per-descriptor int8','zero_point':0,'scale_file':'local_scale.npy','formula':'round(unit_descriptor / (max_abs / 127))','runtime':'normalize int8 values as float32 per block; scalar scale cancels in cosine','network_PTQ':False},pending=meta['pending']+['independent query validation of descriptor compression','integer-only matcher board parity'])
(out/'map.json').write_text(json.dumps(meta,indent=2)+'\n')
s=MapStore.__new__(MapStore)
for n in ['point_ids','xyz','local','vis_offsets','vis_rows','cov_offsets','cov_rows']:setattr(s,n,np.load(out/(n+'.npy'),mmap_mode='r'))
cfg=json.loads((p.parent/'config.json').read_text());active=s.active([0],8,600);q=np.load(base/'local.npy',mmap_mode='r')[active[:32]];m,raw=match(q,s,active,cfg);assert len(m)>0
sizes={'FP32_local_payload':7770*256*4,'INT8_local_payload_plus_scale':7770*(256+4)}
report={'decision':'INT8 descriptor candidate, no pruning','provisional':True,'lost_baseline_good_queries':0,'pruning_rejected':'track>=5,error<=1.75 loses one baseline good-pose query','round1_note':'track>=3,error<=2 pruning was a no-op because source map already used those filters; retained round1 report; round2 is exploratory','bytes':sizes,'descriptor_reduction_percent':100*(1-sizes['INT8_local_payload_plus_scale']/sizes['FP32_local_payload']),'candidate':str(out),'runtime_active_test':{'points':len(active),'matches':len(m)},'limits':'41 reconstructed DEV queries, leave-query-out centroid only; geometry still includes query observations; stored legacy features and all-map scan; not independent/online/K210 evaluation'}
(p/'decision.json').write_text(json.dumps(report,indent=2)+'\n')
lines=['# Map preparation quality audit','','Scope: existing stair development map only. Not Floor6/Floor7 independent test.','Queries are the 41 registered project DEV images; subtract each query descriptor contribution from map centroids. Use stored COLMAP pixel coordinates (feature coordinates checked within 1.01 px), per-image SIMPLE_RADIAL intrinsics, top160 legacy keypoints, full-map block scan and deterministic PnP. Query geometry remains in the reconstruction, so results are optimistic internal checks.','','| Variant | Good pose | Median rotation (deg) | Median center (native units) | Median matches |','|---|---:|---:|---:|---:|']
for name,x in r['results'].items():lines.append(f"| {name} | {x['good_pose']}/{x['queries']} | {x['median_rotation_deg']:.4f} | {x['median_center_native']:.6f} | {x['median_matches']:.0f} |")
lines+=['','Good pose means at least 8 inliers, <=5 degrees and <=0.1 native units relative to reconstructed pose; not meters or external ground truth.','Point reprojection-error median 1.465 px, p90 1.873 px; track median 5 observations. These are filtered reconstructed points, not an unbiased accuracy estimate.','','Decision: retain all 7,770 points and FP32 XYZ. Provisional INT8 descriptor database reduces descriptor+scale payload from 7,956,480 to 2,020,200 bytes (74.61%). It passes the internal paired no-regression guard. FP16 is a fallback. Do not interpret tiny median differences as real improvements.','Pruning to track>=5 and error<=1.75 loses a baseline good-pose query, so is rejected. Initial track>=3/error<=2 filter was already applied in the source and did not prune anything.','','Candidate directory uses symlinks to avoid copying geometry; packaging for microSD must materialize referenced files. Global descriptors are still unbound. Local extractor compatibility remains unverified. INT8 values are normalized in float32 by the PC matcher: this is database quantization, not an integer-only runtime or model PTQ.','','Next test: bind frozen global model/reference database, validate local feature compatibility and calibration, then compare FP32 vs candidate on locked independent video queries with labels. No additional pruning is warranted by current evidence.']
(p/'report.md').write_text('\n'.join(lines)+'\n');print(json.dumps(report))
