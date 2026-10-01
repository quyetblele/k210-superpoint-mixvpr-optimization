"""Check native reconstruction observations and PnP independent of descriptors."""
import _bootstrap
import json
import numpy as np
from spk210.map_nms_gate import DEST,MODEL,ASSETS,context,cv2,qvec2rotmat,read_points3D_binary
from spk210.io import put,sha
images,cameras=context();points=read_points3D_binary(MODEL/'points3D.bin')
p=json.loads((DEST/'protocol.json').read_text());rows=[]
for i in p['split_ids']['dev']:
    im=images[i];cam=cameras[im.camera_id];assert cam.model=='SIMPLE_RADIAL'
    image=cv2.imread(str(ASSETS/im.name));assert image.shape[:2]==(cam.height,cam.width)
    valid=im.point3D_ids>=0;xyz=np.array([points[int(v)].xyz for v in im.point3D_ids[valid]]);xy=im.xys[valid]
    f,cx,cy,k=cam.params;K=np.array([[f,0,cx],[0,f,cy],[0,0,1.]]);dist=np.array([k,0,0,0,0.]);R=qvec2rotmat(im.qvec)
    uv,_=cv2.projectPoints(xyz,cv2.Rodrigues(R)[0],im.tvec,K,dist);error=np.linalg.norm(uv[:,0]-xy,axis=1)
    cv2.setRNGSeed(17)
    ok,rv,tv,ins=cv2.solvePnPRansac(xyz,xy,K,dist,iterationsCount=100,reprojectionError=4,confidence=.99,flags=cv2.SOLVEPNP_EPNP)
    angle=center=None
    if ok:
        est=cv2.Rodrigues(rv)[0];angle=float(np.degrees(np.arccos(np.clip((np.trace(est@R.T)-1)/2,-1,1))))
        center=float(np.linalg.norm(-est.T@tv.ravel()+R.T@im.tvec))
    rows.append({'id':i,'observations':len(xy),'median_reprojection_px':float(np.median(error)),'p95_reprojection_px':float(np.percentile(error,95)),
                 'inliers':0 if ins is None else len(ins),'rotation_deg':angle,'center_native':center,
                 'good_pose':bool(ok and len(ins)>=8 and angle<=5 and center<=.1)})
summary={'good_pose_with_known_correspondences':sum(r['good_pose'] for r in rows),'queries':len(rows),
         'median_image_median_reprojection_px':float(np.median([r['median_reprojection_px'] for r in rows]))}
put(DEST/'geometry_audit.json',{'source_sha256':sha(__file__),'geometry_hashes':{n:sha(MODEL/n) for n in p['geometry_hashes']},'scope':'internal reconstruction consistency, not absolute map accuracy; known correspondences oracle','summary':summary,'frames':rows})
print(json.dumps(summary))
