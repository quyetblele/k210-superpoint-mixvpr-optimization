"""Native COLMAP camera geometry for replacement-map diagnostics."""
import numpy as np
import pycolmap

def camera(cam):
    return pycolmap.Camera(model=cam.model,width=cam.width,height=cam.height,params=cam.params)

def project(xyz,im,cam):
    pose=pycolmap.Rigid3d(pycolmap.Rotation3d(np.r_[im.qvec[1:],im.qvec[0]]),im.tvec)
    return camera(cam).img_from_cam(pose*np.asarray(xyz,np.float64))

def pose(xyz,xy,cam,im):
    if len(xy)<8:
        return {'good_pose':False,'inliers':0,'rotation_deg':None,'center_native':None}
    pycolmap.set_random_seed(17)
    result=pycolmap.estimate_absolute_pose(np.asarray(xy,np.float64),np.asarray(xyz,np.float64),camera(cam),{'ransac':{'max_error':4.0,'min_num_trials':100,'max_num_trials':1000,'confidence':.99}})
    if result is None:return {'good_pose':False,'inliers':0,'rotation_deg':None,'center_native':None}
    estimate=result['cam_from_world'];gt=pycolmap.Rigid3d(pycolmap.Rotation3d(np.r_[im.qvec[1:],im.qvec[0]]),im.tvec)
    angle=float(np.degrees(estimate.rotation.angle_to(gt.rotation)));error=float(np.linalg.norm(estimate.inverse().translation-gt.inverse().translation))
    depths=(estimate*np.asarray(xyz,np.float64))[np.asarray(result['inlier_mask'],bool),2]
    return {'good_pose':bool(len(depths)>=8 and (depths>0).all() and np.isfinite([angle,error]).all() and angle<=5 and error<=.1),'inliers':int(result['num_inliers']),'rotation_deg':angle,'center_native':error}
