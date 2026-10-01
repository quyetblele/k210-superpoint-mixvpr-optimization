"""Read-only COLMAP replacement-map sanity; no feature/localization quality claim."""
import _bootstrap
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
from spk210.map_nms_gate import read_images_binary,read_cameras_binary,read_points3D_binary,qvec2rotmat
from spk210.map_geometry import pose,project
from spk210.runtime import cv2
from spk210.io import sha,put

def run(package,assets,out):
    model=package/'colmap/refined_model_v2';images=read_images_binary(model/'images.bin');cameras=read_cameras_binary(model/'cameras.bin');points=read_points3D_binary(model/'points3D.bin')
    missing=[];bad=[];duplicates=[]
    for i,im in images.items():
        f=assets/im.name
        if not f.is_file():missing.append(im.name);continue
        with Image.open(f) as pic:assert pic.size==(cameras[im.camera_id].width,cameras[im.camera_id].height)
        ids=im.point3D_ids[im.point3D_ids>=0];u,n=np.unique(ids,return_counts=True)
        if (n>1).any():duplicates.append({'image':i,'duplicated_point_ids':u[n>1].tolist()})
        assert np.isfinite(im.xys).all() and np.isfinite(im.qvec).all() and np.isfinite(im.tvec).all()
        for idx,pid in enumerate(im.point3D_ids):
            if pid<0:continue
            if int(pid) not in points:bad.append([i,idx,int(pid)]);continue
            point=points[int(pid)]
            assert ((point.image_ids==i)&(point.point2D_idxs==idx)).any()
    for pid,point in points.items():
        assert np.isfinite(point.xyz).all()
        for i,j in zip(point.image_ids,point.point2D_idxs):assert images[int(i)].point3D_ids[int(j)]==pid
    assert not missing and not bad
    ordered=sorted(images,key=lambda i:images[i].name);queries=[ordered[int(j)] for j in np.linspace(0,len(ordered)-1,20,dtype=int)]
    rows=[]
    for q in queries:
        im=images[q];cam=cameras[im.camera_id];ids=im.point3D_ids;u,c=np.unique(ids[ids>=0],return_counts=True);keep=(ids>=0)&~np.isin(ids,u[c>1]);xyz=np.array([points[int(v)].xyz for v in ids[keep]]);xy=im.xys[keep]
        r=pose(xyz,xy,cam,im);uv=project(xyz,im,cam)
        r.update(query=q,image=im.name,median_reprojection_px=float(np.median(np.linalg.norm(uv-xy,axis=1))));rows.append(r)
    assert all(r['good_pose'] for r in rows)
    out.mkdir(parents=True,exist_ok=True)
    put(out/'geometry_sanity.json',{'status':'PASS_INTERNAL_ONLY','geometry_backend':'pycolmap3.13 native camera and LO-RANSAC; different from retired SIMPLE_RADIAL OpenCV evaluator','package':str(package),'assets':str(assets),'images':len(images),'points3D':len(points),'camera_models':sorted(set(c.model for c in cameras.values())), 'missing_images':missing,'invalid_point_ids':bad,'duplicate_observations':duplicates,'rows':rows,'geometry_hashes':{f:sha(model/f) for f in ['cameras.bin','images.bin','points3D.bin']},'scope':'all source paths/dimensions and reciprocal tracks checked;20 deterministic known-correspondence PnP cases; no independent GT or teacher descriptor baseline yet','source_sha256':sha(__file__)})
    rowsdata=[{'path':str(assets/images[i].name),'image_name':images[i].name,'domain':'project','split':'dev' if i in queries else 'train'} for i in ordered]
    put(out/'data.json',{'scope':'map reference/query roles only; train means reference, NOT neural training; exploratory DEV, not TEST','rows':rowsdata})
    print('PASS',len(images),'images',len(points),'points',len(rows),'internal PnP; duplicates',len(duplicates),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--package',type=Path,required=True);p.add_argument('--assets',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();run(a.package,a.assets,a.output)
