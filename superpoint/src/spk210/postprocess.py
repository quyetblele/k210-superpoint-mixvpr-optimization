"""Explicit, deterministic NMS policy for SuperPoint CPU postprocessing."""
import json
import numpy as np
from .settings import SUPERPOINT

def default_nms():
    path=SUPERPOINT/'configs/postprocess.json'
    return json.loads(path.read_text())['nms'] if path.exists() else 'original'

def greedy_keypoints(score,radius,threshold,max_keypoints):
    values=score.detach().cpu().numpy();yy,xx=np.nonzero(values>threshold)
    order=np.lexsort((xx,yy,-values[yy,xx]));blocked=np.zeros_like(values,dtype=bool);selected=[]
    for k in order:
        y,x=int(yy[k]),int(xx[k])
        if blocked[y,x]:continue
        selected.append((x,y));blocked[max(0,y-radius):y+radius+1,max(0,x-radius):x+radius+1]=True
        if len(selected)==max_keypoints:break
    return np.asarray(selected,np.float32).reshape(-1,2)
