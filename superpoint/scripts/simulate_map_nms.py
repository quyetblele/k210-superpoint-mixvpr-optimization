"""Actual nncase simulation of the immutable real-map diagnostic inputs."""
import _bootstrap
import json
import numpy as np
import nncase
from spk210.settings import SUPERPOINT
from spk210.io import sha, put, guard

dest=SUPERPOINT/'artifacts/map_nms_gate'
dep=SUPERPOINT/'artifacts/deployment/wider_r2_cw'
p=json.loads((dest/'protocol.json').read_text());m=json.loads((dest/'inputs.json').read_text())
assert m['protocol_sha256']==sha(dest/'protocol.json')
assert sha(dep/'model.kmodel')==p['kmodel_sha256']
sim=nncase.Simulator();sim.load_model((dep/'model.kmodel').read_bytes())
assert sim.get_input_tensor(0).dtype==np.dtype('float32')
assert all(sim.get_output_tensor(j).dtype==np.dtype('float32') for j in range(2))
(dest/'int8').mkdir(exist_ok=True)
perm=np.r_[np.rot90(np.arange(64).reshape(8,8),-1).ravel(),64]
outputs={}
for k,row in enumerate(m['records']):
    guard();i=row['id'];path=dest/'inputs'/f'{i}.npz';assert sha(path)==row['input_sha256']
    with np.load(path) as z:x=np.ascontiguousarray(np.rot90(z['image'],-1,(2,3)))
    sim.set_input_tensor(0,nncase.RuntimeTensor.from_numpy(x));sim.run()
    a,b=[np.rot90(sim.get_output_tensor(j).to_numpy().copy(),1,(2,3)) for j in range(2)]
    a=a[:,np.argsort(perm)]
    assert a.shape==(1,65,40,23) and b.shape==(1,256,40,23)
    assert np.isfinite(a).all() and np.isfinite(b).all()
    out=dest/'int8'/f'{i}.npz';np.savez_compressed(out,logits=a,desc=b);outputs[str(i)]=sha(out)
    if k%30==0:print('SIMULATE',k+1,len(m['records']),flush=True)
put(dest/'simulation.json',{'backend':'actual nncase K210 Simulator','kmodel_sha256':p['kmodel_sha256'],'inputs_manifest_sha256':sha(dest/'inputs.json'),'source_sha256':sha(__file__),'outputs':outputs})
