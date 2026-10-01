import onnx,numpy as np,onnxruntime,json,hashlib
from onnx import helper,TensorProto
from pathlib import Path
R=Path('/home/quyet/k210_lab');s=R/'models/s512_br1_prenorm_core_192x352_inferred.onnx';d=R/'models/s512_br1_prenorm_core_192x352_canonical_v2.onnx';m=onnx.load(s);g=m.graph;prod={o:n for n in g.node for o in n.output};cons={}
for n in g.node:
 for x in n.input:cons.setdefault(x,[]).append(n)
reshape=cons['/Concat_output_0'][0];assert reshape.op_type=='Reshape';old='/Concat_output_0';init=helper.make_tensor('br1_static_reshape_1x144x66',TensorProto.INT64,[3],[1,144,-1]);g.initializer.append(init);reshape.input[1]='br1_static_reshape_1x144x66'
cons={}
for n in g.node:
 for x in n.input:cons.setdefault(x,[]).append(n)
# DCE exactly the old chain: start at Concat producer and remove only nodes no longer consumed.
dead=set();todo=[old]
while todo:
 out=todo.pop();n=prod.get(out)
 if n is None or n.name in dead:continue
 if any(c.name not in dead for c in cons.get(out,[])):continue
 dead.add(n.name);todo.extend(n.input)
removed=sorted(dead);keep=[n for n in g.node if n.name not in dead];del g.node[:];g.node.extend(keep);onnx.checker.check_model(m);q=onnx.shape_inference.infer_shapes(m,strict_mode=True);onnx.checker.check_model(q);onnx.save(q,d)
a=onnxruntime.InferenceSession(str(s));b=onnxruntime.InferenceSession(str(d));rows=[]
for n,x in [('linspace',np.linspace(-1,1,3*352*192,dtype='float32').reshape(1,3,352,192)),('random',np.random.default_rng(20260906).normal(size=(1,3,352,192)).astype('float32')),('zeros',np.zeros((1,3,352,192),'float32'))]:
 u=a.run(None,{'normalized_rgb':x})[0];v=b.run(None,{'normalized_rgb':x})[0];rows.append({'case':n,'shape':list(v.shape),'finite':bool(np.isfinite(v).all()),'max_abs':float(np.max(abs(u-v))),'cosine':float(u.ravel()@v.ravel()/(np.linalg.norm(u)*np.linalg.norm(v)))})
r={'reshape':{'name':reshape.name,'data':reshape.input[0],'old_shape':old,'output':reshape.output[0],'replacement':[1,144,-1]},'removed':removed,'before':len(onnx.load(s).graph.node),'after':len(q.graph.node),'sha256':hashlib.sha256(d.read_bytes()).hexdigest(),'parity':rows};(R/'reports/edge_vpr_students/s512_br1_canonical_v2.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
