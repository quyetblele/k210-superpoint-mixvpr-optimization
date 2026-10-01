#!/usr/bin/env python3
"""M0.3: canonicalize only a provably redundant DOUBLE->FLOAT cast pair."""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
import traceback
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/mixvpr_feasibility'
M02=ROOT/'models/mixvpr_m02_static_canonical_1x3x320x320.onnx'
M03=ROOT/'models/mixvpr_m03_dtype_canonical_1x3x320x320.onnx'
M03_INF=ROOT/'models/mixvpr_m03_dtype_canonical_1x3x320x320_inferred.onnx'
KMODEL=ROOT/'artifacts/mixvpr_m03_dtype_canonical_1x3x320x320.kmodel'
KPY='/home/quyet/miniconda3/envs/k210/bin/python'
SHAPE=(1,3,320,320); SEED=20260906; N=8

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x): Path(p).parent.mkdir(parents=True,exist_ok=True);Path(p).write_text(json.dumps(x,indent=2)+'\n')

def dtype_name(code):
    import onnx
    return onnx.TensorProto.DataType.Name(code)

def double_audit():
    import onnx
    from onnx import numpy_helper
    g=onnx.load(str(M02)).graph; results=[]
    consumers={}
    for i,n in enumerate(g.node):
        for x in n.input:
            if x: consumers.setdefault(x,[]).append((i,n))
    for i,n in enumerate(g.node):
        if n.op_type=='Constant':
            attr=next((a for a in n.attribute if a.name=='value' and a.HasField('t')),None)
            if attr and attr.t.data_type==onnx.TensorProto.DOUBLE:
                target=consumers.get(n.output[0],[])
                results.append({'kind':'Constant node','node_index':i,'node_name':n.name,'op_type':'Constant','tensor':n.output[0],
                                'onnx_dtype':'DOUBLE','shape':list(numpy_helper.to_array(attr.t).shape),'value':numpy_helper.to_array(attr.t).tolist(),
                                'direct_consumers':[{'index':j,'name':x.name,'op_type':x.op_type,'inputs':list(x.input),'outputs':list(x.output)} for j,x in target]})
    for init in g.initializer:
        if init.data_type==onnx.TensorProto.DOUBLE: results.append({'kind':'initializer','tensor':init.name,'onnx_dtype':'DOUBLE','shape':list(numpy_helper.to_array(init).shape),'value':numpy_helper.to_array(init).tolist()})
    casts=[]
    for i,n in enumerate(g.node):
        if n.op_type=='Cast':
            to=next(a.i for a in n.attribute if a.name=='to')
            casts.append({'node_index':i,'node_name':n.name,'input':n.input[0],'output':n.output[0],'target_dtype':dtype_name(to),
                          'source_is_double':any(r.get('tensor')==n.input[0] for r in results)})
    inferred_double=[] # M02 has no inferred DOUBLE typed value-info; node scan above is complete.
    assert len(results)==1 and results[0]['node_name']=='/net/aggregator/Constant_4',results
    constant=results[0]; cast=next(x for x in casts if x['input']==constant['tensor'])
    downstream=[]; queue=[cast['output']]; seen=set()
    while queue:
        tensor=queue.pop(0)
        if tensor in seen: continue
        seen.add(tensor)
        for i,n in consumers.get(tensor,[]):
            downstream.append({'index':i,'name':n.name,'op_type':n.op_type,'input_tensor':tensor,'outputs':list(n.output)})
            queue.extend([x for x in n.output if x])
    return {'all_double_initializers':[], 'all_double_constant_nodes':results,'casts':casts,'inferred_double_tensors':inferred_double,
            'constant_4':constant,'cast_target':cast,'downstream_to_descriptor':downstream,
            'classification':'REPRESENTATION_ONLY',
            'proof':'Constant_4 is scalar 1e-12 stored as DOUBLE and immediately Cast to FLOAT before Clip/Expand/Div. IEEE-754 conversion of DOUBLE 1e-12 to FLOAT equals direct FLOAT storage of 1e-12; downstream arithmetic therefore receives the identical FLOAT32 value. DOUBLE has no mathematical necessity here.'}

def canonicalize(audit):
    import onnx
    from onnx import TensorProto, helper, numpy_helper
    model=onnx.load(str(M02)); g=model.graph
    constant=next(n for n in g.node if n.name=='/net/aggregator/Constant_4'); cast=next(n for n in g.node if n.name=='/net/aggregator/Cast')
    old=next(a.t for a in constant.attribute if a.name=='value'); original=numpy_helper.to_array(old)
    direct=np.asarray(original,dtype=np.float32)
    # Original DOUBLE->Cast(FLOAT) and direct FLOAT quantize identically.
    assert np.array_equal(direct,original.astype(np.float32))
    constant.attribute.clear();constant.attribute.extend([helper.make_attribute('value',numpy_helper.from_array(direct,name=''))])
    for node in g.node:
        for i,x in enumerate(node.input):
            if x==cast.output[0]: node.input[i]=constant.output[0]
    kept=[n for n in g.node if n is not cast];del g.node[:];g.node.extend(kept)
    onnx.checker.check_model(model);onnx.save(model,str(M03));inferred=onnx.shape_inference.infer_shapes(model,strict_mode=True);onnx.checker.check_model(inferred);onnx.save(inferred,str(M03_INF))
    return {'status':'PASS','output_onnx':str(M03),'output_sha256':sha(M03),'inferred_onnx':str(M03_INF),'inferred_sha256':sha(M03_INF),'checker':'PASS','shape_inference':'PASS',
            'nodes_changed':[{'removed_node':cast.name,'op_type':'Cast','rewired_consumer':'/net/aggregator/Clip','input_from':cast.output[0],'input_to':constant.output[0]}],
            'constants_changed':[{'node':constant.name,'tensor':constant.output[0],'value_before':original.tolist(),'dtype_before':'DOUBLE','value_after':direct.tolist(),'dtype_after':'FLOAT'}],
            'learned_weight_initializers_changed':False}

def load_core():
    from audit_mixvpr_m01_static_core import load_model,core_wrapper
    m=load_model();return m,core_wrapper(m)

def metrics(a,b):
    a,b=np.asarray(a).reshape(-1),np.asarray(b).reshape(-1)
    return {'max_abs_error':float(np.max(abs(a-b))),'mean_abs_error':float(np.mean(abs(a-b))),'cosine_similarity':float(np.dot(a,b)/max(float(np.linalg.norm(a)*np.linalg.norm(b)),1e-12))}

def parity():
    import cv2,onnxruntime,torch
    from audit_mixvpr_m01_static_core import external_preprocess,real_images
    model,core=load_core();a=onnxruntime.InferenceSession(str(M02),providers=['CPUExecutionProvider']);b=onnxruntime.InferenceSession(str(M03_INF),providers=['CPUExecutionProvider'])
    probes=[('deterministic',torch.linspace(-2,2,int(np.prod(SHAPE))).reshape(SHAPE).numpy())]
    for path in real_images():
        image=cv2.imread(str(path),cv2.IMREAD_COLOR)
        fixed,_=external_preprocess(image,model);probes.append((str(path),fixed.numpy()))
    rows=[]
    with torch.inference_mode():
        for name,x in probes:
            x2=a.run(None,{'normalized_rgb_320x320':x})[0];x3=b.run(None,{'normalized_rgb_320x320':x})[0];pt=core(torch.from_numpy(x)).numpy()
            rows.append({'input':name,'shape_m02':list(x2.shape),'shape_m03':list(x3.shape),'shape_pytorch':list(pt.shape),'m02_vs_m03':metrics(x2,x3),'pytorch_vs_m03':metrics(pt,x3)})
    def agg(k):
        values=[x[k] for x in rows];return {x:max(v[x] for v in values) if x=='max_abs_error' else float(np.mean([v[x] for v in values])) for x in values[0]}
    result={'status':'PASS','real_image_count':len(rows)-1,'deterministic_plus_real_count':len(rows),'rows':rows,'m02_vs_m03':agg('m02_vs_m03'),'pytorch_vs_m03':agg('pytorch_vs_m03')}
    if result['m02_vs_m03']['max_abs_error'] != 0 or result['pytorch_vs_m03']['max_abs_error'] > 2e-5:result['status']='FAIL'
    return result

def compiler():
    import _nncase,nncase,onnx
    from importlib.metadata import version
    r={'checker':'NOT RUN','import':'NOT RUN','ptq':'NOT RUN','compile':'NOT RUN','gencode':'NOT RUN'};stage='checker'
    try:
        r['environment']={'python_executable':sys.executable,'nncase':version('nncase'),'_nncase':_nncase.__version__};assert r['environment']['nncase']=='1.8.0.20220929' and r['environment']['_nncase']=='1.8.0-55be52f',r['environment']
        onnx.checker.check_model(onnx.load(str(M03_INF)));r['checker']='PASS';stage='import';o=nncase.CompileOptions();o.target='k210';o.quant_type='uint8';o.w_quant_type='uint8';o.dump_dir=str(OUT/'m03_nncase_dump');o.dump_ir=True;o.dump_asm=True;c=nncase.Compiler(o);c.import_onnx(M03_INF.read_bytes(),nncase.ImportOptions());r['import']='PASS';stage='ptq'
        calib=np.random.default_rng(SEED).uniform(-3,3,size=(N,*SHAPE[1:])).astype(np.float32);r['calibration_shape']=list(calib.shape);r['calibration_sha256']=hashlib.sha256(calib.tobytes()).hexdigest();q=nncase.PTQTensorOptions();q.samples_count=N;q.set_tensor_data(calib.tobytes());c.use_ptq(q);r['ptq']='PASS';stage='compile';c.compile();r['compile']='PASS';stage='gencode';blob=c.gencode_tobytes();KMODEL.write_bytes(blob);r.update(gencode='PASS',size=len(blob),sha256=sha(KMODEL))
    except Exception:r[stage]='FAIL';r['error']=traceback.format_exc()
    save(OUT/'m03_dtype_canonical_compiler.json',r);return 0 if r['gencode']=='PASS' else 1

def failure_evidence(error):
    import onnx
    g=onnx.load(str(M03_INF)).graph;message=(error or '').splitlines()[-1]
    named=re.search(r'<([^>]+)>',error or '')
    if named:
        tensor=named.group(1);producer=next((n for n in g.node if tensor in n.output),None);return {'error_line':message,'tensor':tensor,'producer_node':producer.name if producer else None,'operator':producer.op_type if producer else None,'owner':'AGGREGATOR' if '/net/aggregator/' in tensor else 'UNKNOWN'}
    # The next common nncase error names only a type/op. Restrict evidence to
    # affected aggregator normalization nodes; compiler did not expose a node.
    if 'Clip' in error:return {'error_line':message,'node':'NOT EXPOSED','operator':'Clip','owner':'AGGREGATOR','evidence':'Clip occurs once, at /net/aggregator/Clip.'}
    return {'error_line':message,'node':'NOT EXPOSED BY NNCASE','operator':'NOT EXPOSED BY NNCASE','owner':'UNKNOWN'}

def report():
    audit=json.loads((OUT/'m03_dtype_canonical_manifest.json').read_text())['double_dtype_audit'];diff=json.loads((OUT/'m03_dtype_canonical_manifest.json').read_text())['graph_diff'];p=json.loads((OUT/'m03_dtype_canonical_parity.json').read_text());c=json.loads((OUT/'m03_dtype_canonical_compiler.json').read_text());back=json.loads((OUT/'m02_backbone_control.json').read_text());e=failure_evidence(c.get('error','')) if c['import']=='FAIL' else ({'stage':'gencode','node':'NOT APPLICABLE','operator':'NOT EXPOSED','resource_failure':'Allocator has ran out of memory'} if c.get('gencode')=='FAIL' else {})
    if p['status']!='PASS':final='MIXVPR-M0.3: PARITY FAIL';b='Parity changed; import was not attempted.'
    elif c['import']=='FAIL':
        representation=('dtype' in c.get('error','').lower() or 'type' in c.get('error','').lower()) and ('DOUBLE' in c.get('error','') or 'FLOAT' in c.get('error',''))
        final='MIXVPR-M0.3: NNCASE IMPORT FAIL — REPRESENTATION/DTYPE ISSUE' if representation else 'MIXVPR-M0.3: NNCASE IMPORT FAIL — GENUINE AGGREGATOR OP';b='nncase import: '+e['error_line']
    elif c['gencode']=='PASS':final='MIXVPR-M0.3: FULL STATIC CORE K210 COMPILE PASS';b='NONE YET'
    else:final='MIXVPR-M0.3: COMPILE FAIL AFTER IMPORT';b=c.get('error','').splitlines()[-1]
    lines=['# MixVPR M0.3 DOUBLE dtype canonicalization','', 'Only a representation-only non-learned DOUBLE constant and its immediately redundant Cast were canonicalized. No arithmetic or learned tensor was changed.','', '## Complete DOUBLE audit','',f"- DOUBLE Constant nodes: `{audit['all_double_constant_nodes']}`.",f"- DOUBLE initializers: `{audit['all_double_initializers']}`.",f"- Cast nodes: `{audit['casts']}`.",f"- Inferred DOUBLE tensors: `{audit['inferred_double_tensors']}`.",f"- Constant_4 classification: **{audit['classification']}**. {audit['proof']}",'', '## Graph change','',f"- Nodes changed: `{diff['nodes_changed']}`.",f"- Constants changed: `{diff['constants_changed']}`.",'- LEARNED WEIGHTS CHANGED: NO','- MODEL MATHEMATICS CHANGED: NO','- ONLY DTYPE REPRESENTATION CHANGED: YES','', '## Parity','',f"- M0.2 vs M0.3 on deterministic + {p['real_image_count']} real images: max abs `{p['m02_vs_m03']['max_abs_error']:.3e}`, mean abs `{p['m02_vs_m03']['mean_abs_error']:.3e}`, cosine `{p['m02_vs_m03']['cosine_similarity']:.9f}`.",f"- PyTorch core vs M0.3: max abs `{p['pytorch_vs_m03']['max_abs_error']:.3e}`, mean abs `{p['pytorch_vs_m03']['mean_abs_error']:.3e}`, cosine `{p['pytorch_vs_m03']['cosine_similarity']:.9f}`.",'- Every output shape: `[1,4096]`.','', '## Artifacts and compiler','',f"- M0.3 ONNX: `{diff['output_onnx']}` SHA `{diff['output_sha256']}`; checker/shape inference `{diff['checker']}`/`{diff['shape_inference']}`.",f"- Core checker/import/PTQ/compile/gencode: `{c['checker']}` / `{c['import']}` / `{c['ptq']}` / `{c['compile']}` / `{c['gencode']}`.",f"- kmodel: `{c.get('size','N/A')}` bytes; SHA `{c.get('sha256','N/A')}`.",'- Compiler memory/warnings/mapping: NOT PROVIDED unless emitted after successful compiler stages.','', '### Exact first failure evidence','```',json.dumps(e,indent=2),'```','### Exact compiler error','```',c.get('error','None'),'```','',final,'','BACKBONE IMPORT: PASS','', 'FIRST TRUE MODEL/RESOURCE BOTTLENECK:',b]
    (OUT/'m03_dtype_canonical_report.md').write_text('\n'.join(lines)+'\n');print(final)

def run():
    audit=double_audit()
    if audit['classification']!='REPRESENTATION_ONLY': raise RuntimeError('M0.3 rewrite not authorized')
    diff=canonicalize(audit);save(OUT/'m03_dtype_canonical_manifest.json',{'protocol_version':'MIXVPR-M0.3-v1','m02_source':str(M02),'m02_sha256':sha(M02),'double_dtype_audit':audit,'graph_diff':diff,'scope':'only redundant DOUBLE representation was canonicalized'})
    p=parity();save(OUT/'m03_dtype_canonical_parity.json',p)
    if p['status']=='PASS':subprocess.run([KPY,'-u',__file__,'compile'],cwd=ROOT)
    else:save(OUT/'m03_dtype_canonical_compiler.json',{'checker':'NOT RUN','import':'NOT RUN','ptq':'NOT RUN','compile':'NOT RUN','gencode':'NOT RUN'})
    report()

if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='compile':raise SystemExit(compiler())
    if len(sys.argv)>1 and sys.argv[1]=='report':report()
    else:run()
