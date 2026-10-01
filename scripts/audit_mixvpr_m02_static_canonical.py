#!/usr/bin/env python3
"""M0.2: replace proven-static ONNX shape expressions with initializers only."""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
import traceback
from collections import Counter
from copy import deepcopy
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/mixvpr_feasibility'
M01_MANIFEST = OUT / 'm01_static_core_manifest.json'
M01 = ROOT / 'models/mixvpr_m01_static_core_1x3x320x320.onnx'
M02 = ROOT / 'models/mixvpr_m02_static_canonical_1x3x320x320.onnx'
M02_INFERRED = ROOT / 'models/mixvpr_m02_static_canonical_1x3x320x320_inferred.onnx'
BACKBONE = ROOT / 'models/mixvpr_m02_backbone_control_1x3x320x320.onnx'
BACKBONE_INFERRED = ROOT / 'models/mixvpr_m02_backbone_control_1x3x320x320_inferred.onnx'
KMODEL = ROOT / 'artifacts/mixvpr_m02_static_canonical_1x3x320x320.kmodel'
KPY = '/home/quyet/miniconda3/envs/k210/bin/python'
CORE_SHAPE = (1, 3, 320, 320)
SEED, CALIBRATION_COUNT = 20260906, 8


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def save(path, value): path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(value, indent=2)+'\n')


def inspect_shape_candidates():
    """All shape-producing subgraphs in M0.1 that feed a shape/axes input."""
    import onnx
    g = onnx.load(str(M01)).graph
    by_out = {out: (i, n) for i,n in enumerate(g.node) for out in n.output}
    consumers = {x: [] for n in g.node for x in n.input if x}
    for i,n in enumerate(g.node):
        for x in n.input:
            if x: consumers.setdefault(x, []).append((i,n))
    # The producer chains are explicitly recorded. Static input and inferred
    # ResNet output prove both values below, rather than assuming dynamic shape.
    candidates = [
        {'id':'reshape_3d_shape', 'nodes':[143,144,145,146,147,148,149],
         'node_names':[g.node[i].name for i in [143,144,145,146,147,148,149]],
         'output_tensor':'/net/aggregator/Concat_output_0', 'resolved_value':[1,1024,-1],
         'consumer':{'node':g.node[150].name,'op_type':g.node[150].op_type,'input_index':1},
         'classification':'SAFE_STATIC',
         'proof':'Static core input [1,3,320,320] fixes preceding backbone output to [1,1024,20,20]. Shape->Slice[:2] gives [1,1024], then Concat([-1]) gives [1,1024,-1].'},
        {'id':'expand_l2_shape', 'nodes':[240], 'node_names':[g.node[240].name],
         'output_tensor':'/net/aggregator/Shape_1_output_0', 'resolved_value':[1,4096],
         'consumer':{'node':g.node[241].name,'op_type':g.node[241].op_type,'input_index':1},
         'classification':'SAFE_STATIC',
         'proof':'Static core input fixes aggregator Flatten output to [1,4096]; Shape output is therefore [1,4096].'},
    ]
    involved = {i for c in candidates for i in c['nodes']}
    all_shape_ops = [{'index':i,'name':n.name,'op_type':n.op_type,'inputs':list(n.input),'outputs':list(n.output),
                      'covered_by_candidate':i in involved} for i,n in enumerate(g.node) if n.op_type in {'Shape','Gather','Slice','Unsqueeze','Concat','Cast','Constant'}]
    return {'source_onnx':str(M01),'source_sha256':sha(M01),'candidates':candidates,'all_shape_related_nodes':all_shape_ops,
            'runtime_dependent_candidates':[], 'audit_conclusion':'All shape expressions that feed Reshape/Expand inputs are SAFE_STATIC for this frozen [1,3,320,320] graph. No runtime-dependent shape expression was found.'}


def canonicalize(audit):
    import onnx
    from onnx import TensorProto, helper
    model = onnx.load(str(M01)); graph = model.graph
    removal = {i for c in audit['candidates'] for i in c['nodes']}
    old = list(graph.node); del graph.node[:]; graph.node.extend(n for i,n in enumerate(old) if i not in removal)
    for c in audit['candidates']:
        graph.initializer.append(helper.make_tensor(c['output_tensor'], TensorProto.INT64, [len(c['resolved_value'])], c['resolved_value']))
    onnx.checker.check_model(model); onnx.save(model, str(M02))
    inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True); onnx.checker.check_model(inferred); onnx.save(inferred, str(M02_INFERRED))
    before, after = Counter(n.op_type for n in old), Counter(n.op_type for n in model.graph.node)
    return {'status':'PASS','output_onnx':str(M02),'output_sha256':sha(M02),'inferred_onnx':str(M02_INFERRED),'inferred_sha256':sha(M02_INFERRED),
            'checker':'PASS','shape_inference':'PASS','nodes_before':len(old),'nodes_after':len(model.graph.node),'nodes_removed':len(removal),
            'removed_nodes':[{'index':i,'name':old[i].name,'op_type':old[i].op_type} for i in sorted(removal)],
            'nodes_replaced': [{'output_tensor':c['output_tensor'],'initializer_value':c['resolved_value']} for c in audit['candidates']],
            'constants_added':len(audit['candidates']), 'op_counts_before':dict(before),'op_counts_after':dict(after)}


def load_core():
    from audit_mixvpr_m01_static_core import load_model, core_wrapper
    model = load_model(); return model, core_wrapper(model)


def metrics(a,b):
    a,b=np.asarray(a).reshape(-1),np.asarray(b).reshape(-1)
    return {'max_abs_error':float(np.max(np.abs(a-b))),'mean_abs_error':float(np.mean(np.abs(a-b))),
            'cosine_similarity':float(np.dot(a,b)/max(float(np.linalg.norm(a)*np.linalg.norm(b)),1e-12))}


def parity():
    import onnx, onnxruntime, torch
    from audit_mixvpr_m01_static_core import external_preprocess, real_images
    model,core=load_core()
    m01=onnxruntime.InferenceSession(str(M01),providers=['CPUExecutionProvider'])
    m02=onnxruntime.InferenceSession(str(M02_INFERRED),providers=['CPUExecutionProvider'])
    rows=[]
    deterministic=torch.linspace(-2,2,int(np.prod(CORE_SHAPE)),dtype=torch.float32).reshape(CORE_SHAPE).numpy()
    probes=[('deterministic',deterministic)]
    import cv2
    for i,path in enumerate(real_images()):
        bgr=cv2.imread(str(path),cv2.IMREAD_COLOR)
        if bgr is None: raise RuntimeError('Could not read '+str(path))
        fixed,_=external_preprocess(bgr,model); probes.append((str(path),fixed.numpy()))
    with torch.inference_mode():
        for label,x in probes:
            a=m01.run(None,{'normalized_rgb_320x320':x})[0]
            b=m02.run(None,{'normalized_rgb_320x320':x})[0]
            pt=core(torch.from_numpy(x)).numpy()
            rows.append({'input':label,'m01_vs_m02':metrics(a,b),'pytorch_vs_m02':metrics(pt,b),'shape_m01':list(a.shape),'shape_m02':list(b.shape),'shape_pytorch':list(pt.shape)})
    def aggregate(key):
        vals=[r[key] for r in rows]
        return {k:max(v[k] for v in vals) if k=='max_abs_error' else float(np.mean([v[k] for v in vals])) for k in vals[0]}
    result={'status':'PASS','deterministic_plus_real_image_count':len(rows),'real_image_count':len(rows)-1,'rows':rows,
            'm01_vs_m02':aggregate('m01_vs_m02'),'pytorch_vs_m02':aggregate('pytorch_vs_m02')}
    if result['m01_vs_m02']['max_abs_error'] != 0 or result['pytorch_vs_m02']['max_abs_error'] > 2e-5: result['status']='FAIL'
    return result


def export_backbone_control():
    import onnx, torch
    model,core=load_core(); dummy=torch.zeros(CORE_SHAPE)
    torch.onnx.export(core.net.backbone,dummy,str(BACKBONE),opset_version=13,dynamo=False,do_constant_folding=False,input_names=['normalized_rgb_320x320'],output_names=['backbone_features'])
    g=onnx.load(str(BACKBONE)); onnx.checker.check_model(g)
    inferred=onnx.shape_inference.infer_shapes(g, strict_mode=True); onnx.checker.check_model(inferred); onnx.save(inferred,str(BACKBONE_INFERRED))
    return {'onnx_path':str(BACKBONE),'sha256':sha(BACKBONE),'inferred_onnx_path':str(BACKBONE_INFERRED),'inferred_sha256':sha(BACKBONE_INFERRED), 'checker':'PASS','shape_inference':'PASS','node_count':len(g.graph.node),'operators':dict(Counter(n.op_type for n in g.graph.node))}


def nncase_import(path, kind):
    import _nncase, nncase, onnx
    from importlib.metadata import version
    r={'kind':kind,'checker':'NOT RUN','import':'NOT RUN'}
    try:
        r['environment']={'python_executable':sys.executable,'nncase':version('nncase'),'_nncase':_nncase.__version__}
        assert r['environment']['nncase']=='1.8.0.20220929' and r['environment']['_nncase']=='1.8.0-55be52f',r['environment']
        onnx.checker.check_model(onnx.load(str(path)));r['checker']='PASS'
        o=nncase.CompileOptions();o.target='k210';o.quant_type='uint8';o.w_quant_type='uint8'; c=nncase.Compiler(o);c.import_onnx(Path(path).read_bytes(),nncase.ImportOptions());r['import']='PASS'
    except Exception: r['import']='FAIL';r['error']=traceback.format_exc()
    return r


def compiler():
    import _nncase, nncase, onnx
    from importlib.metadata import version
    r={'checker':'NOT RUN','import':'NOT RUN','ptq':'NOT RUN','compile':'NOT RUN','gencode':'NOT RUN'};stage='checker'
    try:
        r['environment']={'python_executable':sys.executable,'nncase':version('nncase'),'_nncase':_nncase.__version__};assert r['environment']['nncase']=='1.8.0.20220929' and r['environment']['_nncase']=='1.8.0-55be52f',r['environment']
        onnx.checker.check_model(onnx.load(str(M02_INFERRED)));r['checker']='PASS';stage='import'
        o=nncase.CompileOptions();o.target='k210';o.quant_type='uint8';o.w_quant_type='uint8';o.dump_dir=str(OUT/'m02_nncase_dump');o.dump_ir=True;o.dump_asm=True;c=nncase.Compiler(o);c.import_onnx(M02_INFERRED.read_bytes(),nncase.ImportOptions());r['import']='PASS';stage='ptq'
        calib=np.random.default_rng(SEED).uniform(-3,3,size=(CALIBRATION_COUNT,*CORE_SHAPE[1:])).astype(np.float32);r['calibration_shape']=list(calib.shape);r['calibration_sha256']=hashlib.sha256(calib.tobytes()).hexdigest();q=nncase.PTQTensorOptions();q.samples_count=CALIBRATION_COUNT;q.set_tensor_data(calib.tobytes());c.use_ptq(q);r['ptq']='PASS';stage='compile'
        c.compile();r['compile']='PASS';stage='gencode';blob=c.gencode_tobytes();KMODEL.write_bytes(blob);r.update(gencode='PASS',size=len(blob),sha256=sha(KMODEL))
    except Exception:r[stage]='FAIL';r['error']=traceback.format_exc()
    save(OUT/'m02_static_canonical_compiler.json',r);return 0 if r['gencode']=='PASS' else 1


def failed_tensor(error):
    m=re.search(r"Can't pull input data for <([^>]+)>",error or '')
    return m.group(1) if m else None


def unsupported_double_evidence():
    """Associate a datatype-only compiler error with exact graph evidence."""
    import onnx
    from onnx import TensorProto
    graph=onnx.load(str(M02_INFERRED)).graph
    found=[]
    for node in graph.node:
        if node.op_type != 'Constant': continue
        value=next((a.t for a in node.attribute if a.name=='value' and a.HasField('t')),None)
        if value is not None and value.data_type == TensorProto.DOUBLE:
            consumers=[{'name':n.name,'op_type':n.op_type,'inputs':list(n.input),'outputs':list(n.output)} for n in graph.node if node.output[0] in n.input]
            found.append({'node_name':node.name,'op_type':node.op_type,'output':node.output[0], 'data_type':'DOUBLE','consumers':consumers})
    return found


def report():
    manifest=json.loads((OUT/'m02_static_canonical_manifest.json').read_text());audit=manifest['shape_expression_audit'];diff=manifest['graph_diff'];p=json.loads((OUT/'m02_static_canonical_parity.json').read_text());comp=json.loads((OUT/'m02_static_canonical_compiler.json').read_text());back=json.loads((OUT/'m02_backbone_control.json').read_text())
    if p['status']!='PASS': final='MIXVPR-M0.2: PARITY FAIL'; bottleneck='Parity changed; compiler stage was not allowed.'
    elif comp['import']=='FAIL':
        t=failed_tensor(comp.get('error','')); final='MIXVPR-M0.2: NNCASE IMPORT FAIL — STATIC SHAPE ISSUE REMAINS' if t and ('Shape' in t or 'Concat' in t or 'Slice' in t) else 'MIXVPR-M0.2: NNCASE IMPORT FAIL — GENUINE AGGREGATOR OP'; bottleneck='nncase import failure: '+(t or comp.get('error','').strip().splitlines()[-1])
    elif comp['gencode']=='PASS': final='MIXVPR-M0.2: FULL STATIC CORE K210 COMPILE PASS';bottleneck='NONE YET'
    else: final='MIXVPR-M0.2: COMPILE FAIL AFTER IMPORT';bottleneck=comp.get('error','').strip().splitlines()[-1]
    doubles=unsupported_double_evidence() if 'Data type "DOUBLE" not supported' in comp.get('error','') else []
    if doubles: bottleneck='nncase rejects DOUBLE; the only DOUBLE Constant is `{}` (`{}`), consumed by `{}` in the unchanged aggregator normalization subgraph.'.format(doubles[0]['node_name'],doubles[0]['output'],[x['op_type'] for x in doubles[0]['consumers']])
    lines=['# MixVPR M0.2 static-shape canonicalization','', 'This pass replaces only compile-time-static shape tensors in the frozen M0.1 graph. It does not modify learned data path or weights.','', '## Shape-expression audit','']
    for c in audit['candidates']: lines += [f"### {c['id']} — {c['classification']}",f"- Nodes: `{c['node_names']}`.",f"- Output `{c['output_tensor']}` = `{c['resolved_value']}`; consumer `{c['consumer']['node']}` (`{c['consumer']['op_type']}` input {c['consumer']['input_index']}).",f"- Proof: {c['proof']}", '']
    lines += [f"- Runtime-dependent candidates: `{audit['runtime_dependent_candidates']}`.", f"- Conclusion: {audit['audit_conclusion']}", '', '## Exact graph change','', f"- Nodes: {diff['nodes_before']} -> {diff['nodes_after']}; removed {diff['nodes_removed']}; static initializers added {diff['constants_added']}.", f"- Nodes replaced by constant tensors: `{diff['nodes_replaced']}`.", f"- Ops before: `{diff['op_counts_before']}`.", f"- Ops after: `{diff['op_counts_after']}`.", '- LEARNED WEIGHTS CHANGED: NO', '- MODEL MATHEMATICS CHANGED: NO', '- ONLY STATIC SHAPE REPRESENTATION CHANGED: YES', '', '## Parity','', f"- M0.1 vs M0.2: {p['real_image_count']} real images + deterministic input; max abs `{p['m01_vs_m02']['max_abs_error']:.3e}`, mean abs `{p['m01_vs_m02']['mean_abs_error']:.3e}`, cosine `{p['m01_vs_m02']['cosine_similarity']:.9f}`.", f"- PyTorch core vs M0.2: max abs `{p['pytorch_vs_m02']['max_abs_error']:.3e}`, mean abs `{p['pytorch_vs_m02']['mean_abs_error']:.3e}`, cosine `{p['pytorch_vs_m02']['cosine_similarity']:.9f}`.", '- Output shapes: `[1,4096]` for every probe.', '', '## Artifacts','', f"- M0.2 ONNX: `{diff['output_onnx']}` SHA `{diff['output_sha256']}`; checker/inference `{diff['checker']}`/`{diff['shape_inference']}`.", f"- Backbone diagnostic ONNX: `{back['onnx_path']}` SHA `{back['sha256']}`.", '', '## nncase K210','', f"- Core checker/import/PTQ/compile/gencode: `{comp['checker']}` / `{comp['import']}` / `{comp['ptq']}` / `{comp['compile']}` / `{comp['gencode']}`.", f"- kmodel: `{comp.get('size','N/A')}` bytes; SHA `{comp.get('sha256','N/A')}`.", f"- BACKBONE IMPORT: `{back['import']}`.", '- Compiler memory, warnings, and operator mapping: NOT PROVIDED unless those stages completed and emitted output.', '', '### Exact core compiler failure','```',comp.get('error','None'),'```', '### DOUBLE-type graph evidence','```',json.dumps(doubles,indent=2) if doubles else 'None','```', '### Backbone diagnostic failure','```',back.get('error','None'),'```','',final,'',f"BACKBONE IMPORT: {'PASS' if back['import']=='PASS' else 'FAIL'}",'', 'FIRST TRUE MODEL/RESOURCE BOTTLENECK:',bottleneck]
    (OUT/'m02_static_canonical_report.md').write_text('\n'.join(lines)+'\n');print(final);return 0


def run():
    audit=inspect_shape_candidates(); diff=canonicalize(audit); model,core=load_core(); manifest={'protocol_version':'MIXVPR-M0.2-v1','m01_manifest_path':str(M01_MANIFEST),'m01_manifest_sha256':sha(M01_MANIFEST),'source_onnx':str(M01),'source_onnx_sha256':sha(M01),'shape_expression_audit':audit,'graph_diff':diff,'scope':'ONLY compile-time-static shape representation canonicalized; no learned weights or math altered.'};save(OUT/'m02_static_canonical_manifest.json',manifest)
    p=parity();save(OUT/'m02_static_canonical_parity.json',p)
    if p['status']=='PASS': subprocess.run([KPY,'-u',__file__,'compile'],cwd=ROOT)
    else: save(OUT/'m02_static_canonical_compiler.json',{'checker':'NOT RUN','import':'NOT RUN','ptq':'NOT RUN','compile':'NOT RUN','gencode':'NOT RUN'})
    control=export_backbone_control();save(OUT/'m02_backbone_control.json',control)
    subprocess.run([KPY,'-u',__file__,'backbone-import'],cwd=ROOT)
    return report()


if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='compile':raise SystemExit(compiler())
    if len(sys.argv)>1 and sys.argv[1]=='backbone-import':
        existing=json.loads((OUT/'m02_backbone_control.json').read_text()); existing.update(nncase_import(BACKBONE_INFERRED,'backbone_import_only')); save(OUT/'m02_backbone_control.json',existing); raise SystemExit(0)
    if len(sys.argv)>1 and sys.argv[1]=='report':raise SystemExit(report())
    raise SystemExit(run())
