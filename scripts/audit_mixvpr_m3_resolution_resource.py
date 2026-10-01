#!/usr/bin/env python3
"""M3: resource-only static-resolution sweep for the frozen MixVPR core.

No learned source-project state is modified.  Lower-resolution aggregators are
explicit compiler surrogates: spatially invariant tensors are copied exactly,
while spatial-token-dependent tensors receive deterministic initialization.
"""
from __future__ import annotations
import hashlib, json, sys, traceback
from collections import Counter
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/mixvpr_feasibility'
MODELS, ART = ROOT / 'models', ROOT / 'artifacts'
SEED, N = 20260906, 8
KPY = '/home/quyet/miniconda3/envs/k210/bin/python'
BASE = 320

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p, x):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    Path(p).write_text(json.dumps(x, indent=2) + '\n', encoding='utf-8')
def load_manifest():
    p = OUT / 'm3_resolution_resource_manifest.json'
    return json.loads(p.read_text()) if p.exists() else {'protocol_version':'MIXVPR-M3-v1','seed':SEED,'scope':'RESOURCE FEASIBILITY ONLY; no training, retrieval-quality evaluation, source-project change, or learned-weight resizing.','baseline_320_reused':True,'entries':{}}
def graph_facts(path):
    import onnx
    m=onnx.load(str(path)); return {'path':str(path),'sha256':sha(path),'nodes':len(m.graph.node),'operators':dict(sorted(Counter(x.op_type for x in m.graph.node).items()))}
def tensors(module, dummy):
    import torch
    seen=[]; hs=[]
    def hook(name):
        def f(_m,_i,o):
            if torch.is_tensor(o): seen.append({'name':name,'shape':list(o.shape),'elements':o.numel(),'fp32_logical_bytes':o.numel()*4,'projected_int8_bytes':o.numel()})
        return f
    for name, sub in module.named_modules():
        if name and not list(sub.children()): hs.append(sub.register_forward_hook(hook(name)))
    with torch.inference_mode(): out=module(dummy)
    for h in hs:h.remove()
    seen += [{'name':'graph_input','shape':list(dummy.shape),'elements':dummy.numel(),'fp32_logical_bytes':dummy.numel()*4,'projected_int8_bytes':dummy.numel()}]
    return out, sorted(seen,key=lambda x:x['fp32_logical_bytes'],reverse=True)
def canonicalize(src, dst, inf):
    # Reuse the M0.2/M0.3-proven representation-only canonicalization.
    from audit_mixvpr_m1_resource_localization import canonicalize as c
    return c(src,dst,inf)
def mixvpr_class():
    import importlib.util
    path=Path('/home/quyet/.cache/torch/hub/jarvisyjw_MixVPR_main/models/aggregators/mixvpr.py')
    spec=importlib.util.spec_from_file_location('m3_mixvpr',path); mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod.MixVPR
def prepare(resolutions):
    import onnx, onnxruntime, torch
    import torch.nn as nn
    from audit_mixvpr_m01_static_core import load_model, scalar_metrics
    model=load_model(); backbone=model.net.backbone.eval(); original=model.net.aggregator.eval(); MixVPR=mixvpr_class()
    man=load_manifest(); man.update({'seed':SEED,'frozen_preprocessing_semantics':'Unchanged M0.1 normalized RGB FP32 external preprocessing; only static learned-core spatial size differs.','surrogate_label':'RESOURCE_SURROGATE_ONLY','original_320_aggregator_parameters':sum(p.numel() for p in original.parameters()),'descriptor_dimension':4096})
    class Full(nn.Module):
        def __init__(self,b,a): super().__init__(); self.backbone=b; self.aggregator=a
        def forward(self,x): return self.aggregator(self.backbone(x))
    for r in resolutions:
        if r == 320: continue
        s=r//16; feature=(1,1024,s,s); inp=(1,3,r,r)
        dummy=torch.linspace(-2,2,int(np.prod(inp)),dtype=torch.float32).reshape(inp)
        # ONNX tracing needs a normal (not inference-mode) tensor as its input.
        with torch.no_grad(): feature_value=backbone(dummy)
        if list(feature_value.shape)!=list(feature): raise RuntimeError(f'Unexpected backbone shape at {r}: {list(feature_value.shape)}')
        torch.manual_seed(SEED)
        agg=MixVPR(in_channels=1024,in_h=s,in_w=s,out_channels=1024,mix_depth=4,mlp_ratio=1,out_rows=4).eval()
        old,new=original.state_dict(),agg.state_dict(); reused=[]; initialized=[]
        for k,v in new.items():
            if k in old and tuple(old[k].shape)==tuple(v.shape):
                new[k]=old[k].detach().clone(); reused.append({'name':k,'shape':list(v.shape),'parameters':v.numel()})
            else: initialized.append({'name':k,'original_shape':list(old[k].shape) if k in old else None,'candidate_shape':list(v.shape),'parameters':v.numel(),'reason':'spatial_token_count_changed; deterministic seed '+str(SEED)})
        agg.load_state_dict(new)
        # Inputs whose spatially dependent tensors differ must not be construed as quality models.
        braw=MODELS/f'mixvpr_m3_backbone_{r}x{r}.onnx'; b=MODELS/f'mixvpr_m3_backbone_{r}x{r}_inferred.onnx'
        araw=MODELS/f'mixvpr_m3_aggregator_surrogate_{r}x{r}.onnx'; a=MODELS/f'mixvpr_m3_aggregator_surrogate_{r}x{r}_inferred.onnx'
        fraw=MODELS/f'mixvpr_m3_full_surrogate_{r}x{r}.onnx'; f=MODELS/f'mixvpr_m3_full_surrogate_{r}x{r}_inferred.onnx'
        torch.onnx.export(backbone,dummy,str(braw),opset_version=13,dynamo=False,do_constant_folding=False,input_names=[f'normalized_rgb_{r}x{r}'],output_names=['backbone_features'])
        torch.onnx.export(agg,feature_value,str(araw),opset_version=13,dynamo=False,do_constant_folding=False,input_names=['backbone_features'],output_names=['global_descriptor'])
        torch.onnx.export(Full(backbone,agg).eval(),dummy,str(fraw),opset_version=13,dynamo=False,do_constant_folding=False,input_names=[f'normalized_rgb_{r}x{r}'],output_names=['global_descriptor'])
        changes={'backbone':canonicalize(braw,braw,b),'aggregator':canonicalize(araw,araw,a),'full':canonicalize(fraw,fraw,f)}
        # Export correctness only, never retrieval-quality evaluation.
        ort_b=onnxruntime.InferenceSession(str(b),providers=['CPUExecutionProvider']).run(None,{f'normalized_rgb_{r}x{r}':dummy.numpy()})[0]
        with torch.inference_mode(): pt_b=backbone(dummy).numpy(); pt_a=agg(feature_value).numpy()
        ort_a=onnxruntime.InferenceSession(str(a),providers=['CPUExecutionProvider']).run(None,{'backbone_features':feature_value.numpy()})[0]
        out_b,act_b=tensors(backbone,dummy); out_a,act_a=tensors(agg,feature_value)
        params=lambda x:sum(z.numel() for z in x.parameters())
        e={'resolution':r,'input_shape':list(inp),'feature_shape':list(feature),'spatial_tokens':s*s,
           'backbone':{'parameters':params(backbone),'output_shape':list(out_b.shape),'activations':act_b,'largest_logical_activation':act_b[0],
                       'stem_output_shape':[1,64,r//4,r//4],'layer1_output_shape':[1,256,r//4,r//4],'layer2_output_shape':[1,512,r//8,r//8],'layer3_output_shape':list(out_b.shape),
                       'layer1_projected_int8_logical_bytes':256*(r//4)*(r//4),'inferred_onnx':str(b),'onnx':graph_facts(b),'ort_export_parity':scalar_metrics(pt_b,ort_b)},
           'aggregator':{'label':'RESOURCE_SURROGATE_ONLY','parameters':params(agg),'mix0_parameters':params(agg.mix[0]),'output_shape':list(out_a.shape),'activations':act_a,'largest_logical_activation':act_a[0],
                         'inferred_onnx':str(a),'onnx':graph_facts(a),'ort_export_parity':scalar_metrics(pt_a,ort_a),'weight_transfer':{'reused_unchanged':reused,'deterministically_initialized_shape_changed':initialized,'learned_weights_resized':'NO'}},
           'full':{'label':'RESOURCE_SURROGATE_ONLY','parameters':params(backbone)+params(agg),'input_shape':list(inp),'output_shape':[1,4096],'inferred_onnx':str(f),'onnx':graph_facts(f)},
           'canonicalization_changes':changes}
        for group in ('backbone','aggregator'):
            if e[group]['ort_export_parity']['max_abs_error']>3e-5: raise RuntimeError(f'ONNX export parity failed {r} {group}')
        man['entries'][str(r)]=e
    save(OUT/'m3_resolution_resource_manifest.json',man); print('M3 PREPARE PASS',sorted(man['entries']))
def compiler(kind,res):
    import _nncase, nncase, onnx
    from importlib.metadata import version
    man=load_manifest(); entry=man['entries'][str(res)]; e=entry[kind]; path=Path(e['inferred_onnx'])
    # PTQ samples must match the graph input, not the backbone→aggregator feature.
    shape=entry['input_shape'] if kind in ('backbone','full') else entry['feature_shape']
    result={'resolution':res,'kind':kind,'label':e.get('label','ORIGINAL_PRETRAINED_BACKBONE'),'checker':'NOT RUN','shape_inference':'PREVIOUSLY PASS','import':'NOT RUN','ptq':'NOT RUN','compile':'NOT RUN','gencode':'NOT RUN','compiler_memory_information':'NOT PROVIDED BY nncase Python API/compiler output'}; stage='checker'
    try:
        result['environment']={'nncase':version('nncase'),'_nncase':_nncase.__version__,'python':sys.executable}; assert result['environment']['nncase']=='1.8.0.20220929' and result['environment']['_nncase']=='1.8.0-55be52f'
        onnx.checker.check_model(onnx.load(str(path)));result['checker']='PASS';stage='import'; opt=nncase.CompileOptions();opt.target='k210';opt.quant_type='uint8';opt.w_quant_type='uint8';opt.dump_dir=str(OUT/f'm3_{kind}_{res}_nncase_dump');opt.dump_ir=True;opt.dump_asm=True
        c=nncase.Compiler(opt);c.import_onnx(path.read_bytes(),nncase.ImportOptions());result['import']='PASS';stage='ptq';cal=np.random.default_rng(SEED).uniform(-3,3,size=(N,*shape[1:])).astype(np.float32);q=nncase.PTQTensorOptions();q.samples_count=N;q.set_tensor_data(cal.tobytes());result['calibration_shape']=list(cal.shape);result['calibration_sha256']=hashlib.sha256(cal.tobytes()).hexdigest();c.use_ptq(q);result['ptq']='PASS';stage='compile';c.compile();result['compile']='PASS';stage='gencode';blob=c.gencode_tobytes(); kp=ART/f'mixvpr_m3_{kind}_{res}x{res}.kmodel';kp.write_bytes(blob);result.update(gencode='PASS',kmodel_path=str(kp),kmodel_bytes=len(blob),sha256=sha(kp))
    except Exception:
        result[stage]='FAIL';result['error']=traceback.format_exc()
    save(OUT/f'm3_{kind}_{res}_compiler.json',result); print(f'M3 {kind} {res}: {result["gencode"]}'); return 0 if result['gencode']=='PASS' else 1
def comp(kind,res):
    p=OUT/f'm3_{kind}_{res}_compiler.json'
    return json.loads(p.read_text()) if p.exists() else {'gencode':'NOT RUN'}
def legacy():
    m1=json.loads((OUT/'m1_resource_localization_manifest.json').read_text());return {'resolution':320,'input_shape':[1,3,320,320],'feature_shape':[1,1024,20,20],'spatial_tokens':400,
      'backbone':{'parameters':m1['backbone']['parameters'],'stem_output_shape':[1,64,80,80],'layer1_output_shape':[1,256,80,80],'layer2_output_shape':[1,512,40,40],'layer3_output_shape':[1,1024,20,20],'layer1_projected_int8_logical_bytes':256*80*80,'largest_logical_activation':'See M1 frozen audit'},
      'aggregator':{'parameters':m1['aggregator']['parameters'],'mix0_parameters':321600,'label':'ORIGINAL_PRETRAINED_AGGREGATOR'},'full':{'label':'ORIGINAL_PRETRAINED_FULL_CORE'},'compiler':{'backbone':json.loads((OUT/'m1_backbone_compiler.json').read_text()),'aggregator':json.loads((OUT/'m1_aggregator_compiler.json').read_text()),'full':json.loads((OUT/'m03_dtype_canonical_compiler.json').read_text())}}
def report():
    man=load_manifest(); rows=[legacy()]
    for r in sorted(map(int,man['entries'])):
        e=man['entries'][str(r)]; e['compiler']={k:comp(k,r) for k in ('backbone','aggregator','full')}
        if e['compiler']['full']['gencode']=='NOT RUN': e['compiler']['full']['reason']='Not run: M3 permits full graph only when both independently gencode PASS.'
        rows.append(e)
    def state(x):return x.get('gencode','NOT RUN')
    def high(kind):
        vals=[x['resolution'] for x in rows if state(x['compiler'][kind])=='PASS'];return f'{max(vals)}x{max(vals)}' if vals else 'NONE'
    bp,ap,fp=high('backbone'),high('aggregator'),high('full')
    if fp!='NONE': classification='RESOLUTION HAS ESTABLISHED A FEASIBLE RESOURCE REGION'; sufficient='YES'; next_change='NONE'
    elif bp!='NONE' and ap=='NONE': classification='NEXT BOTTLENECK = AGGREGATOR STRUCTURE';sufficient='NO';next_change='AGGREGATOR SLIMMING'
    elif ap!='NONE' and bp=='NONE': classification='NEXT BOTTLENECK = BACKBONE ACTIVATION / ARCHITECTURE';sufficient='NO';next_change='BACKBONE SLIMMING'
    elif bp!='NONE' and ap!='NONE': classification='NEXT BOTTLENECK = FULL-GRAPH RESOURCE COMBINATION';sufficient='NO';next_change='BOTH'
    else: classification='NO USEFUL INDEPENDENT PASS BOUNDARY ESTABLISHED';sufficient='INDETERMINATE';next_change='UNKNOWN'
    lines=['# MixVPR M3 resource-only resolution sweep','', '**Scope:** compiler/resource feasibility only. No retrieval quality was evaluated. Lower-resolution aggregators are `RESOURCE_SURROGATE_ONLY`; spatial-token-dependent learned tensors were never resized.', '', '## Static-shape and surrogate audit','', '- Original pretrained ResNet50 cropped backbone weights are reused exactly at every resolution.', '- The frozen preprocessing semantics remain normalized RGB FP32 external preprocessing; only learned-core static spatial shape changes.', f'- Deterministic surrogate initialization seed: `{SEED}`. Tensors with unchanged shapes reuse their original pretrained values; changed token-dimension tensors are deterministically initialized and listed in the manifest.', '- “Logical” tensor bytes are tensor-size arithmetic, **not** measured KPU RAM / compiler peak memory.', '', '### Aggregator parameter-shape audit','', '| Resolution | Tokens (original → candidate) | Original mix.0 parameters → candidate | Exact unchanged tensors reused | Shape-dependent tensors not reused |','|---|---|---:|---|---|']
    for x in sorted(rows,key=lambda q:q['resolution'],reverse=True):
        if x['resolution']==320:
            lines.append('| 320x320 | 400 → 400 | 321,600 → 321,600 | all original tensors | none |');continue
        w=x['aggregator']['weight_transfer']; reuse=', '.join(y['name'] for y in w['reused_unchanged']); init=', '.join(y['name'] for y in w['deterministically_initialized_shape_changed']); lines.append(f"| {x['resolution']}x{x['resolution']} | 400 → {x['spatial_tokens']} | 321,600 → {x['aggregator']['mix0_parameters']:,} | `{reuse}` | `{init}` |")
    lines += ['', '## Resource scaling table','', '| Resolution | Backbone output | Tokens | Layer1 INT8 logical bytes | Backbone gencode | Aggregator params | mix.0 params | Aggregator gencode | Full candidate gencode |','|---|---|---:|---:|---|---:|---:|---|---|']
    for x in sorted(rows,key=lambda q:q['resolution'],reverse=True):
        c=x['compiler'];lines.append(f"| {x['resolution']}x{x['resolution']} | `{x['feature_shape']}` | {x['spatial_tokens']:,} | {x['backbone']['layer1_projected_int8_logical_bytes']:,} | {state(c['backbone'])} | {x['aggregator']['parameters']:,} | {x['aggregator']['mix0_parameters']:,} | {state(c['aggregator'])} | {state(c['full'])} |")
    lines += ['', '## Per-resolution compiler record','']
    for x in sorted(rows,key=lambda q:q['resolution'],reverse=True):
        c=x['compiler'];lines += [f"### {x['resolution']}x{x['resolution']}",f"- Backbone stage: checker {c['backbone'].get('checker','NOT RUN')}; import {c['backbone'].get('import','NOT RUN')}; PTQ {c['backbone'].get('ptq','NOT RUN')}; compile {c['backbone'].get('compile','NOT RUN')}; gencode {state(c['backbone'])}.",f"- Aggregator stage: checker {c['aggregator'].get('checker','NOT RUN')}; import {c['aggregator'].get('import','NOT RUN')}; PTQ {c['aggregator'].get('ptq','NOT RUN')}; compile {c['aggregator'].get('compile','NOT RUN')}; gencode {state(c['aggregator'])}.",f"- Full resource surrogate: checker {c['full'].get('checker','NOT RUN')}; import {c['full'].get('import','NOT RUN')}; PTQ {c['full'].get('ptq','NOT RUN')}; compile {c['full'].get('compile','NOT RUN')}; gencode {state(c['full'])}."]
        if c['full'].get('reason'): lines.append(f"- Full candidate: {c['full']['reason']}")
        if c['backbone'].get('compiler_memory_information') and c['backbone']['compiler_memory_information']!='NOT PROVIDED BY nncase Python API/compiler output': lines.append(f"- Backbone compiler MEMORY USAGES: `{c['backbone']['compiler_memory_information']}`.")
        for k in ('backbone','aggregator','full'):
            if c[k].get('error'): lines.append(f"- {k} first failure: `{c[k]['error'].strip().splitlines()[-1]}`")
    lines += ['', '## Result', '', 'MIXVPR-M3 RESOURCE RESULT:', classification, '', 'HIGHEST BACKBONE PASS RESOLUTION:',bp,'', 'HIGHEST AGGREGATOR PASS RESOLUTION:',ap,'', 'HIGHEST FULL RESOURCE PASS RESOLUTION:',fp,'', 'RESOLUTION ALONE APPEARS SUFFICIENT:',sufficient,'', 'NEXT STRUCTURAL CHANGE REQUIRED:',next_change,'', 'TRAINING OR DISTILLATION SHOULD START NOW:','NO']
    (OUT/'m3_resolution_resource_report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8'); man['summary']={'classification':classification,'highest_backbone_pass_resolution':bp,'highest_aggregator_pass_resolution':ap,'highest_full_resource_pass_resolution':fp,'resolution_alone_appears_sufficient':sufficient,'next_structural_change_required':next_change,'training_or_distillation_should_start_now':'NO'};save(OUT/'m3_resolution_resource_manifest.json',man);print(classification)
if __name__=='__main__':
    if len(sys.argv)<2: raise SystemExit('usage: prepare [res ...] | compiler KIND RES | report')
    if sys.argv[1]=='prepare': prepare([int(x) for x in sys.argv[2:]])
    elif sys.argv[1]=='compiler': raise SystemExit(compiler(sys.argv[2],int(sys.argv[3])))
    elif sys.argv[1]=='report': report()
    else: raise SystemExit('unknown action')
