from diagnose import *
OUT=Path(__file__).parent
def write(name,x):
    t=OUT/(name+'.tmp');t.write_text(json.dumps(x,indent=2)+'\n');t.replace(OUT/name)
@torch.inference_mode()
def main():
    deployment=json.loads((OUT/'deployment.json').read_text());assert deployment['simulation']=='PASS'
    export=json.loads((OUT/'export.json').read_text());assert deployment['checkpoint_sha256']==export['checkpoint']['sha256']
    assert digest(OUT/'pilot_trained.kmodel')==deployment['kmodel_sha256']
    model=p.w0.V1FR12().eval();model.load_state_dict(torch.load(export['checkpoint']['path'],map_location='cpu',weights_only=False)['model'])
    protocol=json.loads((BASE/'protocol.json').read_text());evidence=json.loads((OUT/'simulation_manifest.json').read_text());rows=[]
    for i,r in enumerate(protocol['DEV']):
        source=BASE/'dev'/('%03d.npz'%i);out=OUT/'sim_outputs'/('%03d.npz'%i)
        assert digest(source)==evidence[i]['input_cache_sha256'] and digest(out)==evidence[i]['output_sha256']
        with np.load(source) as zz:z={k:zz[k] for k in zz.files}
        a,b=model(torch.from_numpy(z['pair']))
        with np.load(out) as q:qa=torch.from_numpy(q['logits']);qb=torch.from_numpy(q['desc'])
        fm=metrics(a,b,z);qm=metrics(qa,qb,z)
        dense_cos=F.cosine_similarity(b,qb,dim=1)
        rows.append({'index':i,'scene':r['scene'],'fp32':fm,'int8':qm,'descriptor_cosine_mean':float(dense_cos.mean()),'descriptor_cosine_min':float(dense_cos.min()),'detector_probability_mae':float((a.softmax(1)-qa.softmax(1)).abs().mean())})
    def avg(rs,k):return {m:float(np.mean([r[k][m] for r in rs])) for m in rs[0][k]}
    fp=avg(rows,'fp32');quant=avg(rows,'int8')
    ret={k:quant[k]/fp[k] if fp[k]>0 else None for k in ('mnn_precision','repeatability_3px','correct_mnn_count')}
    result={'status':'MEASURED','backend':deployment['backend'],'fp32':fp,'int8':quant,'retention':ret,'descriptor_cosine_mean':float(np.mean([r['descriptor_cosine_mean'] for r in rows])),
        'per_scene':{s:{k:avg([r for r in rows if r['scene']==s],k) for k in ('fp32','int8')} for s in sorted({r['scene'] for r in rows})},
        'scope':'28 existing DEV homography pairs; actual trained pilot K210 kmodel; not independent test or board benchmark',
        'quality_acceptable':'NOT_CLAIMED: FP32 pilot remains weak; retention is separate from absolute readiness'}
    write('int8_quality_rows.json',rows);write('int8_quality.json',result)
    deployment['INT8_QUALITY']='MEASURED';write('deployment.json',deployment)
    print(json.dumps({'fp32_precision':fp['mnn_precision'],'int8_precision':quant['mnn_precision'],'retention':ret,'descriptor_cosine':result['descriptor_cosine_mean']}),flush=True)
if __name__=='__main__':main()
