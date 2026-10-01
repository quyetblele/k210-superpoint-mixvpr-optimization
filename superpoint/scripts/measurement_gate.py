"""Frozen diagnostic gate. No training, deployment promotion or legacy writes."""
import _bootstrap
import argparse
import csv
import json
import subprocess
from pathlib import Path
import numpy as np
import torch.nn.functional as F
from spk210.runtime import torch, ops, w0, SuperPointKPUWrapper
from spk210.io import sha, put
from spk210.settings import ROOT, SUPERPOINT, WORKSPACE
from spk210.protocol import protocol, verify_cache
from spk210.onnx_export import load_model, deployment_dir
from spk210.metrics import evaluate, features, pair_metrics, project, verify


def read(p):
    return json.loads(Path(p).read_text())


def checked(p, digest):
    actual = sha(p)
    if actual != digest:
        raise RuntimeError(f'Identity mismatch: {p}')
    return actual


def git(repo, *args):
    return subprocess.check_output(['git', '-C', str(repo), *args], text=True).strip()


def write_csv(path, rows):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@torch.inference_mode()
def run(out):
    out.mkdir(parents=True, exist_ok=False)
    p = protocol()
    verify_cache()
    model, res, checkpoint = load_model('wider_r2_cw')
    dep = deployment_dir('wider_r2_cw')
    sim, compile_spec = read(dep/'simulation.json'), read(dep/'compile.json')
    export, calibration = read(dep/'export.json'), read(dep/'calibration.json')
    checked(checkpoint, sim['checkpoint_sha256'])
    checked(dep/'model.kmodel', sim['kmodel_sha256'])
    checked(dep/'canonical.onnx', compile_spec['onnx_sha256'])
    checked(dep/'calibration_train.npy', compile_spec['calibration_sha256'])
    assert calibration['split'] == 'TRAIN_ONLY' and sim['status'] == 'PASS'
    for item in calibration['manifest']:
        source = p['rows']['train'][item['index']]
        checked(source['path'], item['source_sha256'])
        checked(ROOT/res/'train'/f"{item['index']:03d}.npz", item['cache_sha256'])
    for row in sim['rows']:
        checked(ROOT/res/'dev'/f"{row['index']:03d}.npz", row['input_sha256'])
        checked(dep/'sim_outputs'/f"{row['index']:03d}.npz", row['output_sha256'])
    assert len(sim['rows']) == len(p['rows']['dev']) == 42
    teacher, _ = w0.import_original_superpoint()
    raw_teacher = SuperPointKPUWrapper(teacher).eval()
    checked(w0.CHECKPOINT, w0.EXPECTED_CHECKPOINT_SHA256)
    source_files = sorted((SUPERPOINT/'src/spk210').glob('*.py')) + [Path(__file__), Path(w0.__file__), Path(ops.__file__)]
    manifest = {
        'candidate': 'wider_r2_cw', 'architecture': {'widths':[24,32,64,96], 'head_hidden':128, 'detector_channels':65, 'descriptor_dim':256},
        'checkpoint': str(checkpoint), 'checkpoint_sha256': sha(checkpoint),
        'teacher_checkpoint': str(w0.CHECKPOINT), 'teacher_sha256': sha(w0.CHECKPOINT),
        'git_head': git(WORKSPACE,'rev-parse','HEAD'), 'git_status': git(WORKSPACE,'status','--short'),
        'source_hashes': {str(f): sha(f) for f in source_files},
        'upstream_commit': git(w0.EDGE/'third_party/SuperGluePretrainedNetwork','rev-parse','HEAD'),
        'upstream_worktree_status': git(w0.EDGE/'third_party/SuperGluePretrainedNetwork','status','--short'),
        'protocol_sha256': sha(ROOT/'protocol.json'), 'export': export, 'calibration': calibration,
        'compile': compile_spec, 'simulation_sha256': sha(dep/'simulation.json'),
        'preprocessing': export['preprocessing'], 'postprocessing': export['CPU_postprocessing'],
        'logical_WH':[184,320], 'deployment_NCHW': export['input'],
        'dataset_roles': {'train_sources':len(p['rows']['train']), 'dev_sources':42, 'test':'NOT YET ESTABLISHED',
                          'scope':'fixed synthetic pairs on 7-Scenes/project DEV; no public benchmark claim'},
        'python_packages': {'torch':torch.__version__, 'numpy':np.__version__},
        'reuse':'actual simulator outputs rehashed, no new compile/calibration; no fake quant',
        'board_performance':'NOT_MEASURED'}
    put(out/'manifest.json', manifest)
    put(out/'protocol.json', {
        'scope':'bounded engineering DEV diagnostic, no public benchmark reproduction',
        'feature_comparison':'42 fixed pairs, original shared postprocessing, MNN, <=3 canonical pixels, equal-scene macro',
        'descriptor_probe':'20 existing DEV queries vs one DEV support image with most shared tracks; geometric oracle pair selection; exclude tracks with duplicate observations in either view; up to256 common query tracks and512 reference observations sorted by Point3D ID; top1/top5 ranks; no fitting',
        'coordinate_variants':[0.0,-0.5], 'coordinate_variant_meaning':'offset in original COLMAP pixel coordinates before affine; legacy and corner-origin to index-origin diagnostic, both reported; no default modification',
        'new_training':False, 'model_selection':False, 'production_map_mutation':False,
        'test':'NOT YET ESTABLISHED', 'source_sha256':sha(__file__)})
    print('IDENTITY_AND_CACHES_PASS', flush=True)

    # Validate decoding against the pinned public upstream forward, not a teacher score target.
    verify()
    upstream_rows=[]
    old_conf=teacher.net.config.copy()
    teacher.net.config.update(nms_radius=4,keypoint_threshold=.005,max_keypoints=160,remove_borders=0)
    for i in (0,14,28):
        with np.load(ROOT/res/'dev'/f'{i:03d}.npz') as z:
            x=torch.from_numpy(z['pair'].copy())
        a,b=raw_teacher(x)
        decoded=features(a,b,np.ones((2,320,184),np.uint8),res)
        upstream=teacher.net({'image':x})
        for j,(xy,d) in enumerate(decoded):
            expected_xy=upstream['keypoints'][j].numpy()
            np.testing.assert_allclose(xy*1.25+[2,0],expected_xy,atol=1e-6,rtol=0)
            err=float(np.max(np.abs(d-upstream['descriptors'][j].T.numpy()))) if len(d) else 0.
            assert err<=2e-6
            upstream_rows.append({'pair':i,'view':j,'keypoints':len(d),'descriptor_max_abs':err})
    teacher.net.config=old_conf
    # Independent explicit permutation/empty/bad-match cases for metric contract.
    xy=np.array([[15.,15.],[40.,40.],[80.,70.],[110.,180.]])
    h=np.array([[1.,.03,2.],[-.02,1.,3.],[.0001,0.,1.]])
    perm=np.array([2,0,3,1]); d=np.eye(4)
    good=pair_metrics((xy,d),(project(xy,h)[perm],d[perm]),h,np.array([0,0,144,256]))
    assert good['correct_mnn_count']==4 and good['mnn_precision']==1
    bad=pair_metrics((xy,d),(project(xy,h)[perm]+100,d[perm]),h,np.array([0,0,144,256]))
    assert bad['correct_mnn_count']==0 and bad['mnn_count']==4
    validation={'status':'INCONCLUSIVE', 'upstream_decoder_contract':'PASS', 'upstream_rows':upstream_rows,
                'geometric_metric_unit_contract':'PASS',
                'public_benchmark_reproduction':'NOT_RUN; HPatches not established locally; no selected/frozen public evaluator yet',
                'limits':['Mask-before-NMS and asymmetric nearest-point repeatability are custom pilot definitions.',
                          'Matching precision is not homography estimation accuracy.',
                          'Passing unit/decoder checks permits scoped diagnostic comparison, not published-score claims.']}
    put(out/'evaluator_validation.json',validation)
    print('UPSTREAM_DECODER_CONTRACT_PASS; PUBLIC_BENCHMARK_INCONCLUSIVE',flush=True)

    class Actual(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.index=0
        def forward(self,x):
            i=self.index; self.index+=1
            with np.load(ROOT/res/'dev'/f'{i:03d}.npz') as z:
                assert np.array_equal(x.numpy(),z['pair'])
            with np.load(dep/'sim_outputs'/f'{i:03d}.npz') as z:
                return tuple(torch.from_numpy(z[k].copy()) for k in ('logits','desc'))
    result={}
    for name,network in [('teacher_fp32',raw_teacher),('student_fp32',model),('student_int8',Actual())]:
        result[name]=evaluate(network,res,p['rows']['dev'])
        rows=result[name]['rows']
        result[name]['distribution']={k:{'mean':float(np.mean([r['metrics'][k] for r in rows])),
                                          'median':float(np.median([r['metrics'][k] for r in rows]))} for k in rows[0]['metrics']}
        result[name]['zero_correct_pairs']=sum(r['metrics']['correct_mnn_count']==0 for r in rows)
        result[name]['zero_candidate_match_pairs']=sum(r['metrics']['mnn_count']==0 for r in rows)
        print(name,json.dumps(result[name]['macro']),flush=True)
    old=read(dep/'int8_quality.json')
    for name,key in [('student_fp32','fp32'),('student_int8','int8')]:
        assert result[name]['rows']==old[key]['rows'], f'Replay changed: {name}'
    result['deltas']={label:{k:result[b]['macro'][k]-result[a]['macro'][k] for k in result[a]['macro']}
                      for label,a,b in [('student_minus_teacher','teacher_fp32','student_fp32'),('int8_minus_fp32','student_fp32','student_int8')]}
    put(out/'feature_comparison.json',result)
    write_csv(out/'feature_pairs.csv',[dict(model=n,index=r['index'],scene=r['scene'],domain=r['domain'],**r['metrics'])
                                      for n in ('teacher_fp32','student_fp32','student_int8') for r in result[n]['rows']])
    natural_probe(out,raw_teacher)
    # Ensure model/evaluator sources and checkpoint remained frozen during measurements.
    for path,digest in manifest['source_hashes'].items(): checked(path,digest)
    checked(checkpoint,manifest['checkpoint_sha256'])
    put(out/'completion.json',{'status':'DIAGNOSTIC_COMPLETE','public_benchmark':'NOT_RUN',
                              'training':False,'board_performance':'NOT_MEASURED'})


@torch.inference_mode()
def natural_probe(out,teacher):
    from spk210.map_nms_gate import DEST, MODEL, context
    images,_=context(); p=read(DEST/'protocol.json')
    inputs=read(DEST/'inputs.json'); entries={r['id']:r for r in inputs['records']}
    sim=read(DEST/'simulation.json')
    checked(DEST/'inputs.json',sim['inputs_manifest_sha256'])
    assert p['checkpoint_sha256']==read(out/'manifest.json')['checkpoint_sha256']
    assert p['kmodel_sha256']==read(out/'manifest.json')['compile']['kmodel_sha256']==sim['kmodel_sha256']
    for n,digest in p['geometry_hashes'].items(): checked(MODEL/n,digest)
    support=p['split_ids']['dev'][::2]; queries=p['split_ids']['dev'][1::2]
    visible={i:set(int(v) for v in images[i].point3D_ids if v>=0) for i in support+queries}
    pairs=[(q,min(support,key=lambda r:(-len(visible[q]&visible[r]),r))) for q in queries]
    selected=sorted(set(i for pair in pairs for i in pair))
    sampled={}; raw_hashes={}; ambiguous={}
    for i in selected:
        src=DEST/'inputs'/f'{i}.npz'; checked(src,entries[i]['input_sha256'])
        with np.load(src) as z:
            x=z['image'].copy(); affine=z['affine'].copy(); mask=z['mask'][0].copy()
        heads={'teacher_fp32':teacher(torch.from_numpy(x))[1]}
        for label,backend in [('student_fp32','fp32'),('student_int8','int8')]:
            file=DEST/backend/f'{i}.npz'
            raw_hashes[str(file)]=checked(file,entries[i]['fp32_sha256'] if backend=='fp32' else sim['outputs'][str(i)])
            with np.load(file) as z: heads[label]=torch.from_numpy(z['desc'].copy())
        im=images[i]; valid=im.point3D_ids>=0
        ids=im.point3D_ids[valid]; original=im.xys[valid]
        unique,counts=np.unique(ids,return_counts=True)
        duplicates=unique[counts>1]
        ambiguous[str(i)]=duplicates.tolist()
        unambiguous=~np.isin(ids,duplicates)
        ids,original=ids[unambiguous],original[unambiguous]
        for offset in (0.,-.5):
            xy=(original+offset)*affine[:2]+affine[2:]
            rounded=np.rint(xy).astype(int)
            inside=(rounded[:,0]>=0)&(rounded[:,0]<184)&(rounded[:,1]>=0)&(rounded[:,1]<320)
            keep=np.zeros(len(ids),bool); keep[inside]=mask[rounded[inside,1],rounded[inside,0]]>0
            sid=ids[keep]; sxy=xy[keep]
            order=np.argsort(sid,kind='stable'); sid=sid[order]; sxy=sxy[order]
            assert len(np.unique(sid))==len(sid)
            for name,b in heads.items():
                desc=ops.sample_descriptors(torch.from_numpy(sxy.astype(np.float32))[None],F.normalize(b,dim=1),8)[0].T.numpy()
                sampled[i,offset,name]=(sid,desc)
    rows=[]
    for q,r in pairs:
        for offset in (0.,-.5):
            for name in ('teacher_fp32','student_fp32','student_int8'):
                qi,qd=sampled[q,offset,name]; ri,rd=sampled[r,offset,name]
                ri,rd=ri[:512],rd[:512]
                common=np.intersect1d(qi,ri)[:256]
                if not len(common):
                    rows.append(dict(query=q,reference=r,offset=offset,model=name,tracks=0,candidates=len(ri),top1=None,top5=None,mean_true_cosine=None))
                    continue
                qd=qd[np.searchsorted(qi,common)]; true=np.searchsorted(ri,common)
                score=qd@rd.T
                # Stable rank by reference Point3D ordering, including exact ties.
                ordering=np.argsort(-score,axis=1,kind='stable')
                ranks=np.argmax(ordering==true[:,None],axis=1)+1
                rows.append(dict(query=q,reference=r,offset=offset,model=name,tracks=len(common),candidates=len(ri),
                                 top1=float(np.mean(ranks==1)),top5=float(np.mean(ranks<=5)),
                                 mean_true_cosine=float(np.mean(score[np.arange(len(common)),true]))))
    summary={}
    for offset in (0.,-.5):
        for name in ('teacher_fp32','student_fp32','student_int8'):
            subset=[r for r in rows if r['offset']==offset and r['model']==name]
            valid=[r for r in subset if r['tracks']]
            summary[f'{name}_offset_{offset}']={'pairs':len(subset),'empty_pairs':len(subset)-len(valid),
                                               'tracks':sum(r['tracks'] for r in valid),
                                               **{k:float(np.mean([r[k] for r in valid])) if valid else None for k in ('top1','top5','mean_true_cosine')}}
    put(out/'natural_descriptor_probe.json',{'scope':'oracle known-track ranking; no detector, no pose prediction, not public benchmark; old COLMAP tracks not feature-specific reconstruction',
                                            'protocol_sha256':sha(out/'protocol.json'),'raw_hashes':raw_hashes,
                                            'excluded_duplicate_track_ids_by_image':ambiguous,'summary':summary,'rows':rows})
    write_csv(out/'natural_descriptor_pairs.csv',rows)
    print('NATURAL_DESCRIPTOR_PROBE',json.dumps(summary),flush=True)
    # Reuse only historical reports whose recorded hashes and protocol association match.
    reused={}
    for ledger in ('reference_fix_evidence_hashes.json','root_cause_evidence_hashes.json'):
        for name,digest in read(DEST/ledger).items():
            reused[str(DEST/name)]=checked(DEST/name,digest)
    geometry=read(DEST/'geometry_audit.json')
    for name,digest in geometry['geometry_hashes'].items(): checked(MODEL/name,digest)
    checked(SUPERPOINT/'scripts/audit_colmap_geometry.py',geometry['source_sha256'])
    local={}
    for backend in ('official_low','student_fp32','student_int8'):
        folder=DEST/f'reference_coverage_{backend}'
        data=read(folder/'results.json'); checked(folder/'protocol.json',data['protocol_sha256'])
        assert data['complete']
        local[backend]=data['results']['TRAIN_PLUS_DEV_SUPPORT_per_reference']['summary']
    put(out/'localization_validation.json',{'status':'INCONCLUSIVE','geometry_internal_consistency':'PASS',
        'geometry_summary':geometry['summary'],'historical_covered_sanity':local,'verified_report_hashes':reused,
        'limits':['Historical teacher original NMS versus student greedy-corner NMS is not network-only isolation.',
                  'Legacy COLMAP coordinate offset needs explicit end-to-end upstream comparison.',
                  'Native map units are not established meters; map includes query observations.',
                  'Report hashes validated; full historical teacher feature-bank generation not replayed.',
                  'Known-track descriptor probe bypasses detection and is not feature-specific triangulation.']})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args().output)
