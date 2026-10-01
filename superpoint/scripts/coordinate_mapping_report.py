"""Verify coordinate A/B artifacts and evaluate frozen improved-keypoint diagnostic."""
import _bootstrap
import json,csv
import numpy as np
import torch.nn.functional as F
from coordinate_mapping_ab import DEST,BRANCHES
from fixed_keypoint_pnp import read,check,pose,match,context,MODEL,read_points3D_binary
from spk210.runtime import torch,ops
from spk210.settings import SUPERPOINT,ROOT,WORKSPACE
from spk210.io import sha,put

@torch.inference_mode()
def main():
    fixed=SUPERPOINT/'artifacts/fixed_keypoint_pnp/run_001'; prior=SUPERPOINT/'artifacts/keypoint_cause_audit/run_003'
    fe=read(fixed/'evidence_hashes.json'); pe=read(prior/'evidence_hashes.json')
    for f in ['protocol.json','common_inputs.json']:check(fixed/f,fe[f])
    check(prior/'keypoints_and_matches.json',pe['keypoints_and_matches.json'])
    p=read(fixed/'protocol.json'); common=read(fixed/'common_inputs.json'); loc=read(prior/'keypoints_and_matches.json')
    images,cameras=context(); points=read_points3D_binary(MODEL/'points3D.bin')
    for f,h in p['geometry_hashes'].items():check(MODEL/f,h)
    refs={int(q):rr for q,rr in p['references'].items()}; rids=sorted(set(i for rr in refs.values() for i in rr)); records=[];cal=[]
    spec=read(DEST/'protocol.json');check(DEST/'shared_points.npz',spec['shared_points_sha256'])
    initial_metrics=[]
    checks={'updates':400,'shared_removed':sum(len(r['excluded_indices']) for r in spec['audit']),'retained_pairs':len(spec['audit']),'replays':0}
    for name in BRANCHES:
        report=read(DEST/name/'report.json');assert report['updates']==400 and report['status']=='COMPLETE'
        assert report['initial_sha256']==spec['initial_sha256'] and report['plan_sha256']==spec['plan_sha256']
        check(DEST/name/'best.pt',report['checkpoint_sha256'])
        progress=read(DEST/name/'progress.json');initial_metrics.append(progress['evaluations'][0]['metrics']);assert [r['step'] for r in progress['evaluations']]==[0,100,200,300,400]
        losses=progress['losses'];assert len(losses)==400 and all(np.isfinite([r['detector'],r['geometry']]).all() for r in losses)
        dep=SUPERPOINT/'artifacts/deployment'/name;comp=read(dep/'compile.json');cal.append(comp['calibration_sha256'])
        assert comp['compile']=='PASS' and comp['KPU_Conv2D_count']==12 and comp['CPU_Conv2D_count']==0
        check(dep/'calibration_train.npy',cal[-1]);check(dep/'model.kmodel',comp['kmodel_sha256']);assert read(dep/'onnx_parity.json')['status']=='PASS'
        simulation=read(dep/'simulation.json');assert simulation['status']=='PASS'
        for r in simulation['rows']:check(dep/'sim_outputs'/f"{r['index']:03d}.npz",r['output_sha256'])
        raw=DEST/name/'natural'; nr=read(raw/'results.json');check(raw/'protocol.json',nr['protocol_sha256'])
        ns=read(raw/'simulation.json');assert ns['checkpoint_sha256']==report['checkpoint_sha256'] and ns['kmodel_sha256']==comp['kmodel_sha256']
        improved={}
        for backend in ['fp32','int8']:
            bank={}; qdesc={}
            for i in sorted(rids+p['query_ids']):
                f=raw/backend/f'{i}.npz';check(f,nr['results'][backend]['raw_hashes'][str(i)])
                if backend=='int8':check(f,ns['records'][str(i)]['output_sha256'])
                with np.load(f) as z:b=torch.from_numpy(z['desc'].copy())
                if i in p['query_ids']:xy=np.array(loc[f'fp32_gftt_common_subpixel_{i}']['network_xy'])
                else:
                    f=fixed/'teacher_fp32'/f'{i}.npz';check(f,fe[str(f.relative_to(fixed))])
                    with np.load(f) as z:xy=z['network_xy'].copy();ids=z['ids'].copy()
                d=ops.sample_descriptors(torch.from_numpy(xy.astype(np.float32))[None],F.normalize(b,dim=1),8)[0].T.numpy()
                if i in p['query_ids']:qdesc[i]=d
                else:bank[i]=(ids,d)
            rows=[];details={}
            for q in p['query_ids']:
                xy=np.array(loc[f'fp32_gftt_common_subpixel_{q}']['pnp_colmap_xy']); mm=match(qdesc[q],bank,refs[q],common['active_ids'][str(q)])
                ix=np.array([j for j,v,s in mm],int);xyz=np.array([points[v].xyz for j,v,s in mm]).reshape(-1,3);im=images[q];cam=cameras[im.camera_id]
                r=pose(xyz,xy[ix],cam,im);assert r==pose(xyz,xy[ix],cam,im);checks['replays']+=1;r['query']=q;rows.append(r)
                details[str(q)]={'query_indices':ix.tolist(),'point3d_ids':[v for j,v,s in mm]}
            improved[backend]={'good_poses':sum(r['good_pose'] for r in rows),'rows':rows,'matches':details}
            v=nr['results'][backend]
            records.append({'arm':'B' if BRANCHES[name] else 'A','backend':backend,'best_step':report['best_step'],'descriptor_top1_pct':100*v['descriptor']['top1'],'fixed_teacher_pnp':v['fixed']['good_poses'],'fixed_improved_pnp':improved[backend]['good_poses'],'own_pnp':v['own']['good_poses']})
        put(DEST/name/'improved_pnp.json',improved)
    assert initial_metrics[0]==initial_metrics[1]
    assert len(set(cal))==1
    check(SUPERPOINT/'artifacts/deployment/desc_control_r2_cw/calibration_train.npy',cal[0])
    checks.update(status='PASS',calibration_identical_to_prior=True,source_sha256=sha(__file__))
    put(DEST/'checks.json',checks);put(DEST/'summary.json',records)
    table=['| Arm | Backend | Step | Descriptor DEV top1 | Fixed teacher PnP | Fixed improved PnP | Student-keypoint pose |','|---|---|---:|---:|---:|---:|---:|']
    for r in records:table.append(f"| {r['arm']} | {r['backend']} | {r['best_step']} | {r['descriptor_top1_pct']:.2f}% | {r['fixed_teacher_pnp']}/20 | {r['fixed_improved_pnp']}/20 | {r['own_pnp']}/20 |")
    put(DEST/'evidence_hashes.json',{str(f.relative_to(DEST)):sha(f) for f in sorted(DEST.rglob('*')) if f.is_file() and f.name!='evidence_hashes.json'})
    print('\n'.join(table));print(json.dumps(checks))

if __name__=='__main__':main()
