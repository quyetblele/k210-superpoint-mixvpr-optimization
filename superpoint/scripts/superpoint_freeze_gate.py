"""Evaluate predeclared INT8 retention gates before opening the sealed TEST."""
import _bootstrap
import json
from spk210.settings import SUPERPOINT
from spk210.io import sha,put

def run():
    root=SUPERPOINT/'artifacts/full_training';dep=SUPERPOINT/'artifacts/deployment/full_capacity_r2_cw';read=lambda p:json.loads(p.read_text());p=read(root/'freeze_gate_protocol.json');assert sha(root/'best.pt')==p['checkpoint_sha256']
    parity=read(dep/'onnx_parity.json');assert parity['status']=='PASS' and parity['checkpoint_sha256']==p['checkpoint_sha256'];assert parity['onnx_sha256']==sha(dep/'canonical.onnx')
    quality=read(dep/'int8_quality.json');floor=read(root/'floor6/results.json');c=read(dep/'compile.json');assert c['CPU_Conv2D_count']==0 and c['KPU_Conv2D_count']==12;assert c['checkpoint_sha256']==p['checkpoint_sha256'] and sha(dep/'model.kmodel')==c['kmodel_sha256'];assert quality['checkpoint_sha256']==p['checkpoint_sha256'] and quality['kmodel_sha256']==c['kmodel_sha256']
    f=quality['fp32']['macro'];i=quality['int8']['macro'];ff=floor['summary']['fp32/own'];fi=floor['summary']['int8/own'];limits=p['quality_gate'];metrics={'DEV_correct_mnn_retention':i['correct_mnn_count']/f['correct_mnn_count'],'DEV_precision_drop_pp':100*(f['mnn_precision']-i['mnn_precision']),'floor6_own_pose':fi['good_poses'],'floor6_own_mean_inlier_retention':fi['mean_inliers']/ff['mean_inliers']}
    checks={'DEV_correct_mnn':metrics['DEV_correct_mnn_retention']>=limits['DEV_correct_mnn_retention_min'],'DEV_precision':metrics['DEV_precision_drop_pp']<=limits['DEV_precision_drop_max_pp'],'floor6_pose':metrics['floor6_own_pose']>=limits['floor6_own_pose_min'],'floor6_inliers':metrics['floor6_own_mean_inlier_retention']>=limits['floor6_own_mean_inlier_retention_min']}
    result={'status':'ELIGIBLE_TO_FREEZE' if all(checks.values()) else 'DO_NOT_FREEZE','checks':checks,'metrics':metrics,'criteria':limits,'checkpoint_sha256':p['checkpoint_sha256'],'kmodel_sha256':c['kmodel_sha256'],'gate_protocol_sha256':sha(root/'freeze_gate_protocol.json'),'test_opened':False,'fp32_dev':f,'int8_dev':i,'floor6_fp32':ff,'floor6_int8':fi};put(root/'freeze_decision.json',result);print(json.dumps(result),flush=True)
if __name__=='__main__':run()
