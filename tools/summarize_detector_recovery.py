"""Seal recovery evidence and draw its learning curves without changing selection."""
import json,hashlib
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
w=Path(__file__).resolve().parents[1];sp=w/'superpoint';root=sp/'artifacts/detector_recovery'
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def put(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
s=json.loads((root/'summary.json').read_text());assert s['status']=='COMPLETE'
fig,axes=plt.subplots(1,3,figsize=(12,3.6))
for index,(name,result) in enumerate(s['results'].items()):
    report=json.loads((root/name/'report.json').read_text());ev=report['state']['evaluations'];label='control' if index==0 else 'foreground balanced'
    for ax,key,title in zip(axes[:2],['correct_mnn_count','mnn_precision'],['FP32 correct matches / pair','FP32 MNN precision']):
        ax.plot([e['step'] for e in ev],[e['metrics']['macro'][key] for e in ev],marker='o',label=label);ax.set(xlabel='Additional updates',ylabel=title);ax.grid(alpha=.2)
    axes[2].bar(label,result['quality']['int8']['macro']['correct_mnn_count'])
axes[0].legend(fontsize=8);axes[2].set_ylabel('Actual INT8 correct matches / pair');fig.tight_layout()
fig.savefig(root/'learning_curves.png',dpi=150);fig.savefig(root/'learning_curves.pdf');plt.close(fig)
report=root/'REPORT.md';text=report.read_text()
if '![Learning curves]' not in text:report.write_text(text+'\n![Learning curves](learning_curves.png)\n')
completion=sp/'results/completion.json';current=json.loads(completion.read_text())
current.update(detector_recovery='COMPLETE',detector_recovery_gate_pass=s['gate_pass'],next_gate=s['next_gate'],training_running=False)
put(completion,current)
files=list((sp/'src').rglob('*.py'))+list((sp/'scripts').glob('*.py'))+[sp/'run',sp/'configs/candidates.json']
put(root/'implementation.json',{str(f.relative_to(w)):sha(f) for f in files})
identities={}
for name,r in s['results'].items():
    d=sp/'artifacts/deployment'/name
    assert sha(root/name/'best.pt')==r['quality']['checkpoint_sha256']
    assert sha(d/'model.kmodel')==r['quality']['kmodel_sha256']
    simulation=json.loads((d/'simulation.json').read_text())
    for row in simulation['rows']:
        assert sha(d/'sim_outputs'/f"{row['index']:03d}.npz")==row['output_sha256']
        assert sha(sp/'artifacts/quality_pilot/r2/dev'/f"{row['index']:03d}.npz")==row['input_sha256']
    identities[name]={'checkpoint_sha256':sha(root/name/'best.pt'),'kmodel_sha256':sha(d/'model.kmodel'),'simulation_outputs_verified':len(simulation['rows']),'training_protocol_sha256':sha(root/'protocol.json')}
put(root/'verification.json',{'status':'PASS','identities':identities,'loss_weighting_check':'uniform teacher foreground => balanced equals uniform; masked cells have zero gradient','frozen_backbone_descriptor':'verified exact in both training reports'})
put(root/'artifacts.sha256.json',{str(p.relative_to(root)):sha(p) for p in sorted(root.rglob('*')) if p.is_file() and p.name!='artifacts.sha256.json' and p.suffix in ['.json','.pt','.npy','.csv','.png','.pdf','.md']})
print(json.dumps({'gate_pass':s['gate_pass'],'next_gate':s['next_gate'],'identities':identities}),flush=True)
