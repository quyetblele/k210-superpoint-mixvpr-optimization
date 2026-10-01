"""Cross raw heads from FP32 and actual kmodel outputs to localize quality loss."""
import json
import numpy as np
from .runtime import torch
from .settings import ROOT
from .protocol import protocol
from .io import sha,put
from .metrics import evaluate
from .onnx_export import deployment_dir,load_model

def run(candidate):
    model,res,checkpoint=load_model(candidate);dest=deployment_dir(candidate)
    quality=json.loads((dest/'int8_quality.json').read_text())
    simulation=json.loads((dest/'simulation.json').read_text())
    assert quality['checkpoint_sha256']==sha(checkpoint)==simulation['checkpoint_sha256']
    assert quality['kmodel_sha256']==sha(dest/'model.kmodel')==simulation['kmodel_sha256']
    for row in simulation['rows']:
        assert sha(ROOT/res/'dev'/f"{row['index']:03d}.npz")==row['input_sha256']
        assert sha(dest/'sim_outputs'/f"{row['index']:03d}.npz")==row['output_sha256']
    class Hybrid(torch.nn.Module):
        def __init__(self,quantized_detector,quantized_descriptor):
            super().__init__();self.index=0;self.use=(quantized_detector,quantized_descriptor)
        def forward(self,x):
            fp32=model(x)
            with np.load(dest/'sim_outputs'/f'{self.index:03d}.npz') as z:q=tuple(torch.from_numpy(z[k].copy()) for k in ['logits','desc'])
            self.index+=1
            return tuple(q[j] if flag else fp32[j] for j,flag in enumerate(self.use))
    variants={'FP32_detector_INT8_descriptor':(False,True),'INT8_detector_FP32_descriptor':(True,False)}
    results={name:evaluate(Hybrid(*flags),res,protocol()['rows']['dev']) for name,flags in variants.items()}
    put(dest/'head_diagnosis.json',{'status':'COMPLETE','checkpoint_sha256':sha(checkpoint),'kmodel_sha256':sha(dest/'model.kmodel'),'reference_fp32':quality['fp32'],'reference_int8':quality['int8'],'hybrids':results,'scope':'Engineering causal isolation using actual simulator tensors; hybrid outputs are NOT deployable INT8 results and do not count as final INT8 quality. No weights, thresholds or calibration changed.'})
