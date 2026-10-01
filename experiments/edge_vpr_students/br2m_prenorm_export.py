import sys,torch,onnx
from pathlib import Path
R=Path('/home/quyet/k210_lab');sys.path.insert(0,str(R/'experiments/edge_vpr_students'));from s512_br2m_gate import BR2M
class C(torch.nn.Module):
 def __init__(self,m):super().__init__();self.m=m
 def forward(self,x):return self.m.raw(x)
torch.manual_seed(20260906);m=BR2M().eval();x=torch.zeros(1,3,352,192);p=R/'models/s512_br2m_prenorm_core_192x352.onnx';torch.onnx.export(C(m),x,str(p),opset_version=13,dynamo=False,input_names=['normalized_rgb'],output_names=['raw_descriptor']);q=onnx.shape_inference.infer_shapes(onnx.load(p),strict_mode=True);onnx.checker.check_model(q);onnx.save(q,R/'models/s512_br2m_prenorm_core_192x352_inferred.onnx')
