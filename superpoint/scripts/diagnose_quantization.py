import _bootstrap
import argparse
from spk210.onnx_export import selected_candidate
from spk210.quantization_diagnosis import run
p=argparse.ArgumentParser();p.add_argument('--candidate');a=p.parse_args()
run(selected_candidate(a.candidate))
