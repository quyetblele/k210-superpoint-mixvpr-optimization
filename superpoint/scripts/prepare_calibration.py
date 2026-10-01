import _bootstrap
import argparse
from spk210.onnx_export import calibrate, selected_candidate
p = argparse.ArgumentParser()
p.add_argument('--candidate')
a = p.parse_args()
calibrate(selected_candidate(a.candidate))
