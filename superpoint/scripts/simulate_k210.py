import _bootstrap
import argparse, json
from spk210.settings import ROOT
from spk210.nncase_backend import simulate
p = argparse.ArgumentParser()
p.add_argument('--candidate')
a = p.parse_args()
c = a.candidate or json.loads((ROOT / 'summary.json').read_text())['selected_for_next_gate']
simulate(c)
