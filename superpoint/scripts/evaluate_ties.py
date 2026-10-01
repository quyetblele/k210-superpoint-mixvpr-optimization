import _bootstrap
import argparse
from spk210.tie_breaking import run
p=argparse.ArgumentParser();p.add_argument('--candidate',required=True);a=p.parse_args()
run(a.candidate)
