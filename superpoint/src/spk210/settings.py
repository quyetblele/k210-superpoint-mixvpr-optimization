"""Workspace paths and the locked pilot architecture/training envelope."""
from pathlib import Path
WORKSPACE = Path(__file__).resolve().parents[3]
SUPERPOINT = WORKSPACE / 'superpoint'
ROOT = SUPERPOINT / 'artifacts/quality_pilot'
WIDTHS = {'fr12': [12, 24, 32, 64], 'wider': [24, 32, 64, 96]}
RES = {'r1': (144, 256, 1.0, 0), 'r2': (184, 320, 1.25, 2)}
SEED = 20260919
STEPS = 400
CANDIDATES = [('fr12', 'r1'), ('wider', 'r1'), ('fr12', 'r2'), ('wider', 'r2')]

def candidate_name(width, res):
    return width + '_' + res + ('_cw' if res == 'r2' else '')

def parse_candidate(name):
    matches = [(w, r) for (w, r) in CANDIDATES if candidate_name(w, r) == name]
    if not matches:
        raise ValueError('Unknown candidate: ' + name)
    return matches[0]
