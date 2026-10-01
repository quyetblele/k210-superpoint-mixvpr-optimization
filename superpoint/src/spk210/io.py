"""Atomic evidence writes and host resource guards (not board memory measurements)."""
import hashlib, json
from pathlib import Path
from .settings import ROOT

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda : f.read(1048576), b''):
            h.update(b)
    return h.hexdigest()

def put(path, value):
    t = Path(str(path) + '.tmp')
    t.write_text(json.dumps(value, indent=2) + '\n')
    t.replace(path)

def guard():
    available = int(next((x for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:'))).split()[1]) / 1024
    if available < 1536:
        raise RuntimeError('Available host RAM below 1536 MiB; resume from saved checkpoint')
    import shutil
    if shutil.disk_usage(ROOT).free < 2 * 1024 ** 3:
        raise RuntimeError('Less than 2 GiB free disk')

def save(path, value):
    import torch
    t = Path(str(path) + '.tmp')
    torch.save(value, t)
    t.replace(path)
