"""Read the locked experiment contract; implementation changes are recorded separately."""
import json
from .settings import ROOT, SUPERPOINT
from .io import sha

def protocol():
    configured = SUPERPOINT / 'configs/pilot400.protocol.json'
    frozen = ROOT / 'protocol.json'
    if not frozen.exists():
        ROOT.mkdir(parents=True, exist_ok=True)
        frozen.write_bytes(configured.read_bytes())
    assert sha(configured) == sha(frozen), 'Frozen protocol mismatch'
    p = json.loads(frozen.read_text())
    assert p['GT1_used'] is False and p['updates_each'] == 400
    return p

def verify_cache():
    protocol()
    done = json.loads((ROOT / 'cache_done.json').read_text())
    assert done['protocol_sha256'] == sha(ROOT / 'protocol.json')
    for (name, h) in done['files'].items():
        assert sha(ROOT / name) == h, 'Cache identity mismatch: ' + name
    return done
