"""Frozen official teacher, clean-view per-image KD, disk-backed lazy cache."""
import hashlib
import json
import sqlite3
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import success_first_train_core as core

POLICY = 'official_per_image_clean_bicubic320_fp32_v1'

class PerImageTeacher:
    def __init__(self, directory, device):
        self.device = device
        self.teacher_hash = core.sha256(core.OFFICIAL)
        self.contract = {'policy': POLICY, 'teacher_sha256': self.teacher_hash,
                         'augmentation': 'teacher clean image; student photometric only',
                         'descriptor': 'FP32 normalized 4096D'}
        directory = Path(directory) / self.teacher_hash
        directory.mkdir(parents=True, exist_ok=True)
        manifest = directory / 'contract.json'
        if manifest.exists() and json.loads(manifest.read_text()) != self.contract:
            raise RuntimeError('teacher cache contract mismatch')
        core.atomic_json(manifest, self.contract)
        self.db = sqlite3.connect(directory / 'descriptors.sqlite')
        self.db.execute('CREATE TABLE IF NOT EXISTS descriptors(key TEXT PRIMARY KEY, value BLOB NOT NULL)')
        self.model = None

    def targets(self, samples):
        outputs = []
        for sample in samples:
            path = Path(sample['path'])
            # Content hash prevents stale targets if an image changes in place.
            key = hashlib.sha256((str(path.resolve())+core.sha256(path)).encode()).hexdigest()
            row = self.db.execute('SELECT value FROM descriptors WHERE key=?',(key,)).fetchone()
            if row is None:
                if self.model is None:
                    rng = core.rng_state()
                    self.model = core.build_teacher().to(self.device).eval().requires_grad_(False)
                    core.restore_rng(rng)
                with torch.inference_mode(), torch.amp.autocast('cuda', enabled=False):
                    x = core.image_tensor(str(path),320,None)[None].to(self.device)
                    value = F.normalize(self.model(x).float(),dim=1)[0].cpu().numpy()
                if value.shape != (4096,) or not np.isfinite(value).all():
                    raise RuntimeError('invalid per-image teacher descriptor')
                self.db.execute('INSERT INTO descriptors VALUES (?,?)',(key,value.astype('<f4').tobytes()))
                self.db.commit()
            else:
                value = np.frombuffer(row[0],dtype='<f4').copy()
            outputs.append(value)
        return torch.from_numpy(np.stack(outputs)).to(self.device)
