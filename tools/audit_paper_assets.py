"""Inventory research assets; optionally delete only recorded Python bytecode caches.

No dataset, checkpoint, log or experimental evidence is automatically deleted.
Reports use unique UTC run directories and preserve previous audits.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    with path.open('rb') as stream:
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
        return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--clean-bytecode', action='store_true')
    args = parser.parse_args()
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    dest = ROOT / 'docs/audit/paper' / stamp
    dest.mkdir(parents=True)
    totals, counts, bytecode, links = Counter(), Counter(), [], []
    with gzip.open(dest / 'inventory.jsonl.gz', 'wt') as out:
        for base, dirs, files in os.walk(ROOT, followlinks=False):
            dirs[:] = sorted(d for d in dirs if d != '.git' and not d.startswith('.venv'))
            if Path(base).is_relative_to(ROOT / 'docs/audit'):
                dirs[:] = []
                continue
            for name in sorted(files):
                p = Path(base) / name
                rel = p.relative_to(ROOT)
                stat = p.lstat()
                group = '/'.join(rel.parts[:2])
                totals[group] += stat.st_size
                counts[group] += 1
                item = {'path': str(rel), 'bytes': stat.st_size, 'symlink': p.is_symlink()}
                if p.is_symlink():
                    item['target'] = os.readlink(p)
                    links.append(item)
                if '__pycache__' in rel.parts and p.suffix == '.pyc' and not p.is_symlink():
                    item.update(action='DELETE_REGENERABLE_BYTECODE', sha256=sha(p))
                    bytecode.append(item.copy())
                elif rel.parts[0] == 'experiments' or rel.parts[:2] == ('superpoint', 'artifacts'):
                    item['action'] = 'HISTORICAL_EVIDENCE_KEEP_IN_PLACE'
                else:
                    item['action'] = 'KEEP'
                out.write(json.dumps(item) + '\n')
    scenes = {}
    for scene in sorted((ROOT / 'datasets/7scenes').iterdir()):
        if not scene.is_dir():
            continue
        types = Counter()
        for p in scene.glob('seq-*/*'):
            if p.is_file():
                types['.'.join(p.name.split('.')[1:])] += 1
        splits = {p.name: p.read_text().splitlines() for p in scene.glob('*Split.txt')}
        scenes[scene.name] = {'file_types': dict(types), 'official_splits': splits}
    archives = []
    for p in sorted((ROOT / 'datasets/7scenes_archives').glob('*.zip')):
        with zipfile.ZipFile(p) as z:
            members = z.infolist()
            archives.append({'path': str(p.relative_to(ROOT)), 'bytes': p.stat().st_size,
                             'members': len(members), 'first_members': [x.filename for x in members[:12]],
                             'action': 'KEEP_NOT_PROVEN_REDUNDANT',
                             'note': 'Directory inspected only; CRC/content equivalence not established.'})
    # Search existing user projects for HPatches, excluding environments and caches.
    found = []
    search_roots = [ROOT / 'datasets', ROOT.parent / 'edge_ai_project', ROOT.parent / 'edge_ai_server', ROOT.parent / 'research']
    for project in search_roots:
        if not project.is_dir() or project.is_symlink():
            continue
        for base, dirs, files in os.walk(project, followlinks=False):
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ('__pycache__', 'node_modules', 'site-packages', 'venv')]
            if len(Path(base).relative_to(project).parts) >= 5:
                dirs[:] = []
            for name in dirs + files:
                if 'hpatch' in name.lower():
                    found.append(str(Path(base) / name))
    # Persist the proposed deletion list before performing any deletion.
    ledger = {'requested': args.clean_bytecode, 'candidates': bytecode, 'deleted': []}
    ledger_path = dest / 'cleanup.json'
    ledger_path.write_text(json.dumps(ledger, indent=2) + '\n')
    if args.clean_bytecode:
        for item in bytecode:
            p = ROOT / item['path']
            if p.is_symlink() or sha(p) != item['sha256']:
                raise RuntimeError(f'Changed cleanup candidate: {p}')
            p.unlink()
            ledger['deleted'].append(item)
            ledger_path.write_text(json.dumps(ledger, indent=2) + '\n')
    summary = {'timestamp_utc': stamp, 'inventory_scope': str(ROOT),
               'logical_bytes_by_group': dict(totals), 'files_by_group': dict(counts),
               'scenes': scenes, 'archives': archives, 'symlinks': links,
               'hpatches_name_search': found,
               'deleted_files': len(ledger['deleted']),
               'deleted_logical_bytes': sum(x['bytes'] for x in ledger['deleted']),
               'git_status': subprocess.check_output(['git', '-C', str(ROOT), 'status', '--short'], text=True),
               'hpatches_search_roots': [str(p) for p in search_roots],
               'scope_note': 'Metadata inventory; not a full dataset integrity/duplicate-content proof. HPatches name search depth limited to 5 within listed roots.'}
    (dest / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps({'report': str(dest), 'files': sum(counts.values()),
                      'deleted_files': summary['deleted_files'],
                      'deleted_logical_bytes': summary['deleted_logical_bytes'],
                      'hpatches_name_search': found}, indent=2))


if __name__ == '__main__':
    main()
