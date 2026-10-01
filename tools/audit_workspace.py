"""Inventory roles and Python entrypoints; never deletes or moves files."""
import ast,gzip,hashlib,json,os
from pathlib import Path
root=Path(__file__).resolve().parents[1];dest=root/'docs/audit';dest.mkdir(exist_ok=True)
count=0;sources=[];totals={}
with gzip.open(dest/'current_inventory.json.gz','wt',encoding='utf-8') as output:
    output.write('[\n');first=True
    for base,dirs,files in os.walk(root):
        dirs[:]=[d for d in dirs if d not in ['.git','.venv-mixvpr-cuda','__pycache__']]
        for name in files:
            path=Path(base)/name;rel=path.relative_to(root)
            if 'docs/audit' in str(rel):continue
            size=path.lstat().st_size;role='KEEP';reason='Preserve; no automatic deletion decision'
            if rel.parts[0]=='experiments':role='ARCHIVE_IN_PLACE';reason='Historical provenance and absolute-path dependencies'
            elif rel.parts[0]=='archive':role='ARCHIVE';reason='Explicit archive'
            elif rel.parts[:2]==('superpoint','src') or rel.parts[:2]==('superpoint','scripts'):reason='Canonical SuperPoint implementation/entrypoint'
            elif rel.parts[0] in ['datasets','models','reports','artifacts']:reason='Dataset/model/engineering evidence; protected from automatic cleanup'
            item={'path':str(rel),'bytes':size,'role':role,'reason':reason,'symlink':path.is_symlink()}
            if not first:output.write(',\n')
            output.write(json.dumps(item));first=False;count+=1;totals[rel.parts[0]]=totals.get(rel.parts[0],0)+size
            if path.suffix=='.py' and not path.is_symlink():
                text=path.read_text(errors='replace')
                try:
                    tree=ast.parse(text);symbols=[n.name for n in tree.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef))];doc=ast.get_docstring(tree)
                except SyntaxError:symbols=[];doc='SYNTAX_ERROR'
                sources.append({'path':str(rel),'sha256':hashlib.sha256(text.encode()).hexdigest(),'symbols':symbols,'docstring':doc})
    output.write('\n]\n')
(dest/'current_source_index.json').write_text(json.dumps(sources,indent=2)+'\n')
(dest/'current_inventory_summary.json').write_text(json.dumps({'files':count,'Python_sources':len(sources),'logical_bytes_by_root':totals,'policy':'Read-only classification; no dataset/checkpoint/benchmark deletion; logical bytes are not disk physical allocation.'},indent=2)+'\n')
print('Audit complete:',count,'files;',len(sources),'Python sources; no mutations outside audit evidence')
