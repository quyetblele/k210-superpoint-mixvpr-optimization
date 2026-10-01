#!/usr/bin/env python3
"""Exact cosine Recall@K of the frozen T0 descriptor databases under GT0."""
from __future__ import annotations
import json,hashlib
from pathlib import Path
import h5py,numpy as np
ROOT=Path('/home/quyet/k210_lab');EDGE=Path('/home/quyet/edge_ai_project');OUT=ROOT/'reports/edge_vpr_students';ART=ROOT/'artifacts/edge_vpr_students/gt0'
def put(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
def load(p):
 d={}
 with h5py.File(p,'r') as h:
  def v(n,o):
   if isinstance(o,h5py.Dataset) and n.endswith('/global_descriptor'):d[n[:-18]]=o[:].astype('float32').reshape(-1)
  h.visititems(v)
 return d
def main():
 m=json.loads((OUT/'gt0_retrieval_protocol_manifest.json').read_text());result={'protocol_manifest_sha256':hashlib.sha256((OUT/'gt0_retrieval_protocol_manifest.json').read_bytes()).hexdigest(),'similarity':'L2-normalized descriptor dot product; same cosine ranking semantics as localize_live.retrieve_topk_global','per_floor':{}}
 for floor,x in m['floors'].items():
  pkg=EDGE/'map_packages'/f'{floor}_v1'; db=load(pkg/'features/global-feats-v2-mixvpr.h5'); refs=(ART/f'{floor}_reference.txt').read_text().splitlines();qs=(ART/f'{floor}_query.txt').read_text().splitlines();labels=json.loads((ART/f'{floor}_positive_labels.json').read_text())['primary']; R=np.stack([db[n]/max(np.linalg.norm(db[n]),1e-12) for n in refs]);rows=[]
  for q in qs:
   pos=set(labels[q]);v=db[q]/max(np.linalg.norm(db[q]),1e-12);order=np.argsort(-(R@v),kind='mergesort');top=[refs[i] for i in order[:20]];rows.append({'query':q,'positive_count':len(pos),'top20':top,'top1_score':float(R[order[0]]@v),'valid':bool(pos),'hit':{str(k):bool(pos.intersection(top[:k])) for k in (1,5,10,20)}})
  valid=[z for z in rows if z['valid']];result['per_floor'][floor]={'reference_count':len(refs),'query_count':len(qs),'zero_positive_count':len(rows)-len(valid),'recall':{f'R@{k}':float(np.mean([z['hit'][str(k)] for z in valid])) for k in (1,5,10,20)},'failures_at_1':[z for z in valid if not z['hit']['1']][:10],'rows':rows}
 # Combined is a macro average across floor-valid queries, never cross-floor retrieval.
 valid_rows=[z for x in result['per_floor'].values() for z in x['rows'] if z['valid']];result['combined_macro_by_query']={f'R@{k}':float(np.mean([z['hit'][str(k)] for z in valid_rows])) for k in (1,5,10,20)}
 parity=ART/'t0_direct_h5_parity.json'; parity_rows=json.loads(parity.read_text()) if parity.exists() else []
 result['descriptor_provenance']={'status':'HISTORICAL_H5_PROXY_NOT_CURRENT_T0' if parity_rows and min(x['cosine'] for x in parity_rows)<0.9999 else 'UNVERIFIED','direct_h5_parity_path':str(parity),'direct_h5_parity':parity_rows}
 put(ART/'t0_recall_results.json',result)
 lines=['# GT0 T0 retrieval results','', '**WITHHELD AS T0 BASELINE.** Direct current-T0-versus-map-H5 checks have cosine 0.924–0.976, so the historical `global-feats-v2-mixvpr.h5` is not numerically equivalent to the current online T0 preprocessing. The table below is retained only as an audit proxy and must not be used to select a teacher/student or reported as T0.', '', '| Floor | refs | queries | zero-positive excluded | historical-H5 proxy R@1 | proxy R@5 | proxy R@10 | proxy R@20 |','|---|---:|---:|---:|---:|---:|---:|---:|']
 for f,x in result['per_floor'].items():r=x['recall'];lines.append(f"| {f} | {x['reference_count']} | {x['query_count']} | {x['zero_positive_count']} | {r['R@1']:.4f} | {r['R@5']:.4f} | {r['R@10']:.4f} | {r['R@20']:.4f} |")
 r=result['combined_macro_by_query'];lines += ['',f"Historical-H5 proxy combined: R@1 `{r['R@1']:.4f}`, R@5 `{r['R@5']:.4f}`, R@10 `{r['R@10']:.4f}`.",'', 'Valid current T0 Recall@1/@5/@10: **NOT VALIDATED**. Rebuild both frozen GT0 reference and query descriptors using the exact current online preprocessing before computing it.', '', '## Deterministic failure inspection','', 'The first ten proxy R@1 failures per floor, including query, Top-20 references and pose-positive count, are stored in `artifacts/edge_vpr_students/gt0/t0_recall_results.json`. They are not model-quality evidence. Visual cause labels require human inspection and are intentionally not inferred automatically.']
 (OUT/'gt0_t0_recall_results.md').write_text('\n'.join(lines)+'\n');print('GT0 T0',result['combined_macro_by_query'])
if __name__=='__main__':main()
