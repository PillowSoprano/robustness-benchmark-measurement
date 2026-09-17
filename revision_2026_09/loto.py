from pathlib import Path
import sys,json,csv
import numpy as np
A=Path(__file__).resolve().parent;sys.path.insert(0,str(A))
from reference_common import REFERENCES,crossing
out=[]
for dataset in ['mbr_admission','mbr_reference']:
 z=np.load(A/'frozen_losses'/('mbr' if dataset=='mbr_reference' else dataset)/'errors.npz');cl=z['clusters'];groups=np.unique(cl)
 tables={tuple(k.split('|')):z[k] for k in z.files if k!='clusters'}
 members=sorted({k[0] for k in tables if k[0] not in REFERENCES});doms=sorted({k[1] for k in tables})
 for d in doms:
  levels=sorted({float(k[2]) for k in tables if k[1]==d})
  for ref in REFERENCES:
   for m in members:
    curves=[];crosses=[]
    for dropped in groups:
     keep=cl!=dropped
     y=[float(tables[ref,d,str(s)][keep].mean()/tables[m,d,str(s)][keep].mean()) for s in levels]
     curves.append(y);crosses.append(crossing(levels,y))
    finite=[x for x in crosses if x is not None]
    out.append(dict(dataset=dataset,reference=ref,model=m,domain=d,levels=levels,omitted_trajectories=groups.tolist(),skills=curves,clean_min=min(c[0] for c in curves),clean_max=max(c[0] for c in curves),crossings=crosses,no_crossing_count=crosses.count(None),crossing_min=min(finite) if finite else None,crossing_max=max(finite) if finite else None))
(A/'summary/loto.json').write_text(json.dumps(out,indent=2))
with (A/'summary/loto_summary.csv').open('w') as f:
 cols=['dataset','reference','model','domain','clean_min','clean_max','no_crossing_count','crossing_min','crossing_max'];w=csv.DictWriter(f,fieldnames=cols,extrasaction='ignore');w.writeheader();w.writerows(out)
for r in out:
 if (r['dataset']=='mbr_admission' and r['reference']=='ar1' and r['model']=='mlp_s123') or (r['dataset']=='mbr_reference' and r['domain']=='obs_noise' and r['model'].startswith('gru') and r['reference'] in ('persistence','history_mean')):print({k:r[k] for k in ['model','reference','clean_min','clean_max','no_crossing_count','crossing_min','crossing_max']})
print('admission straddles',[(r['reference'],r['model']) for r in out if r['dataset']=='mbr_admission' and r['clean_min']<=1<=r['clean_max']])
