from pathlib import Path
import json
import numpy as np

REFERENCES=('persistence','history_mean','drift','ar1')

def fit_ar1(sequences):
    x=np.concatenate([s[:-1] for s in sequences]).astype(float)
    y=np.concatenate([s[1:] for s in sequences]).astype(float)
    xm,ym=x.mean(0),y.mean(0)
    slope=np.mean((x-xm)*(y-ym),0)/(np.mean((x-xm)**2,0)+1e-6)
    return slope,ym-slope*xm

def reference_predictions(history,horizon,ar):
    history=np.asarray(history,dtype=float)
    last=history[:,-1];h=np.arange(1,horizon+1)[None,:,None]
    out={'persistence':np.repeat(last[:,None],horizon,axis=1),
         'history_mean':np.repeat(history.mean(1)[:,None],horizon,axis=1),
         'drift':last[:,None]+h*(last-history[:,0])[:,None]/(history.shape[1]-1)}
    state=last.copy();pred=[]
    for _ in range(horizon):
        state=ar[0]*state+ar[1];pred.append(state.copy())
    out['ar1']=np.stack(pred,axis=1)
    return out

def rmse(pred,target,weights=None):
    e=np.asarray(pred,dtype=float)-np.asarray(target,dtype=float)
    if weights is not None:e=e*weights
    return np.sqrt(np.mean(e*e,axis=(1,2)))

def crossing(levels,values):
    if np.isfinite(values[0]) and values[0]<1:return 0.
    for i in range(1,len(levels)):
        a,b=values[i-1:i+1]
        if np.isfinite(a) and np.isfinite(b) and a>=1 and b<1:
            return float(levels[i-1]+(a-1)/(a-b)*(levels[i]-levels[i-1]))
    return None

def analyze(tables,clusters,dest,metadata):
    dest=Path(dest);dest.mkdir(exist_ok=True)
    np.savez_compressed(dest/'errors.npz',clusters=clusters,**{
        f'{m}|{d}|{s}':v for (m,d,s),v in tables.items()})
    groups=np.unique(clusters); rng=np.random.default_rng(20260915)
    bgroup=rng.choice(groups,size=(2000,len(groups)),replace=True)
    # All existing trajectories carry the same number of evaluation windows.
    sizes=[np.sum(clusters==g) for g in groups]
    assert len(set(sizes))==1,sizes
    idx=np.stack([np.concatenate([np.flatnonzero(clusters==g) for g in row]) for row in bgroup])
    members=sorted({m for m,_,_ in tables if m not in REFERENCES})
    domains=sorted({d for _,d,_ in tables})
    report={'metadata':metadata,'references':list(REFERENCES),'members':members,'domains':{},'curves':{},'pointwise_invariance_failures':[]}
    for d in domains:
        levels=sorted({s for _,dd,s in tables if dd==d})
        report['domains'][d]=levels
        for ref in REFERENCES:
            for m in members:
                means=[];lo=[];hi=[];boots=[];nr=[];mr=[];counts=[]
                for s in levels:
                    a=tables[(ref,d,s)];b=tables[(m,d,s)]
                    ok=np.isfinite(a)&np.isfinite(b)
                    # One mask for the complete panel, established before reference selection.
                    for other in members:assert np.array_equal(np.isfinite(tables[(other,d,s)]),ok)
                    for rr in REFERENCES:assert np.array_equal(np.isfinite(tables[(rr,d,s)]),ok)
                    counts.append(int(ok.sum()))
                    if ok.sum()<len(clusters)*0.5:
                        means.append(None);lo.append(None);hi.append(None);nr.append(None);mr.append(None);boots.append(np.full(len(idx),np.nan));continue
                    av,bv=float(a[ok].mean()),float(b[ok].mean())
                    assert av>0 and bv>0,'zero loss requires explicit handling'
                    aa,bb=a[idx],b[idx]
                    bs=np.nanmean(aa,1)/np.nanmean(bb,1)
                    means.append(av/bv);lo.append(float(np.nanpercentile(bs,2.5)));hi.append(float(np.nanpercentile(bs,97.5)));nr.append(av);mr.append(bv);boots.append(bs)
                mat=np.stack(boots,axis=1)
                vals=np.array([np.nan if x is None else x for x in means])
                bc=[crossing(levels,row) for row in mat]
                finite=[v for v in bc if v is not None]
                report['curves'][f'{ref}|{m}|{d}']={'skill':means,'ci_low':lo,'ci_high':hi,'reference_loss':nr,'model_loss':mr,'n_valid':counts,'crossing':crossing(levels,vals),'bootstrap_no_crossing_fraction':1-len(finite)/len(bc),'conditional_crossing_ci':np.percentile(finite,[2.5,97.5]).tolist() if finite else None}
        for s in levels:
            means={m:np.nanmean(tables[(m,d,s)]) if np.isfinite(tables[(m,d,s)]).sum()>=len(clusters)*0.5 else np.nan for m in members}
            if not all(np.isfinite(list(means.values()))):continue
            ordered=sorted(members,key=lambda m:means[m])
            for ref in REFERENCES:
                a=np.nanmean(tables[(ref,d,s)])
                rank=sorted(members,key=lambda m:-a/means[m])
                if rank!=ordered:report['pointwise_invariance_failures'].append([d,s,ref])
    assert not report['pointwise_invariance_failures']
    (dest/'results.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    return report
