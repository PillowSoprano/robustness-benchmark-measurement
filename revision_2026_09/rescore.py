"""Reproduce all core reference ratios and bootstrap intervals from frozen losses."""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
from reference_common import analyze
ROOT=Path(__file__).resolve().parent

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--system',choices=['mbr','tep','mbr_admission','tep_admission','tep_bias_cap','both'],default='both')
    parser.add_argument('--output',type=Path,default=ROOT/'recomputed')
    args=parser.parse_args()
    manifest=json.loads((ROOT/'frozen_losses/MANIFEST.json').read_text())
    for name,digest in manifest.items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
    for system in ['mbr','tep','mbr_admission','tep_admission','tep_bias_cap'] if args.system=='both' else [args.system]:
        base=ROOT/'frozen_losses'/system
        z=np.load(base/'errors.npz',allow_pickle=False)
        tables={}
        for k in z.files:
            if k=='clusters':continue
            model,domain,severity=k.split('|')
            tables[(model,domain,float(severity))]=z[k]
        expected=json.loads((base/'results.json').read_text())
        dest=args.output/system;dest.parent.mkdir(parents=True,exist_ok=True)
        actual=analyze(tables,z['clusters'],dest,expected['metadata'])
        assert actual==expected,system+' results differ from frozen release'
        print(system+': all curves, intervals and censoring exactly reproduced',flush=True)
if __name__=='__main__':main()
