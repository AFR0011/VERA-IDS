#!/usr/bin/env python3
"""Check feasibility of a CICIDS2017 exact-feature-group-safe split."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

SPLITS=("train","val","test")
TARGETS={"y_stage1_attack","y_stage2_family","y_stage2_fine"}
EXPECTED_ROWS=2099879
EXPECTED_GROUPS=1735180

def parts(d):
    out=[]
    for p in ("*.parquet","*.csv","*.csv.gz"): out += sorted(d.glob(p))
    return list(dict.fromkeys(out))

def mix64(x,seed):
    x=x.astype(np.uint64,copy=True)+np.uint64(seed)+np.uint64(0x9E3779B97F4A7C15)
    x=(x^(x>>np.uint64(30)))*np.uint64(0xBF58476D1CE4E5B9)
    x=(x^(x>>np.uint64(27)))*np.uint64(0x94D049BB133111EB)
    return x^(x>>np.uint64(31))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--processed-root",default="processed_V5")
    ap.add_argument("--out-root",default="outputs/12_jisa_finalization/03_group_safe_split_feasibility")
    ap.add_argument("--seed",type=int,default=123)
    ap.add_argument("--min-support",type=int,default=200)
    a=ap.parse_args()
    root=Path(a.processed_root)/"A_stratified"/"CICIDS2017"
    out=Path(a.out_root); out.mkdir(parents=True,exist_ok=True)
    rows=[]; feat=None
    for sp in SPLITS:
        for path in parts(root/sp):
            frames=[pd.read_parquet(path)] if path.suffix==".parquet" else pd.read_csv(path,chunksize=200000)
            for df in frames:
                if feat is None: feat=[c for c in df.columns if c not in TARGETS]
                h=pd.util.hash_pandas_object(df[feat],index=False).to_numpy(dtype=np.uint64)
                rows.append(pd.DataFrame({"h":h,"family":df.y_stage2_family.astype(str).to_numpy()}))
    r=pd.concat(rows,ignore_index=True)
    if len(r)!=EXPECTED_ROWS or r.h.nunique()!=EXPECTED_GROUPS:
        raise RuntimeError(f"surface mismatch rows={len(r)} groups={r.h.nunique()}")
    meta=r.groupby("h").agg(n=("family","size"),family=("family","first"),nfam=("family","nunique")).reset_index()
    bucket=mix64(meta.h.to_numpy(dtype=np.uint64),a.seed)%np.uint64(1000000)
    meta["split"]=np.where(bucket<600000,"train",np.where(bucket<800000,"val","test"))
    def recount():
        m=pd.Series(meta.split.to_numpy(),index=meta.h.to_numpy())
        rr=r.copy(); rr["split"]=rr.h.map(m)
        tab=rr.groupby(["family","split"]).size().unstack(fill_value=0)
        for sp in SPLITS:
            if sp not in tab: tab[sp]=0
        return tab[list(SPLITS)].astype(int),rr
    support,rr=recount(); moves=[]
    for fam in support.index:
        for target in SPLITS:
            while int(support.loc[fam,target])<a.min_support:
                donors=sorted([s for s in SPLITS if s!=target and int(support.loc[fam,s])>a.min_support],key=lambda s:int(support.loc[fam,s]),reverse=True)
                done=False
                for donor in donors:
                    cand=meta[(meta.nfam==1)&(meta.family==fam)&(meta.split==donor)].sort_values(["n","h"])
                    for idx,row in cand.iterrows():
                        n=int(row.n)
                        if int(support.loc[fam,donor])-n<a.min_support: continue
                        meta.at[idx,"split"]=target
                        support.loc[fam,donor]-=n; support.loc[fam,target]+=n
                        moves.append({"family":str(fam),"from":donor,"to":target,"rows":n})
                        done=True; break
                    if done: break
                if not done: raise RuntimeError(f"cannot repair {fam} {target}")
    support,rr=recount()
    totals=rr.groupby("split").size().to_dict()
    support_out=support.reset_index(); support_out["min_support"]=support_out[list(SPLITS)].min(axis=1)
    support_out.to_csv(out/"split_family_support.csv",index=False)
    pd.DataFrame(moves).to_csv(out/"support_repair_moves.csv",index=False)
    attacks=[f for f in support.index if f!="Benign"]
    elig=[]
    for h in attacks:
        other=support.drop(index=[h,"Benign"],errors="ignore")
        ok=int(support.loc[h,"val"])>=a.min_support and int(support.loc[h,"test"])>=a.min_support and bool((other.min(axis=1)>=a.min_support).all())
        elig.append({"holdout_family":h,"unknown_val":int(support.loc[h,"val"]),"unknown_test":int(support.loc[h,"test"]),"eligible":ok})
    pd.DataFrame(elig).to_csv(out/"eligible_holdouts.csv",index=False)
    summary={"rows":len(r),"unique_exact_feature_groups":len(meta),"family_conflict_groups":int((meta.nfam>1).sum()),"repair_moves":len(moves),"repair_rows":int(sum(x["rows"] for x in moves)),"split_rows":{s:int(totals.get(s,0)) for s in SPLITS},"split_fractions":{s:float(totals.get(s,0)/len(r)) for s in SPLITS},"zero_cross_split_exact_feature_overlap":bool(meta.groupby("h").split.nunique().max()==1),"all_families_supported":bool((support.min(axis=1)>=a.min_support).all()),"eligible_holdouts":int(sum(x["eligible"] for x in elig)),"provenance_boundary":"Raw day/file separation is not preserved on this fallback surface."}
    (out/"group_safe_split_feasibility_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(json.dumps(summary,indent=2))
    if not(summary["zero_cross_split_exact_feature_overlap"] and summary["all_families_supported"] and summary["eligible_holdouts"]==len(attacks)): raise SystemExit(2)
if __name__=="__main__": main()
