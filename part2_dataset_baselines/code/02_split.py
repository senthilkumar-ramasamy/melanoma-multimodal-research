"""Step 2 - lesion-grouped, stratified train/val/test split (about 70/15/15).
Usage: python 02_split.py --out OUTDIR --seeds 42 43 44 45 46"""
import argparse, os
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from common import *

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="out")
ap.add_argument("--seeds", type=int, nargs="+", default=[42])
args = ap.parse_args()
meta = pd.read_csv(os.path.join(args.out, "audit", "clean_metadata.csv"))
os.makedirs(os.path.join(args.out, "splits"), exist_ok=True)
summary = []
for seed in args.seeds:
    sgk = StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=seed)
    rest_idx, test_idx = next(sgk.split(meta, meta.label, meta.group_id))
    rest = meta.iloc[rest_idx]
    sgk2 = StratifiedGroupKFold(n_splits=6, shuffle=True, random_state=seed)
    tr_i, va_i = next(sgk2.split(rest, rest.label, rest.group_id))
    part = pd.Series("train", index=meta.index)
    part.iloc[test_idx] = "test"
    part.loc[rest.index[va_i]] = "val"
    out = meta[["image_id", "lesion_id", "group_id", "dx", "label"]].copy(); out["partition"] = part.values
    # leakage assertions
    gp = out.groupby("group_id").partition.nunique()
    assert (gp == 1).all(), "STOP: a group appears in more than one partition"
    lp = out.groupby("lesion_id").partition.nunique()
    assert (lp == 1).all(), "STOP: a lesion appears in more than one partition"
    out.to_csv(os.path.join(args.out, "splits", f"split_seed{seed}.csv"), index=False)
    for p in ["train", "val", "test"]:
        s = out[out.partition == p]
        summary.append(dict(seed=seed, partition=p, images=len(s), groups=s.group_id.nunique(),
                            lesions=s.lesion_id.nunique(), melanoma=int(s.label.sum()),
                            melanoma_pct=round(100 * s.label.mean(), 2)))
sm = pd.DataFrame(summary)
sm.to_csv(os.path.join(args.out, "splits", "partition_summary.csv"), index=False)
print(sm.to_string(index=False))
print("\nOK: zero shared lesion groups between partitions for every seed.")
