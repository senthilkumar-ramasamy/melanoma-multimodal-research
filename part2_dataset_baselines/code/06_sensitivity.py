"""Step 6 - near-duplicate sensitivity analysis.
Flags test images whose perceptual hash is within --flag_dist of ANY training image (different lesion), then re-computes
test metrics (same validation-derived threshold) on the remaining test images.
Usage: python 06_sensitivity.py --out OUTDIR --seed 42"""
import argparse, os
import numpy as np, pandas as pd
from common import *
ap = argparse.ArgumentParser()
ap.add_argument("--out", default="out"); ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--flag_dist", type=int, default=2); ap.add_argument("--min_sens", type=float, default=0.90)
args = ap.parse_args()
meta = pd.read_csv(os.path.join(args.out, "audit", "clean_metadata.csv"))
sp = pd.read_csv(os.path.join(args.out, "splits", f"split_seed{args.seed}.csv"))[["image_id", "partition"]]
m = meta[["image_id", "dhash"]].merge(sp, on="image_id")
def bits(h): return np.array([(int(x) >> k) & 1 for x in h for k in range(64)], dtype=np.uint8).reshape(len(h), 64)
tr = m[m.partition == "train"]; te = m[m.partition == "test"]
Btr = bits(tr.dhash.values); Bte = bits(te.dhash.values)
mind = np.array([(Btr != b).sum(1).min() for b in Bte])
flag = dict(zip(te.image_id, mind <= args.flag_dist))
print(f"test images: {len(te)}; with a training image within Hamming <= {args.flag_dist}: {int(sum(flag.values()))}")
rows = []
for name in ["img_effb0", "meta_lr", "meta_rf", "meta_hgb"]:
    f = os.path.join(args.out, "preds", f"{name}_seed{args.seed}.csv")
    if not os.path.exists(f): continue
    d = pd.read_csv(f); va = d[d.partition == "val"]; t_ = d[d.partition == "test"].copy()
    thr, ok = pick_threshold(va.label, va.prob, args.min_sens)
    t_["flag"] = t_.image_id.map(flag)
    for lab, s in [("all test images", t_), ("excluding flagged images", t_[~t_.flag])]:
        if s.label.nunique() < 2: continue
        r = metrics_at(s.label.values, s.prob.values, thr)
        rows.append(dict(model=name, subset=lab, n=len(s), n_pos=int(s.label.sum()), threshold=thr,
                         **{k: r[k] for k in ["sens", "spec", "prec", "f1", "roc_auc", "pr_auc"]}))
res = pd.DataFrame(rows); res.to_csv(os.path.join(args.out, "near_duplicate_sensitivity.csv"), index=False)
print(res.round(3).to_string(index=False)); print("Sensitivity analysis written")
