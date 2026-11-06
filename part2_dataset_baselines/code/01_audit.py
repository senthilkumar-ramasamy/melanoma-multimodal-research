"""Step 1 - data audit. Writes audit/ with clean metadata, group ids, and a JSON report.
Usage: python 01_audit.py --meta PATH/HAM10000_metadata.csv --img_dirs DIR1 [DIR2 ...] --out OUTDIR"""
import argparse, os, sys, itertools
import numpy as np, pandas as pd
from PIL import Image
from common import *

ap = argparse.ArgumentParser()
ap.add_argument("--meta", required=True)
ap.add_argument("--img_dirs", nargs="+", required=True)
ap.add_argument("--out", default="out")
ap.add_argument("--near_dup_dist", type=int, default=2, help="max dHash Hamming distance at which cross-lesion pairs are FLAGGED and listed")
ap.add_argument("--merge_dist", type=int, default=0, help="max dHash Hamming distance at which flagged pairs are MERGED into one group (default 0: identical hashes only; larger values chain unrelated look-alike images into giant groups)")
ap.add_argument("--ext", default=".jpg")
args = ap.parse_args()
os.makedirs(os.path.join(args.out, "audit"), exist_ok=True)
rep = {}

meta = pd.read_csv(args.meta)
missing_cols = [c for c in REQUIRED_COLS if c not in meta.columns]
if missing_cols:
    sys.exit(f"STOP: expected column(s) {missing_cols} not found. Columns present: {list(meta.columns)}. "
             "Tell Claude the actual column names before continuing.")
rep["metadata_rows"] = len(meta)
rep["metadata_columns"] = list(meta.columns)
rep["duplicate_image_ids_in_metadata"] = int(meta.image_id.duplicated().sum())

# locate image files
paths = {}
for d in args.img_dirs:
    for fn in os.listdir(d):
        if fn.lower().endswith((".jpg", ".jpeg", ".png")):
            paths.setdefault(os.path.splitext(fn)[0], os.path.join(d, fn))
rep["image_files_found"] = len(paths)
rep["metadata_rows_without_image"] = int((~meta.image_id.isin(paths)).sum())
rep["image_files_without_metadata_row"] = int(len(set(paths) - set(meta.image_id)))
log = []   # exclusion log

meta["path"] = meta.image_id.map(paths)
for _, r in meta[meta.path.isna()].iterrows():
    log.append((r.image_id, "E2_no_image_file"))
meta = meta[meta.path.notna()].copy()

# decode every image, hash, dhash
widths, heights, sha, dh, bad = [], [], [], [], []
for i, p in enumerate(meta.path):
    try:
        with Image.open(p) as im:
            im.load(); widths.append(im.width); heights.append(im.height)
            dh.append(dhash(im))
        sha.append(sha256_file(p))
    except Exception as e:
        bad.append(meta.image_id.iloc[i]); widths.append(0); heights.append(0); sha.append(""); dh.append(0)
    if (i + 1) % 2000 == 0:
        print(f"  decoded {i+1}/{len(meta)}", flush=True)
meta["width"], meta["height"], meta["sha256"], meta["dhash"] = widths, heights, sha, dh
for b in bad: log.append((b, "E1_undecodable"))
meta = meta[~meta.image_id.isin(bad)].copy()
rep["undecodable_images"] = len(bad)
rep["image_size_counts"] = meta.groupby(["width", "height"]).size().sort_values(ascending=False).head(10).to_dict().__repr__()

# labels
meta["dx"] = meta["dx"].astype(str).str.strip()
unlabeled = meta[meta.dx.isin(["", "nan", "None"])]
for b in unlabeled.image_id: log.append((b, "E2_no_label"))
meta = meta[~meta.image_id.isin(unlabeled.image_id)].copy()
meta["label"] = (meta.dx == POSITIVE_DX).astype(int)

# exact duplicates (keep first)
dup = meta[meta.sha256.duplicated(keep="first")]
for b in dup.image_id: log.append((b, "E3_exact_duplicate"))
rep["exact_duplicate_images_removed"] = len(dup)
meta = meta[~meta.image_id.isin(dup.image_id)].copy()

# conflicting labels within a lesion
conf = meta.groupby("lesion_id").dx.nunique()
conf_ids = conf[conf > 1].index
rep["lesions_with_conflicting_labels"] = int(len(conf_ids))
for b in meta[meta.lesion_id.isin(conf_ids)].image_id: log.append((b, "E4_conflicting_lesion_label"))
meta = meta[~meta.lesion_id.isin(conf_ids)].copy()

# near-duplicate pairs across different lesions (dHash)
meta = meta.reset_index(drop=True)
hs = meta.dhash.values; les = meta.lesion_id.values
pairs = []
order = np.argsort(hs)
# brute force with early skip; fine for ~10k images (50M compares in numpy chunks)
arr = np.array([[(h >> k) & 1 for k in range(64)] for h in hs], dtype=np.uint8)
for i in range(len(meta)):
    d = (arr[i + 1:] != arr[i]).sum(1)
    for j in np.where(d <= args.near_dup_dist)[0]:
        jj = i + 1 + j
        if les[i] != les[jj]:
            pairs.append((meta.image_id[i], meta.image_id[jj], int(d[j])))
pd.DataFrame(pairs, columns=["image_a", "image_b", "hamming"]).to_csv(
    os.path.join(args.out, "audit", "near_duplicate_pairs_across_lesions.csv"), index=False)
rep["near_duplicate_pairs_across_lesions"] = len(pairs)
rep["flag_dist"] = args.near_dup_dist; rep["merge_dist"] = args.merge_dist
rep["near_duplicate_pairs_by_hamming"] = {str(k): int(sum(1 for p_ in pairs if p_[2] == k)) for k in range(args.near_dup_dist + 1)}

# groups: union-find on lesion ids (+ near-duplicate links)
parent = {l: l for l in meta.lesion_id.unique()}
def find(x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]; x = parent[x]
    return x
img2les = dict(zip(meta.image_id, meta.lesion_id))
def _largest(maxd):
    par = {l: l for l in meta.lesion_id.unique()}
    def fd(x):
        while par[x] != x:
            par[x] = par[par[x]]; x = par[x]
        return x
    for a_, b_, d_ in pairs:
        if d_ <= maxd:
            ra_, rb_ = fd(img2les[a_]), fd(img2les[b_])
            if ra_ != rb_: par[ra_] = rb_
    return int(meta.groupby(meta.lesion_id.map(fd)).size().max())
rep["largest_group_if_merged_at_flag_dist"] = _largest(args.near_dup_dist)
for a, b, d_ in pairs:
    if d_ > args.merge_dist: continue
    ra, rb = find(img2les[a]), find(img2les[b])
    if ra != rb: parent[ra] = rb
meta["group_id"] = meta.lesion_id.map(find)
gs = meta.groupby("group_id").size()
rep["n_groups"] = int(len(gs)); rep["largest_group_images"] = int(gs.max())
if gs.max() > 50:
    print("WARNING: a merged group has > 50 images; near-duplicate merging may be too aggressive. "
          "Re-run with a smaller --merge_dist and tell Claude.")

# metadata missingness (on retained rows)
rep["missingness"] = {c: int(meta[c].isna().sum()) for c in ["age", "sex", "localization"]}
rep["sex_values"] = meta.sex.value_counts(dropna=False).to_dict()
rep["unknown_text_values"] = {c: int((meta[c].astype(str).str.lower() == "unknown").sum()) for c in ["sex", "localization"]}
rep["age_range"] = [float(meta.age.min()), float(meta.age.max())]
rep["dx_counts"] = meta.dx.value_counts().to_dict()
rep["dx_type_counts"] = meta.dx_type.value_counts().to_dict()
rep["n_images_retained"] = int(len(meta)); rep["n_lesions_retained"] = int(meta.lesion_id.nunique())
rep["n_positive"] = int(meta.label.sum())
rep["images_per_lesion"] = meta.groupby("lesion_id").size().describe().to_dict()
rep["histopathology_share"] = float((meta.dx_type == "histo").mean())
pd.DataFrame(log, columns=["image_id", "rule"]).to_csv(os.path.join(args.out, "audit", "exclusion_log.csv"), index=False)
rep["exclusions_by_rule"] = pd.Series([r for _, r in log]).value_counts().to_dict() if log else {}
meta.drop(columns=["dhash"]).assign(dhash=meta.dhash.astype(str)).to_csv(os.path.join(args.out, "audit", "clean_metadata.csv"), index=False)
write_json(rep, os.path.join(args.out, "audit", "audit_report.json"))
import json; print(json.dumps(rep, indent=2, default=str))
print("\nAudit done. Files in", os.path.join(args.out, "audit"))
