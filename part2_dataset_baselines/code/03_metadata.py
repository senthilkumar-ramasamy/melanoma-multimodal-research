"""Step 3 - metadata-only baselines (LR, RF, HGB) with grouped 5-fold CV tuning on TRAIN.
Writes predictions: out/preds/meta_<model>_seed<S>.csv
Usage: python 03_metadata.py --out OUTDIR --seeds 42 43 44 45 46"""
import argparse, os, time
import numpy as np, pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from common import *

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="out"); ap.add_argument("--seeds", type=int, nargs="+", default=[42])
ap.add_argument("--n_jobs", type=int, default=2)
args = ap.parse_args()
meta = pd.read_csv(os.path.join(args.out, "audit", "clean_metadata.csv"))
os.makedirs(os.path.join(args.out, "preds"), exist_ok=True)

def make_pre(scale):
    num = [("imp", SimpleImputer(strategy="median", add_indicator=True))]
    if scale: num.append(("sc", StandardScaler()))
    cat = [("imp", SimpleImputer(strategy="constant", fill_value="unknown")),
           ("oh", OneHotEncoder(handle_unknown="ignore"))]
    return ColumnTransformer([("num", Pipeline(num), META_NUM), ("cat", Pipeline(cat), META_CAT)], sparse_threshold=0.0)  # dense output: required by HistGradientBoosting

def make_hgb():
    try: return HistGradientBoostingClassifier(class_weight="balanced", random_state=0)
    except TypeError: return HistGradientBoostingClassifier(random_state=0)   # older scikit-learn: no class_weight

# treat literal "unknown" strings as their own category (kept), true NaN -> "unknown" via imputer
MODELS = {
    "lr":  (True,  LogisticRegression(class_weight="balanced", max_iter=2000),
            {"clf__C": [0.01, 0.1, 1, 10]}),
    "rf":  (False, RandomForestClassifier(class_weight="balanced", n_jobs=1, random_state=0),
            {"clf__n_estimators": [200, 500], "clf__max_depth": [None, 5, 10], "clf__min_samples_leaf": [1, 5]}),
    "hgb": (False, make_hgb(),
            {"clf__learning_rate": [0.05, 0.1], "clf__max_depth": [3, None], "clf__max_iter": [100, 300]}),
}
for seed in args.seeds:
    sp = pd.read_csv(os.path.join(args.out, "splits", f"split_seed{seed}.csv"))
    d = sp.merge(meta[["image_id"] + META_NUM + META_CAT], on="image_id")
    tr = d[d.partition == "train"]
    for name, (scale, est, grid) in MODELS.items():
        t0 = time.time()
        if os.path.exists(os.path.join(args.out, "preds", f"meta_{name}_seed{seed}_cfg.json")):
            print(f"seed {seed} {name}: already done, skipped", flush=True); continue
        pipe = Pipeline([("pre", make_pre(scale)), ("clf", est)])
        cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
        gs = GridSearchCV(pipe, grid, scoring="average_precision", cv=cv, n_jobs=args.n_jobs, refit=True, error_score="raise")
        gs.fit(tr[META_NUM + META_CAT], tr.label, groups=tr.group_id)
        res = d[d.partition.isin(["val", "test"])].copy()
        res["prob"] = gs.predict_proba(res[META_NUM + META_CAT])[:, 1]
        res[["image_id", "group_id", "dx", "partition", "label", "prob"]].to_csv(
            os.path.join(args.out, "preds", f"meta_{name}_seed{seed}.csv"), index=False)
        write_json({"best_params": gs.best_params_, "cv_pr_auc": gs.best_score_},
                   os.path.join(args.out, "preds", f"meta_{name}_seed{seed}_cfg.json"))
        print(f"seed {seed} {name}: cv PR-AUC={gs.best_score_:.3f} best={gs.best_params_} ({time.time()-t0:.0f}s)", flush=True)
