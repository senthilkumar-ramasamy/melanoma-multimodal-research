"""Step 5 - evaluation: thresholds from VALIDATION, test metrics, lesion-clustered bootstrap CIs, figures.
Usage: python 05_evaluate.py --out OUTDIR --seed 42 --extra_seeds 43 44 45 46"""
import argparse, os, glob
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import roc_curve, precision_recall_curve
from common import *

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="out"); ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--extra_seeds", type=int, nargs="*", default=[])
ap.add_argument("--n_boot", type=int, default=2000)
ap.add_argument("--min_sens", type=float, default=0.90)
args = ap.parse_args()
plt.rcParams.update({"font.family": "serif", "font.size": 8})
FIG = os.path.join(args.out, "figures"); os.makedirs(FIG, exist_ok=True)
MODELS = {"img_effb0": "Image-only (EfficientNet-B0)", "meta_lr": "Metadata-only: logistic regression",
          "meta_rf": "Metadata-only: random forest", "meta_hgb": "Metadata-only: gradient boosting"}
COL = {"img_effb0": "#C00000", "meta_lr": "#2E75B6", "meta_rf": "#548235", "meta_hgb": "#ED7D31"}
rng = np.random.RandomState(123)

def load(model, seed):
    f = os.path.join(args.out, "preds", f"{model}_seed{seed}.csv")
    return pd.read_csv(f) if os.path.exists(f) else None

def evaluate(df):
    va, te = df[df.partition == "val"], df[df.partition == "test"]
    t, ok = pick_threshold(va.label, va.prob, args.min_sens)
    return t, ok, metrics_at(te.label, te.prob, t), te

KEYS = ["sens", "spec", "prec", "f1", "roc_auc", "pr_auc", "bal_acc", "acc", "brier"]
def bootstrap(te, t):
    groups = [g.index.values for _, g in te.groupby("group_id")]
    y = te.label.values; p = te.prob.values; vals = {k: [] for k in KEYS}
    for _ in range(args.n_boot):
        pick = rng.randint(0, len(groups), len(groups))
        idx = np.concatenate([groups[i] for i in pick])
        pos = te.index.get_indexer(idx)
        m = metrics_at(y[pos], p[pos], t)
        for k in KEYS: vals[k].append(m[k])
    return {k: (np.nanpercentile(v, 2.5), np.nanpercentile(v, 97.5)) for k, v in vals.items()}

rows, main, fp_rows, store = [], {}, [], {}
for m, label in MODELS.items():
    df = load(m, args.seed)
    if df is None: print("missing predictions for", m, "(skipped)"); continue
    df = df.reset_index(drop=True)
    t, ok, met, te = evaluate(df); te = te.reset_index(drop=True)
    ci = bootstrap(te, t); store[m] = (df, te, t, met)
    main[m] = dict(threshold=t, sens_target_met_on_val=ok, **met, ci={k: list(v) for k, v in ci.items()})
    rows.append([label] + [f"{met[k]:.3f} [{ci[k][0]:.3f}, {ci[k][1]:.3f}]" for k in KEYS])
    neg = te[te.label == 0].copy(); neg["fp"] = (neg.prob >= t).astype(int)
    for dx, g in neg.groupby("dx"): fp_rows.append([label, dx, len(g), int(g.fp.sum()), g.fp.mean()])
    print(f"{label}: thr={t:.3f} (val sens target met: {ok})  test sens={met['sens']:.3f} spec={met['spec']:.3f} PR-AUC={met['pr_auc']:.3f}")
header = ["Model", "Sens.", "Spec.", "Prec.", "F1", "ROC-AUC", "PR-AUC", "Bal. acc.", "Acc.", "Brier"]
pd.DataFrame(rows, columns=header).to_csv(os.path.join(args.out, "results_table_V.csv"), index=False)
pd.DataFrame(fp_rows, columns=["model", "dx", "n_negatives", "false_positives", "fp_rate"]).to_csv(os.path.join(args.out, "false_positive_rates.csv"), index=False)
write_json(main, os.path.join(args.out, "results_main.json"))

# repeated runs summary (other seeds, same protocol, own validation thresholds)
if args.extra_seeds:
    rep = []
    for m, label in MODELS.items():
        ms = []
        for s in [args.seed] + args.extra_seeds:
            df = load(m, s)
            if df is None: continue
            t, ok, met, _ = evaluate(df); ms.append(met)
        if len(ms) > 1:
            rep.append([label, len(ms)] + [f"{np.nanmean([x[k] for x in ms]):.3f} ± {np.nanstd([x[k] for x in ms], ddof=1):.3f}" for k in KEYS])
    pd.DataFrame(rep, columns=["Model", "n_runs"] + header[1:]).to_csv(os.path.join(args.out, "results_repeated_runs.csv"), index=False)

# ---------- figures ----------
def sv(name): plt.tight_layout(); plt.savefig(os.path.join(FIG, name), dpi=300); plt.close()
fig, ax = plt.subplots(1, 2, figsize=(7, 3.2))
for m, (df, te, t, met) in store.items():
    fpr, tpr, _ = roc_curve(te.label, te.prob); ax[0].plot(fpr, tpr, color=COL[m], label=f"{MODELS[m]} (AUC {met['roc_auc']:.2f})")
    pr, rc, _ = precision_recall_curve(te.label, te.prob); ax[1].plot(rc, pr, color=COL[m], label=f"{MODELS[m]} (AP {met['pr_auc']:.2f})")
ax[0].plot([0, 1], [0, 1], "--", color="grey"); ax[0].set_xlabel("False-positive rate"); ax[0].set_ylabel("Sensitivity"); ax[0].set_title("ROC (test)")
ax[1].axhline(te.label.mean(), ls="--", color="grey"); ax[1].set_xlabel("Recall (sensitivity)"); ax[1].set_ylabel("Precision"); ax[1].set_title("Precision-recall (test)")
for a in ax: a.legend(fontsize=6, frameon=False); a.spines[["top", "right"]].set_visible(False)
sv("fig_roc_pr.png")

n = len(store); fig, ax = plt.subplots(1, n, figsize=(1.9 * n, 2.1))
ax = np.atleast_1d(ax)
for a, (m, (df, te, t, met)) in zip(ax, store.items()):
    cm = np.array([[met["tn"], met["fp"]], [met["fn"], met["tp"]]]); a.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2): a.text(j, i, str(cm[i, j]), ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
    a.set_xticks([0, 1]); a.set_yticks([0, 1]); a.set_xticklabels(["Non-mel", "Mel"]); a.set_yticklabels(["Non-mel", "Mel"])
    a.set_title(MODELS[m].replace("Metadata-only: ", "Meta: ").replace(" (EfficientNet-B0)", ""), fontsize=6.5); a.set_xlabel("Predicted"); a.set_ylabel("Actual")
sv("fig_confusion.png")

fig, ax = plt.subplots(figsize=(3.4, 3.1)); ax.plot([0, 1], [0, 1], "--", color="grey", label="Perfect")
for m, (df, te, t, met) in store.items():
    bins = np.quantile(te.prob, np.linspace(0, 1, 6)); bins[0] -= 1e-9; bins[-1] += 1e-9
    b = np.digitize(te.prob, np.unique(bins)[1:-1]); xs = [te.prob[b == k].mean() for k in np.unique(b)]; ys = [te.label[b == k].mean() for k in np.unique(b)]
    ax.plot(xs, ys, "o-", color=COL[m], ms=3, label=MODELS[m])
ax.set_xlabel("Mean predicted probability"); ax.set_ylabel("Observed melanoma fraction"); ax.legend(fontsize=5.5, frameon=False); ax.spines[["top", "right"]].set_visible(False)
sv("fig_reliability.png")

fp = pd.DataFrame(fp_rows, columns=["model", "dx", "n", "fp", "rate"])
if len(fp):
    piv = fp.pivot(index="dx", columns="model", values="rate")
    fig, ax = plt.subplots(figsize=(3.6, 2.6)); piv.plot.bar(ax=ax, color=[COL[[k for k, v in MODELS.items() if v == c][0]] for c in piv.columns], width=0.8)
    ax.set_ylabel("False-positive rate (test)"); ax.set_xlabel("True diagnostic category"); ax.legend(fontsize=5, frameon=False); ax.spines[["top", "right"]].set_visible(False)
    plt.xticks(rotation=0); sv("fig_fp_by_category.png")

# class distribution + partitions + real-image gallery
rep_ = json.load(open(os.path.join(args.out, "audit", "audit_report.json")))
sp = pd.read_csv(os.path.join(args.out, "splits", f"split_seed{args.seed}.csv"))
tab = sp.groupby(["partition", "dx"]).size().unstack(0).fillna(0)[["train", "val", "test"]].sort_values("train", ascending=False)
fig, ax = plt.subplots(figsize=(3.6, 2.5)); tab.plot.barh(stacked=True, ax=ax, color=["#2E75B6", "#70AD47", "#ED7D31"]); ax.invert_yaxis()
ax.set_xlabel("Images (audited copy)"); ax.set_ylabel(""); ax.legend(fontsize=6, frameon=False); ax.spines[["top", "right"]].set_visible(False); sv("fig_partition_composition.png")
meta = pd.read_csv(os.path.join(args.out, "audit", "clean_metadata.csv"))
order = ["nv", "mel", "bkl", "bcc", "akiec", "vasc", "df"]; found = [d for d in order if d in set(meta.dx)]
fig, ax = plt.subplots(1, len(found), figsize=(1.35 * len(found), 1.6))
for a, dx in zip(np.atleast_1d(ax), found):
    r = meta[meta.dx == dx].sort_values("image_id").iloc[0]; a.imshow(Image.open(r.path).convert("RGB")); a.axis("off"); a.set_title(dx, fontsize=8)
sv("fig_real_samples.png")
print("\nResults and figures written to", args.out)
