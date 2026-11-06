"""Shared helpers for the dataset audit and baseline pipeline (HAM10000, melanoma vs. non-melanoma)."""
import json, hashlib, os
import numpy as np, pandas as pd
from sklearn.metrics import (roc_auc_score, average_precision_score, brier_score_loss,
                             confusion_matrix, f1_score)

REQUIRED_COLS = ["lesion_id", "image_id", "dx", "dx_type", "age", "sex", "localization"]
POSITIVE_DX = "mel"
META_NUM = ["age"]
META_CAT = ["sex", "localization"]          # dx_type is deliberately NOT a model input (see paper, Sec. VII-A)

def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()

def dhash(img, size=8):
    """64-bit difference hash (perceptual) from a PIL image."""
    g = img.convert("L").resize((size + 1, size))
    a = np.asarray(g, dtype=np.int16)
    bits = (a[:, 1:] > a[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)

def hamming(a, b):
    return bin(a ^ b).count("1")

def write_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))

def pick_threshold(y, p, min_sens=0.90):
    """Highest-specificity threshold with sensitivity >= min_sens on VALIDATION data.
    If unattainable, returns the threshold with the best achievable sensitivity (flagged)."""
    y = np.asarray(y); p = np.asarray(p)
    cands = np.unique(np.concatenate([p, [0.0, 1.0]]))
    best = None; fallback = None
    for t in cands:
        pred = p >= t
        tp = ((pred) & (y == 1)).sum(); fn = ((~pred) & (y == 1)).sum()
        tn = ((~pred) & (y == 0)).sum(); fp = ((pred) & (y == 0)).sum()
        sens = tp / max(tp + fn, 1); spec = tn / max(tn + fp, 1)
        if sens >= min_sens and (best is None or spec > best[1] or (spec == best[1] and t > best[0])):
            best = (t, spec, sens)
        if fallback is None or sens > fallback[2] or (sens == fallback[2] and spec > fallback[1]):
            fallback = (t, spec, sens)
    if best is not None:
        return float(best[0]), True
    return float(fallback[0]), False

def metrics_at(y, p, t):
    y = np.asarray(y); p = np.asarray(p); pred = (p >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    sens = tp / max(tp + fn, 1); spec = tn / max(tn + fp, 1)
    prec = tp / max(tp + fp, 1)
    f1 = 0.0 if (prec + sens) == 0 else 2 * prec * sens / (prec + sens)
    out = dict(sens=sens, spec=spec, prec=prec, f1=f1, bal_acc=(sens + spec) / 2,
               acc=(tp + tn) / len(y), tp=int(tp), fp=int(fp), tn=int(tn), fn=int(fn))
    if len(np.unique(y)) == 2:
        out.update(roc_auc=roc_auc_score(y, p), pr_auc=average_precision_score(y, p),
                   brier=brier_score_loss(y, p))
    else:
        out.update(roc_auc=np.nan, pr_auc=np.nan, brier=np.nan)
    return out
