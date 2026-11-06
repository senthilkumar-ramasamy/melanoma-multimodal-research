"""Step 4 - image-only baseline: EfficientNet-B0 (ImageNet-pretrained), two-phase training.
Writes out/preds/img_effb0_seed<S>.csv and a config/log JSON.
Usage: python 04_image.py --out OUTDIR --seed 42 [--epochs1 3 --epochs2 30 --bs 32]"""
import argparse, os, time, json, random
import numpy as np, pandas as pd
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision.transforms import v2 as T
from sklearn.metrics import average_precision_score
from PIL import Image
from common import write_json

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="out"); ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--epochs1", type=int, default=3); ap.add_argument("--epochs2", type=int, default=30)
ap.add_argument("--patience", type=int, default=5); ap.add_argument("--bs", type=int, default=32)
ap.add_argument("--lr1", type=float, default=1e-3); ap.add_argument("--lr2", type=float, default=1e-4)
ap.add_argument("--wd", type=float, default=1e-5); ap.add_argument("--size", type=int, default=224)
ap.add_argument("--workers", type=int, default=2)
ap.add_argument("--no_pretrained", action="store_true", help="TESTING ONLY (random init); never use for reported results")
ap.add_argument("--max_images", type=int, default=0, help="TESTING ONLY: subsample for a quick smoke test")
args = ap.parse_args()

def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
seed_all(args.seed)
torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
dev = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", dev, "| torch", torch.__version__, "| torchvision", torchvision.__version__, flush=True)

meta = pd.read_csv(os.path.join(args.out, "audit", "clean_metadata.csv"))
sp = pd.read_csv(os.path.join(args.out, "splits", f"split_seed{args.seed}.csv"))
d = sp.merge(meta[["image_id", "path"]], on="image_id")
if args.max_images:
    d = d.groupby("partition", group_keys=False).apply(lambda x: x.sample(min(len(x), args.max_images), random_state=0))

# cache resized images once (uint8) so training is not bound by JPEG decoding
cache = os.path.join(args.out, f"cache_{args.size}.npy"); idx_file = os.path.join(args.out, f"cache_{args.size}_ids.csv")
need = meta.image_id.tolist()
if not (os.path.exists(cache) and os.path.exists(idx_file) and pd.read_csv(idx_file).image_id.tolist() == need):
    print("building image cache ...", flush=True)
    arr = np.lib.format.open_memmap(cache, mode="w+", dtype=np.uint8, shape=(len(meta), args.size, args.size, 3))
    for i, p in enumerate(meta.path):
        with Image.open(p) as im:
            arr[i] = np.asarray(im.convert("RGB").resize((args.size, args.size), Image.BILINEAR))
        if (i + 1) % 2000 == 0: print(f"  cached {i+1}/{len(meta)}", flush=True)
    arr.flush(); pd.DataFrame({"image_id": need}).to_csv(idx_file, index=False)
arr = np.load(cache, mmap_mode="r")
row_of = {k: i for i, k in enumerate(need)}

mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
train_tf = T.Compose([T.RandomHorizontalFlip(), T.RandomVerticalFlip(), T.RandomRotation(180),
                      T.RandomResizedCrop(args.size, scale=(0.8, 1.0), antialias=True),
                      T.ColorJitter(brightness=0.1, contrast=0.1), T.ToDtype(torch.float32, scale=True), T.Normalize(mean, std)])
eval_tf = T.Compose([T.ToDtype(torch.float32, scale=True), T.Normalize(mean, std)])
class DS(Dataset):
    def __init__(s, df, tf): s.ids = df.image_id.tolist(); s.y = df.label.values.astype(np.float32); s.tf = tf
    def __len__(s): return len(s.ids)
    def __getitem__(s, i):
        x = torch.from_numpy(np.array(arr[row_of[s.ids[i]]])).permute(2, 0, 1)
        return s.tf(x), s.y[i]
tr, va, te = (d[d.partition == p] for p in ["train", "val", "test"])
g = torch.Generator(); g.manual_seed(args.seed)
mk = lambda df, tf, sh: DataLoader(DS(df, tf), batch_size=args.bs, shuffle=sh, num_workers=args.workers, generator=g if sh else None, pin_memory=(dev == "cuda"))
dl_tr, dl_va, dl_te = mk(tr, train_tf, True), mk(va, eval_tf, False), mk(te, eval_tf, False)

w = torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1 if not args.no_pretrained else None
model = torchvision.models.efficientnet_b0(weights=w)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)
model.to(dev)
pos_w = torch.tensor([(tr.label == 0).sum() / max((tr.label == 1).sum(), 1)], dtype=torch.float32, device=dev)
crit = nn.BCEWithLogitsLoss(pos_weight=pos_w)

@torch.no_grad()
def predict(dl):
    model.eval(); ps = []
    for x, _ in dl: ps.append(torch.sigmoid(model(x.to(dev))).squeeze(1).cpu().numpy())
    return np.concatenate(ps)

def run_phase(params, lr, epochs, tag, state):
    opt = torch.optim.Adam(params, lr=lr, weight_decay=args.wd)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=2)
    bad = 0
    for ep in range(epochs):
        model.train(); t0 = time.time(); tot = 0
        for x, y in dl_tr:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad(); loss = crit(model(x).squeeze(1), y); loss.backward(); opt.step(); tot += loss.item() * len(y)
        p = predict(dl_va); ap_ = average_precision_score(va.label, p) if va.label.nunique() == 2 else float("nan")
        sch.step(ap_ if ap_ == ap_ else 0)
        print(f"[{tag}] epoch {ep+1}/{epochs} loss={tot/len(tr):.4f} val PR-AUC={ap_:.4f} ({time.time()-t0:.0f}s)", flush=True)
        state["log"].append(dict(phase=tag, epoch=ep + 1, loss=tot / len(tr), val_pr_auc=ap_))
        if ap_ > state["best"]:
            state["best"] = ap_; bad = 0
            torch.save(model.state_dict(), os.path.join(args.out, f"best_img_seed{args.seed}.pt"))
        else:
            bad += 1
            if tag == "finetune" and bad >= args.patience: print("early stopping"); break

state = {"best": -1, "log": []}
for p in model.parameters(): p.requires_grad = False
for p in model.classifier.parameters(): p.requires_grad = True
run_phase(model.classifier.parameters(), args.lr1, args.epochs1, "head", state)
for p in model.parameters(): p.requires_grad = True
run_phase(model.parameters(), args.lr2, args.epochs2, "finetune", state)

model.load_state_dict(torch.load(os.path.join(args.out, f"best_img_seed{args.seed}.pt"), map_location=dev))
os.makedirs(os.path.join(args.out, "preds"), exist_ok=True)
res = pd.concat([va.assign(prob=predict(dl_va)), te.assign(prob=predict(dl_te))])
res[["image_id", "group_id", "dx", "partition", "label", "prob"]].to_csv(os.path.join(args.out, "preds", f"img_effb0_seed{args.seed}.csv"), index=False)
write_json({"args": vars(args), "device": dev, "torch": torch.__version__, "torchvision": torchvision.__version__,
            "best_val_pr_auc": state["best"], "log": state["log"],
            "gpu": torch.cuda.get_device_name(0) if dev == "cuda" else None},
           os.path.join(args.out, "preds", f"img_effb0_seed{args.seed}_cfg.json"))
print("done. best val PR-AUC:", state["best"])
