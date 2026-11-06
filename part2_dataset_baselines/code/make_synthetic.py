"""TESTING ONLY: builds a small fake HAM10000-like dataset to smoke-test the pipeline. Never used for reported results."""
import os, sys, numpy as np, pandas as pd
from PIL import Image
out = sys.argv[1]; rng = np.random.RandomState(0)
for d in ("img1", "img2"): os.makedirs(os.path.join(out, d), exist_ok=True)
dxs = ["nv"] * 60 + ["mel"] * 12 + ["bkl"] * 12 + ["bcc"] * 8 + ["akiec"] * 6 + ["vasc"] * 4 + ["df"] * 4
rows = []; n = 0
for li, dx in enumerate(dxs):
    k = rng.choice([1, 1, 2, 3])
    age = rng.choice([30, 45, 55, 70, np.nan], p=[.2, .25, .25, .25, .05]); sex = rng.choice(["male", "female", "unknown"], p=[.5, .45, .05])
    loc = rng.choice(["back", "trunk", "lower extremity", "face", "unknown"])
    for j in range(k):
        iid = f"ISIC_{n:07d}"; n += 1
        base = np.array([150, 100, 80]) if dx != "mel" else np.array([90, 60, 70])
        im = np.clip(base + rng.randn(48, 64, 3) * 20, 0, 255).astype(np.uint8)
        Image.fromarray(im).save(os.path.join(out, "img1" if n % 2 else "img2", iid + ".jpg"))
        rows.append(dict(lesion_id=f"HAM_{li:07d}", image_id=iid, dx=dx, dx_type=rng.choice(["histo", "follow_up"]), age=age, sex=sex, localization=loc))
pd.DataFrame(rows).to_csv(os.path.join(out, "HAM10000_metadata.csv"), index=False)
print("synthetic images:", len(rows))
