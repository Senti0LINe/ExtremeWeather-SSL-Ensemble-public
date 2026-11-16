"""Forecaster panel: 4 frames in -> probability map + hotspots + uncertainty.

Example:
    python tools/make_panel.py --uid val/140/SCW2021-0065 --role val --stride 56 --device cpu
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))
from ewci import data, features  # noqa: E402
from ewci.model import CINet  # noqa: E402
from ewci.tensors import build_input  # noqa: E402

FULL = Path("/data1/group3/DATASAT/experiments/ci_det_v2/annotations_full")
BG = Path("/data1/group3/DATASAT/experiments/ci_det_v2/annotations_stage1_bg")
D = PROJECT / "experiments"
LAT0, LON0 = 40.00448, 103.99570
H, W, S = 2001, 2101, 224
_READER = None


def _init(role):
    global _READER
    _READER = data.ObsReader(role)


class GridDataset(Dataset):
    def __init__(self, patches, size=128):
        self.patches = patches
        self.size = size

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, i):
        raw = _READER.read_patch(self.patches[i])
        return (torch.from_numpy(build_input(raw, self.size)),
                torch.from_numpy(features.extract(raw)), i)


def load_samples(role):
    root = FULL if role == "test" else BG
    out = {}
    with open(root / (role + "_samples.jsonl")) as fh:
        for line in fh:
            if line.strip():
                s = json.loads(line)
                out[s["sample_uid"]] = s
    return out


def load_patches(role):
    return data.load_patches_from(FULL if role == "test" else BG, role)


def load_models(ckpts, device):
    out = []
    for p in ckpts:
        if Path(p).exists():
            m = CINet(in_ch=41).to(device)
            m.load_state_dict(torch.load(p, map_location=device))
            m.eval()
            out.append(m)
    return out


@torch.no_grad()
def predict_cnn(models, tensors, device, tta=True):
    per = []
    x = tensors.to(device)
    for m in models:
        acc = torch.sigmoid(m(x))
        if tta:
            for dims in ([3], [2], [2, 3]):
                acc = acc + torch.sigmoid(m(torch.flip(x, dims=dims)))
            acc = acc / 4.0
        per.append(acc.cpu().numpy())
    a = np.stack(per)
    return a.mean(0), (a.std(0) if len(per) > 1 else np.zeros(len(tensors)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--uid", default="")
    ap.add_argument("--role", default="test")
    ap.add_argument("--stride", type=int, default=56)
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="reports/panel")
    args = ap.parse_args()
    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")

    smp = load_samples(args.role)
    patches_all = load_patches(args.role)
    uid = args.uid
    if not uid:
        from collections import Counter
        c = Counter()
        for p in patches_all:
            c[p["sample_uid"]] += len(p.get("objects", []))
        uid = c.most_common(1)[0][0]
    oi = int(smp[uid]["obs_index"])

    rows = sorted(set(list(range(0, H - S + 1, args.stride)) + [H - S]))
    cols = sorted(set(list(range(0, W - S + 1, args.stride)) + [W - S]))
    grid = [{"sample_uid": uid, "obs_index": oi, "row": r, "col": c, "patch_size": S, "objects": []}
            for r in rows for c in cols]
    print("timestamp", uid, "grid", len(rows), "x", len(cols), "=", len(grid), flush=True)

    ckpts = [D / "p1_cnn/best.pt"] + [D / ("p1_seed%d/best.pt" % s) for s in range(1, 5)]
    models = load_models(ckpts, device)
    if not models:
        raise SystemExit("no checkpoint found")
    print("models", len(models), flush=True)

    dl = DataLoader(GridDataset(grid, args.size), batch_size=48, shuffle=False,
                    num_workers=8, worker_init_fn=lambda _: _init(args.role))
    probs, stds, feats, idxs = [], [], [], []
    for tens, feat, idx in dl:
        p, s = predict_cnn(models, tens, device)
        probs.append(p); stds.append(s); feats.append(feat.numpy()); idxs.append(idx.numpy())
    p_cnn = np.concatenate(probs); e_cnn = np.concatenate(stds)
    X = np.concatenate(feats); order = np.argsort(np.concatenate(idxs))
    p_cnn, e_cnn, X = p_cnn[order], e_cnn[order], X[order]
    print("cnn done", p_cnn.shape, flush=True)

    import joblib
    import lightgbm as lgb
    cache = D / "final/gbdt_cache.pkl"
    if cache.exists():
        gb = joblib.load(cache)
    else:
        tr = np.load(D / "p0_gbdt/features_train.npz", allow_pickle=True)
        prm = dict(n_estimators=700, learning_rate=0.03, num_leaves=63, min_child_samples=40,
                   subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                   class_weight="balanced", n_jobs=8, verbose=-1)
        gb = [lgb.LGBMClassifier(**prm, random_state=s).fit(tr["X"], tr["y"]) for s in (0, 1, 2)]
        joblib.dump(gb, cache)
    p_gbdt = np.mean([m.predict_proba(X)[:, 1] for m in gb], 0)

    calib = D / "final/calib.json"
    if calib.exists():
        k = json.loads(calib.read_text())
        w = k.get("w", 0.75)
        p_raw = w * p_cnn + (1 - w) * p_gbdt
        p_cal = np.interp(p_raw, k["x"], k["y"])
        tau = k["tau"]
    else:
        w = 0.75
        p_raw = w * p_cnn + (1 - w) * p_gbdt
        p_cal, tau = p_raw, 0.5
    tau = args.threshold if args.threshold is not None else tau

    import zarr
    z = zarr.open(data.STORE_PATH, mode="r")
    bt = np.asarray(z[args.role]["obs"][oi - 1, 6, :, :]).astype(np.float32) * 0.01
    bt[bt < -100] = np.nan

    hot = []
    for k in np.argsort(p_cal)[::-1][:8]:
        if p_cal[k] < tau:
            continue
        r, c = int(grid[k]["row"]), int(grid[k]["col"])
        hot.append({"lat": round(LAT0 - 0.01 * (r + S / 2), 2),
                    "lon": round(LON0 + 0.01 * (c + S / 2), 2),
                    "p": float(p_cal[k]), "epistemic": float(e_cnn[k])})

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    fig, ax = plt.subplots(1, 2, figsize=(15, 7))
    ax[0].imshow(bt, cmap="turbo", vmin=200, vmax=300)
    ax[0].set_title("Himawari B13 (T-10 min)  " + uid)
    im = ax[1].imshow(p_cal.reshape(len(rows), len(cols)), cmap="magma", vmin=0, vmax=1,
                      extent=(0, W, H, 0), interpolation="bilinear")
    ax[1].set_title("CI probability (stride %d, %d x %d)" % (args.stride, len(rows), len(cols)))
    for h in hot:
        r = int((LAT0 - h["lat"]) / 0.01 - S / 2); c = int((h["lon"] - LON0) / 0.01 - S / 2)
        for a in ax:
            a.add_patch(Rectangle((c, r), S, S, fill=False, edgecolor="lime", lw=1.2))
    for a in ax:
        a.set_xlim(0, W); a.set_ylim(H, 0)
    plt.colorbar(im, ax=ax[1], fraction=0.03)
    plt.tight_layout()
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    png = out.with_suffix(".png"); plt.savefig(png, dpi=150); plt.close()

    summary = {"uid": uid, "role": args.role, "grid": [len(rows), len(cols)],
               "max_prob": float(p_cal.max()), "mean_prob": float(p_cal.mean()),
               "cells_above_tau": int((p_cal >= tau).sum()), "tau": float(tau),
               "models": len(models), "mean_epistemic": float(e_cnn.mean()), "hotspots": hot}
    out.with_suffix(".json").write_text(json.dumps(summary, indent=2))

    b64 = base64.b64encode(png.read_bytes()).decode()
    rows_html = "".join("<tr><td>%.2f</td><td>%.2f</td><td>%.3f</td><td>%.3f</td></tr>"
                        % (h["lat"], h["lon"], h["p"], h["epistemic"]) for h in hot)
    html = """<html><head><meta charset="utf-8"><title>CI panel</title>
<style>body{font-family:sans-serif;margin:20px}table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:4px 8px}</style>
</head><body><h2>CI assistant panel — %s</h2>
<p><b>Max probability</b> %.2f · <b>cells above tau (%.2f)</b> %d · <b>mean epistemic</b> %.3f · <b>models</b> %d</p>
<img src="data:image/png;base64,%s" style="width:100%%;max-width:1500px">
<h3>Hotspots</h3><table><tr><th>lat</th><th>lon</th><th>probability</th><th>epistemic</th></tr>%s</table>
<p style="color:#666">Research prototype — regional risk (~125 km cells), not a per-cell nowcast, not for public warning.</p>
</body></html>""" % (uid, summary["max_prob"], tau, summary["cells_above_tau"], summary["mean_epistemic"], len(models), b64, rows_html)
    out.with_suffix(".html").write_text(html)
    print(json.dumps(summary, indent=1))
    print("saved", png, out.with_suffix(".html"))


if __name__ == "__main__":
    main()
