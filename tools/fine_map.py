"""Fine-grid (stride 56) CI probability map for one timestamp.

Produces a 33 x 35 risk grid (1,155 windows) instead of the 17 x 18 grid used
for evaluation, for display only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import zarr
from torch.utils.data import DataLoader

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "tools"))
from ewci import data  # noqa: E402
from ewci.model import CINet  # noqa: E402
from train_cnn import CIPatchDataset, _worker_init  # noqa: E402

FULL = Path("/data1/group3/DATASAT/experiments/ci_det_v2/annotations_full")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="experiments/p1_cnn/best.pt")
    ap.add_argument("--role", default="test")
    ap.add_argument("--uid", default="")
    ap.add_argument("--stride", type=int, default=56)
    ap.add_argument("--out", default="experiments/final/fine_map.png")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    samples = data.load_samples_from = None  # placeholder to keep linters quiet
    from ewci.data import load_patches_from
    smp = {}
    with open(FULL / (args.role + "_samples.jsonl")) as fh:
        for line in fh:
            if line.strip():
                s = json.loads(line)
                smp[s["sample_uid"]] = s
    patches = load_patches_from(FULL, args.role)
    uid = args.uid
    if not uid:
        from collections import Counter
        c = Counter()
        for p in patches:
            c[p["sample_uid"]] += len(p.get("objects", []))
        uid = c.most_common(1)[0][0]
    print("timestamp", uid, "objects", sum(len(p.get("objects", [])) for p in patches if p["sample_uid"] == uid))

    H, W, S = 2001, 2101, 224
    rows = sorted(set(list(range(0, H - S + 1, args.stride)) + [H - S]))
    cols = sorted(set(list(range(0, W - S + 1, args.stride)) + [W - S]))
    oi = int(smp[uid]["obs_index"])
    grid_patches = [{"sample_uid": uid, "obs_index": oi, "row": r, "col": c, "patch_size": S, "objects": []}
                    for r in rows for c in cols]
    print("grid", len(rows), "x", len(cols), "=", len(grid_patches))

    device = torch.device(args.device)
    model = CINet(in_ch=41).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()
    ds = CIPatchDataset(grid_patches, args.role)
    dl = DataLoader(ds, batch_size=64, shuffle=False, num_workers=10, pin_memory=True,
                    worker_init_fn=lambda _: _worker_init(args.role))
    out = []
    with torch.no_grad():
        for x, _ in dl:
            out.append(torch.sigmoid(model(x.to(device, non_blocking=True))).cpu())
    p = torch.cat(out).numpy().reshape(len(rows), len(cols))

    z = zarr.open(data.STORE_PATH, mode="r")
    bt = np.asarray(z[args.role]["obs"][oi - 1, 6, :, :]).astype(np.float32) * 0.01
    bt[bt < -100] = np.nan

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    fig, ax = plt.subplots(1, 2, figsize=(15, 7))
    ax[0].imshow(bt, cmap="turbo", vmin=200, vmax=300)
    ax[1].imshow(p, cmap="magma", vmin=0, vmax=1, extent=(0, W, H, 0), interpolation="bilinear")
    for q in [q for q in patches if q["sample_uid"] == uid]:
        for o in q.get("objects", []):
            x1, y1, x2, y2 = o["full_grid_bbox_xyxy"]
            ax[0].add_patch(Rectangle((x1, y1), max(3, x2 - x1), max(3, y2 - y1), fill=False, edgecolor="lime", lw=1.0))
            ax[1].add_patch(Rectangle((x1, y1), max(3, x2 - x1), max(3, y2 - y1), fill=False, edgecolor="lime", lw=0.6))
    ax[0].set_title("Himawari B13 T-10  |  green = CI")
    ax[1].set_title("CNN CI probability, stride %d (%d x %d)" % (args.stride, len(rows), len(cols)))
    for a in ax:
        a.set_xlim(0, W); a.set_ylim(H, 0)
    plt.colorbar(ax[1].images[0], ax=ax[1], fraction=0.03)
    plt.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out, dpi=150)
    np.savez_compressed(str(Path(args.out).with_suffix(".npz")), p=p, rows=rows, cols=cols, uid=uid)
    print("saved", args.out)


if __name__ == "__main__":
    main()