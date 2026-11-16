"""Train the compact spatio-temporal CNN on the frozen split (single GPU)."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))
from ewci import data, metrics as M  # noqa: E402
from ewci.tensors import build_input  # noqa: E402

_READER = None
_SIZE = 128


def _worker_init(role: str):
    global _READER
    _READER = data.ObsReader(role)


class CIPatchDataset(Dataset):
    def __init__(self, patches, role, size=_SIZE):
        self.patches = patches
        self.role = role
        self.size = size
        self.labels = np.array([data.has_ci(p) for p in patches], dtype=np.float32)

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, i):
        raw = _READER.read_patch(self.patches[i])
        return torch.from_numpy(build_input(raw, self.size)), self.labels[i]


class GeM(nn.Module):
    def __init__(self, p=3.0):
        super().__init__()
        self.p = nn.Parameter(torch.tensor(float(p)))

    def forward(self, x):
        return x.clamp(min=1e-6).pow(self.p).mean(dim=(2, 3)).pow(1.0 / self.p)


class Block(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.n1 = nn.GroupNorm(min(8, cout), cout)
        self.c2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.n2 = nn.GroupNorm(min(8, cout), cout)
        self.skip = None
        if stride != 1 or cin != cout:
            self.skip = nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False),
                                      nn.GroupNorm(min(8, cout), cout))

    def forward(self, x):
        h = F.gelu(self.n1(self.c1(x)))
        h = self.n2(self.c2(h))
        s = x if self.skip is None else self.skip(x)
        return F.gelu(h + s)


class CINet(nn.Module):
    def __init__(self, in_ch=41, width=32):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(in_ch, width, 3, 2, 1, bias=False),
                                  nn.GroupNorm(8, width), nn.GELU())
        self.b1 = Block(width, width * 2, 2)
        self.b2 = Block(width * 2, width * 4, 2)
        self.b3 = Block(width * 4, width * 8, 2)
        self.pool = GeM()
        self.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(width * 8, 1))

    def forward(self, x):
        x = self.stem(x)
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        return self.head(self.pool(x)).squeeze(-1)


def augment(x):
    if torch.rand(1).item() < 0.5:
        x = torch.flip(x, dims=[3])
    if torch.rand(1).item() < 0.5:
        x = torch.flip(x, dims=[2])
    x = x * (0.98 + 0.04 * torch.rand(x.shape[0], 1, 1, 1, device=x.device))
    if torch.rand(1).item() < 0.5:
        h, w = x.shape[-2:]
        bh, bw = h // 6, w // 6
        for b in range(x.shape[0]):
            top = torch.randint(0, h - bh, (1,)).item()
            left = torch.randint(0, w - bw, (1,)).item()
            x[b, :, top : top + bh, left : left + bw] = 0.0
    return x


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    out = []
    for x, _ in loader:
        out.append(torch.sigmoid(model(x.to(device, non_blocking=True))).cpu())
    return torch.cat(out).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--val-ts", type=int, default=20)
    ap.add_argument("--minutes", type=float, default=55)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--init", default="", help="optional pretrained checkpoint")
    ap.add_argument("--out", default="experiments/p1_cnn")
    args = ap.parse_args()

    global _SIZE
    _SIZE = args.size
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device", device, torch.cuda.get_device_name(0) if device.type == "cuda" else "")

    tr_patches = data.load_patches("train")
    dev_all = [p for p in data.load_patches_from(Path("/data1/group3/DATASAT/experiments/ci_det_v2/annotations_full"), "val")
               if p["sample_uid"] in set(json.loads((PROJECT / "splits_v1.json").read_text())["dev"]["early_stop_uids"])]
    dev_uids = sorted({p["sample_uid"] for p in dev_all})
    small = set(dev_uids[: args.val_ts])
    dev_small = [p for p in dev_all if p["sample_uid"] in small]
    bench = data.benchmark_patches()
    print("train %d | dev %d | bench %d" % (len(tr_patches), len(dev_all), len(bench)))

    ds = CIPatchDataset(tr_patches, "train")
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=args.workers,
                    pin_memory=True, drop_last=True, persistent_workers=args.workers > 0,
                    worker_init_fn=lambda _: _worker_init("train"))
    dl_dev = DataLoader(CIPatchDataset(dev_small, "val"), batch_size=64, shuffle=False,
                        num_workers=4, worker_init_fn=lambda _: _worker_init("val"))
    dl_dev_full = DataLoader(CIPatchDataset(dev_all, "val"), batch_size=64, shuffle=False,
                             num_workers=6, worker_init_fn=lambda _: _worker_init("val"))
    dl_bench = DataLoader(CIPatchDataset(bench, "val"), batch_size=64, shuffle=False,
                          num_workers=6, worker_init_fn=lambda _: _worker_init("val"))

    model = CINet(in_ch=41).to(device)
    if args.init:
        sd = torch.load(args.init, map_location=device)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print("init from %s (missing %d, unexpected %d)" % (args.init, len(missing), len(unexpected)), flush=True)
    n_pos = float(np.sum(ds.labels)); n_neg = float(len(ds.labels) - n_pos)
    pos_weight = torch.tensor([n_neg / n_pos], device=device)
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-5)

    t_start = time.time()
    best = -1.0
    hist = []
    for ep in range(1, args.epochs + 1):
        model.train()
        tot, n = 0.0, 0
        for x, y in dl:
            if time.time() - t_start > args.minutes * 60:
                break
            x = augment(x.to(device, non_blocking=True))
            y = y.to(device)
            opt.zero_grad(set_to_none=True)
            loss = crit(model(x), y)
            loss.backward()
            opt.step()
            tot += loss.item() * len(y); n += len(y)
        sched.step()
        p = predict(model, dl_dev, device)
        y_dev = np.array([data.has_ci(pt) for pt in dev_small])
        auc = M.ranking(y_dev, p)["ROC_AUC"]
        hist.append({"epoch": ep, "loss": tot / max(1, n), "dev_auc_small": auc,
                     "minutes": (time.time() - t_start) / 60})
        print("epoch %d loss %.4f dev-%dts AUC %.4f %.1f min" % (ep, tot / max(1, n), len(small), auc, (time.time() - t_start) / 60), flush=True)
        if auc > best:
            best = auc
            torch.save(model.state_dict(), out / "best.pt")
        if time.time() - t_start > args.minutes * 60:
            print("time budget reached", flush=True)
            break

    model.load_state_dict(torch.load(out / "best.pt", map_location=device))
    p_dev = predict(model, dl_dev_full, device)
    p_bench = predict(model, dl_bench, device)
    y_dev = np.array([data.has_ci(pt) for pt in dev_all])
    y_bench = np.array([data.has_ci(pt) for pt in bench])
    res = {"dev": {"n": len(y_dev), "pos": int(y_dev.sum()), **M.ranking(y_dev, p_dev)},
           "bench": {"n": len(y_bench), "pos": int(y_bench.sum()), **M.ranking(y_bench, p_bench)},
           "history": hist}
    np.savez_compressed(out / "probs.npz", p_dev=p_dev, y_dev=y_dev, p_bench=p_bench, y_bench=y_bench)
    json.dump(res, open(out / "metrics.json", "w"), indent=2)
    print(json.dumps({k: v for k, v in res.items() if k != "history"}, indent=2))


if __name__ == "__main__":
    main()