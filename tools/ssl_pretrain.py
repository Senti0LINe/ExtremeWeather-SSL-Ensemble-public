"""Self-supervised contrastive pretraining on unlabelled CI windows.

The unlabelled pool is the 3-4 月 grid: any 224 window of the training period
can be used, we simply never look at its label.  Only the encoder is trained;
the supervised head is fine-tuned afterwards.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))
from ewci import data  # noqa: E402
from ewci.model import CINet  # noqa: E402
from ewci.ssl import augment_view, nt_xent  # noqa: E402
from ewci.tensors import build_input  # noqa: E402

_READER = None
_SIZE = 128


def _init():
    global _READER
    _READER = data.ObsReader("train")


class UnlabelledWindows(Dataset):
    """Deterministic sample of 224 windows from the training period."""

    def __init__(self, n: int, seed: int = 0, size: int = 128):
        samples = data.load_samples("train")
        self.uids = list(samples)
        self.obs_index = {u: int(samples[u]["obs_index"]) for u in self.uids}
        rng = np.random.RandomState(seed)
        H, W, S = 2001, 2101, 224
        self.items = [(int(rng.randint(0, len(self.uids))),
                       int(rng.randint(0, H - S)),
                       int(rng.randint(0, W - S))) for _ in range(n)]
        self.size = size

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        k, r, c = self.items[i]
        oi = self.obs_index[self.uids[k]]
        raw = _READER.obs[oi - 4 : oi, data.IR_SLICE, r : r + 224, c : c + 224]
        return torch.from_numpy(build_input(np.asarray(raw), self.size))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30000)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch", type=int, default=96)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--minutes", type=float, default=150)
    ap.add_argument("--out", default="experiments/p2_ssl")
    args = ap.parse_args()

    torch.set_num_threads(8)
    device = torch.device("cuda")
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    ds = UnlabelledWindows(args.n, seed=1)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=args.workers,
                    pin_memory=True, drop_last=True, worker_init_fn=lambda _: _init(),
                    persistent_workers=True)
    model = CINet(in_ch=41, emb_dim=128).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-6)
    t0 = time.time()
    print("unlabelled windows %d | batch %d | steps/epoch %d" % (len(ds), args.batch, len(dl)), flush=True)
    for ep in range(1, args.epochs + 1):
        model.train()
        tot, n = 0.0, 0
        for x in dl:
            if time.time() - t0 > args.minutes * 60:
                break
            x = x.to(device, non_blocking=True)
            v1, v2 = augment_view(x), augment_view(x)
            z1, z2 = model.embed(v1), model.embed(v2)
            loss = nt_xent(z1, z2)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            tot += loss.item() * x.shape[0]; n += x.shape[0]
        sched.step()
        print("epoch %d loss %.4f %.1f min" % (ep, tot / max(1, n), (time.time() - t0) / 60), flush=True)
        torch.save(model.state_dict(), out / "encoder.pt")
        if time.time() - t0 > args.minutes * 60:
            print("time budget reached", flush=True)
            break
    print("saved", out / "encoder.pt", flush=True)


if __name__ == "__main__":
    main()
