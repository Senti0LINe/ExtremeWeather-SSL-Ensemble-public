"""Predict with a frozen CNN on dev / benchmark / test_eval, with optional TTA."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "tools"))
from ewci import data  # noqa: E402
from ewci.model import CINet  # noqa: E402
from train_cnn import CIPatchDataset, _worker_init  # noqa: E402

FULL = Path("/data1/group3/DATASAT/experiments/ci_det_v2/annotations_full")


def select(split):
    if split == "dev":
        uids = set(json.loads((PROJECT / "splits_v1.json").read_text())["dev"]["early_stop_uids"])
        return [p for p in data.load_patches_from(FULL, "val") if p["sample_uid"] in uids], "val"
    if split == "val":
        return data.benchmark_patches(), "val"
    if split == "test_eval":
        uids = set(json.loads((PROJECT / "splits_test_eval.json").read_text())["uids"])
        return [p for p in data.load_patches_from(FULL, "test") if p["sample_uid"] in uids], "test"
    raise SystemExit("unknown split")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", choices=["dev", "val", "test_eval"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--workers", type=int, default=10)
    args = ap.parse_args()

    patches, role = select(args.split)
    patches.sort(key=lambda p: (int(p["obs_index"]), int(p["row"]), int(p["col"])))
    print("windows", len(patches), "role", role, "tta", args.tta, flush=True)

    model = CINet(in_ch=41).cuda()
    model.load_state_dict(torch.load(args.ckpt, map_location="cuda"))
    model.eval()
    ds = CIPatchDataset(patches, role, args.size)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=False, num_workers=args.workers,
                    pin_memory=True, worker_init_fn=lambda _: _worker_init(role))
    out = []
    with torch.no_grad():
        for x, _ in dl:
            x = x.cuda(non_blocking=True)
            acc = torch.sigmoid(model(x))
            if args.tta:
                for dims in ([3], [2], [2, 3]):
                    acc = acc + torch.sigmoid(model(torch.flip(x, dims=dims)))
                acc = acc / 4.0
            out.append(acc.cpu())
    p = torch.cat(out).numpy()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, p=p,
                        uid=np.array([x["sample_uid"] for x in patches]),
                        row=np.array([int(x["row"]) for x in patches]),
                        col=np.array([int(x["col"]) for x in patches]))
    print("saved", args.out, p.shape, flush=True)


if __name__ == "__main__":
    main()
