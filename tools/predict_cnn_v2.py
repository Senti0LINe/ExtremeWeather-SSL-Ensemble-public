"""Predict with frozen CINet_v2 on dev / benchmark / test_eval, with 4-flip TTA."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))
from ewci import data  # noqa: E402
from ewci.model_v2 import CINet_v2  # noqa: E402
from ewci.tensors import build_input  # noqa: E402

FULL = Path("/data1/group3/DATASAT/experiments/ci_det_v2/annotations_full")
_READER = None


def _worker_init(role: str):
    global _READER
    _READER = data.ObsReader(role)


class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, patches: List[dict], role: str, size: int = 224):
        self.patches = patches
        self.role = role
        self.size = size

    def __len__(self) -> int:
        return len(self.patches)

    def __getitem__(self, idx: int) -> torch.Tensor:
        global _READER
        if _READER is None:
            _READER = data.ObsReader(self.role)
        raw = _READER.read_patch(self.patches[idx])
        return torch.from_numpy(build_input(raw, self.size))


def select(split: str) -> Tuple[List[dict], str]:
    if split == "dev":
        uids = set(
            json.loads((PROJECT / "splits_v1.json").read_text())["dev"][
                "early_stop_uids"
            ]
        )
        return [
            p
            for p in data.load_patches_from(FULL, "val")
            if p["sample_uid"] in uids
        ], "val"
    if split == "val" or split == "benchmark":
        return data.benchmark_patches(), "val"
    if split == "test_eval" or split == "test":
        uids = set(
            json.loads((PROJECT / "splits_test_eval.json").read_text())["uids"]
        )
        return [
            p
            for p in data.load_patches_from(FULL, "test")
            if p["sample_uid"] in uids
        ], "test"
    raise SystemExit(f"Unknown split: {split}")


def main():
    ap = argparse.ArgumentParser(description="CINet_v2 TTA Inference Generator")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument(
        "--split",
        choices=["dev", "val", "benchmark", "test_eval", "test"],
        required=True,
    )
    ap.add_argument("--out", required=True)
    ap.add_argument("--tta", action="store_true", default=True)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--heatmap-size", type=int, default=28)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    patches, role = select(args.split)
    patches.sort(
        key=lambda p: (int(p["obs_index"]), int(p["row"]), int(p["col"]))
    )
    print(
        f"[INFER] Windows: {len(patches)} | Role: {role} | TTA: {args.tta} | Size: {args.size}",
        flush=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CINet_v2(
        in_ch=41, width=32, heatmap_size=args.heatmap_size
    ).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()

    ds = SimpleDataset(patches, role, size=args.size)
    dl = DataLoader(
        ds,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        worker_init_fn=lambda _: _worker_init(role),
    )

    out_p = []
    out_hm_max = []
    with torch.no_grad():
        for x in dl:
            x = x.to(device, non_blocking=True)
            # Base forward
            cls_logits, hm_logits = model(x, return_heatmap=True)
            acc_p = torch.sigmoid(cls_logits)
            acc_hm = torch.sigmoid(hm_logits)

            if args.tta:
                # 4-flip TTA: horizontal [3], vertical [2], both [2, 3]
                for dims in ([3], [2], [2, 3]):
                    x_flip = torch.flip(x, dims=dims)
                    c_l, h_l = model(x_flip, return_heatmap=True)
                    acc_p = acc_p + torch.sigmoid(c_l)
                    # Unflip heatmap back to canonical orientation
                    h_prob = torch.sigmoid(h_l)
                    h_unflip = torch.flip(h_prob, dims=dims)
                    acc_hm = acc_hm + h_unflip

                acc_p = acc_p / 4.0
                acc_hm = acc_hm / 4.0

            out_p.append(acc_p.cpu())
            out_hm_max.append(acc_hm.amax(dim=(1, 2, 3)).cpu())

    p = torch.cat(out_p).numpy()
    hm_max = torch.cat(out_hm_max).numpy()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        p=p,
        hm_max=hm_max,
        uid=np.array([x["sample_uid"] for x in patches]),
        row=np.array([int(x["row"]) for x in patches]),
        col=np.array([int(x["col"]) for x in patches]),
    )
    print(
        f"[INFER] Saved {out_path} (p: {p.shape}, hm_max: {hm_max.shape})",
        flush=True,
    )


if __name__ == "__main__":
    main()
