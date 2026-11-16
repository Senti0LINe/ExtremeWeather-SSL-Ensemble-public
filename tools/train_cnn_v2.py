"""Train CINet_v2 with Local Extremum Attention (LEA), Sub-window Heatmap Head,
and Hard Negative Background Window Balanced Sampling.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))
from ewci import data, metrics as M  # noqa: E402
from ewci.model_v2 import CINet_v2  # noqa: E402
from ewci.tensors import build_input  # noqa: E402

_READER = None


def _worker_init(role: str):
    global _READER
    _READER = data.ObsReader(role)


class CIPatchDataset_v2(Dataset):
    """Dataset producing 41-channel physical tensors, binary CI labels,

    and continuous 2D Gaussian heatmaps of convective cell centroids.
    """

    def __init__(
        self,
        patches: List[dict],
        role: str,
        size: int = 224,
        heatmap_size: int = 28,
        sigma: float = 2.0,
        is_train: bool = False,
    ):
        self.patches = patches
        self.role = role
        self.size = size
        self.heatmap_size = heatmap_size
        self.sigma = sigma
        self.is_train = is_train
        self.labels = np.array(
            [data.has_ci(p) for p in patches], dtype=np.float32
        )

        # Precompute coordinate grid for Gaussian heatmap generation
        y_coords = torch.arange(self.heatmap_size, dtype=torch.float32)
        x_coords = torch.arange(self.heatmap_size, dtype=torch.float32)
        grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing="ij")
        self.grid_y = grid_y.unsqueeze(0)  # [1, H_hm, W_hm]
        self.grid_x = grid_x.unsqueeze(0)  # [1, H_hm, W_hm]

    def __len__(self) -> int:
        return len(self.patches)

    def _build_heatmap(self, patch: dict) -> torch.Tensor:
        """Gaussian keypoint targets anchored at each object's TRUE centre.

        Two corrections over the original version (see EXPERIMENTS E14 P1/P2):
        * the centre is recovered from full_grid_bbox_xyxy minus the window
          origin, so edge-truncated objects no longer pull the target inward
          (measured bias on the frozen test set: mean 3.3 km, max 62.2 km);
        * an anchor is placed only when that true centre lies inside this
          window, and the nearest grid point is forced to exactly 1.0 so the
          focal loss positive mask fires (previously 60.1% of positive windows
          had an empty mask and num_pos collapsed to 1).
        """
        H_hm = self.heatmap_size
        W_hm = self.heatmap_size
        patch_size = float(patch.get("patch_size", 224))
        row = float(patch.get("row", 0))
        col = float(patch.get("col", 0))
        scale_y = H_hm / patch_size
        scale_x = W_hm / patch_size
        hm = torch.zeros((1, H_hm, W_hm), dtype=torch.float32)

        for o in patch.get("objects", []):
            if o.get("label", 0) != 1:
                continue
            if not o.get("state_valid", True):
                continue

            full = o.get("full_grid_bbox_xyxy")
            bbox = o.get("bbox_xyxy")
            if full is not None and len(full) == 4:
                cx = ((full[0] + full[2]) / 2.0 - col) * scale_x
                cy = ((full[1] + full[3]) / 2.0 - row) * scale_y
            elif bbox is not None and len(bbox) == 4:
                cx = (bbox[0] + bbox[2]) / 2.0 * scale_x
                cy = (bbox[1] + bbox[3]) / 2.0 * scale_y
            else:
                continue

            if not (0.0 <= cx < W_hm and 0.0 <= cy < H_hm):
                continue

            d2 = (self.grid_x - cx) ** 2 + (self.grid_y - cy) ** 2
            g = torch.exp(-d2 / (2.0 * self.sigma * self.sigma))
            # round() can land on H_hm/W_hm for centres in the last half pixel
            iy = min(H_hm - 1, max(0, int(round(cy))))
            ix = min(W_hm - 1, max(0, int(round(cx))))
            g[0, iy, ix] = 1.0
            hm = torch.maximum(hm, g)
        return hm

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        global _READER
        if _READER is None:
            _READER = data.ObsReader(self.role)

        patch = self.patches[idx]
        raw = _READER.read_patch(patch)
        x_np = build_input(raw, self.size)
        x = torch.from_numpy(x_np)  # [41, size, size]
        y_cls = torch.tensor(self.labels[idx], dtype=torch.float32)
        y_hm = self._build_heatmap(patch)  # [1, heatmap_size, heatmap_size]

        if self.is_train:
            # 1. Random horizontal flip
            if torch.rand(1).item() < 0.5:
                x = torch.flip(x, dims=[2])
                y_hm = torch.flip(y_hm, dims=[2])

            # 2. Random vertical flip
            if torch.rand(1).item() < 0.5:
                x = torch.flip(x, dims=[1])
                y_hm = torch.flip(y_hm, dims=[1])

            # 3. Brightness temperature jitter (+/- 2%)
            scale = 0.98 + 0.04 * torch.rand(1).item()
            x = x * scale

            # 4. Cutout regularization
            if torch.rand(1).item() < 0.5:
                h, w = x.shape[1], x.shape[2]
                bh, bw = h // 6, w // 6
                top = torch.randint(0, h - bh, (1,)).item()
                left = torch.randint(0, w - bw, (1,)).item()
                x[:, top : top + bh, left : left + bw] = 0.0

        return x, y_cls, y_hm


def gaussian_focal_loss(
    pred_logits: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 2.0,
    beta: float = 4.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Penalty-reduced Gaussian focal loss for continuous keypoint heatmaps.

    Args:
        pred_logits: unnormalized heatmap logits [B, 1, H, W]
        target: ground truth Gaussian heatmaps [B, 1, H, W] in [0, 1]
    """
    pred = torch.sigmoid(pred_logits)
    pred = torch.clamp(pred, min=eps, max=1.0 - eps)

    pos_mask = target.ge(0.99)
    neg_mask = target.lt(0.99)

    pos_loss = -torch.pow(1.0 - pred, alpha) * torch.log(pred) * pos_mask
    neg_weights = torch.pow(1.0 - target, beta)
    neg_loss = (
        -neg_weights
        * torch.pow(pred, alpha)
        * torch.log(1.0 - pred)
        * neg_mask
    )

    num_pos = max(1.0, float(pos_mask.float().sum().item()))
    return (pos_loss.sum() + neg_loss.sum()) / num_pos


@torch.no_grad()
def evaluate_probs(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    p_cls_list = []
    hm_max_list = []
    for x, _, _ in loader:
        x = x.to(device, non_blocking=True)
        cls_logits, hm_logits = model(x, return_heatmap=True)
        p_cls = torch.sigmoid(cls_logits).cpu()
        hm_probs = torch.sigmoid(hm_logits).cpu()
        p_cls_list.append(p_cls)
        hm_max_list.append(hm_probs.amax(dim=(1, 2, 3)))
    return torch.cat(p_cls_list).numpy(), torch.cat(hm_max_list).numpy()


def main():
    ap = argparse.ArgumentParser(
        description="CINet_v2 Multi-task Training with Balanced Negative Sampling"
    )
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--heatmap-size", type=int, default=28)
    ap.add_argument("--sigma", type=float, default=2.0)
    ap.add_argument(
        "--lambda-hm",
        type=float,
        default=1.0,
        help="Weight for heatmap focal loss",
    )
    ap.add_argument(
        "--balanced-sampling",
        action="store_true",
        default=True,
        help="Enable 1:1 balanced sampling between CI and cold background windows",
    )
    ap.add_argument(
        "--pos-ratio",
        type=float,
        default=0.5,
        help="Target positive ratio in balanced sampling",
    )
    ap.add_argument("--val-ts", type=int, default=20)
    ap.add_argument("--minutes", type=float, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--init", default="", help="Optional pretrained checkpoint")
    ap.add_argument("--out", default="experiments/p2_seed0")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"[INIT] Device: {device} | {torch.cuda.get_device_name(0) if device.type == 'cuda' else ''}"
    )
    print(
        f"[INIT] Seed: {args.seed} | Size: {args.size} | Heatmap: {args.heatmap_size} | Lambda HM: {args.lambda_hm}"
    )

    # 1. Load splits
    tr_patches = data.load_patches("train")
    dev_all = [
        p
        for p in data.load_patches_from(
            Path(
                "/data1/group3/DATASAT/experiments/ci_det_v2/annotations_full"
            ),
            "val",
        )
        if p["sample_uid"]
        in set(
            json.loads((PROJECT / "splits_v1.json").read_text())["dev"][
                "early_stop_uids"
            ]
        )
    ]
    dev_uids = sorted({p["sample_uid"] for p in dev_all})
    small_uids = set(dev_uids[: args.val_ts])
    dev_small = [p for p in dev_all if p["sample_uid"] in small_uids]
    bench = data.benchmark_patches()

    n_tr_pos = sum(data.has_ci(p) for p in tr_patches)
    n_tr_bg = len(tr_patches) - n_tr_pos
    print(
        f"[DATA] Train: {len(tr_patches)} ({n_tr_pos} CI pos, {n_tr_bg} cold bg) | Dev: {len(dev_all)} | Bench: {len(bench)}"
    )

    # 2. Setup Datasets
    ds_train = CIPatchDataset_v2(
        tr_patches,
        "train",
        size=args.size,
        heatmap_size=args.heatmap_size,
        sigma=args.sigma,
        is_train=True,
    )
    ds_dev_small = CIPatchDataset_v2(
        dev_small,
        "val",
        size=args.size,
        heatmap_size=args.heatmap_size,
        sigma=args.sigma,
        is_train=False,
    )
    ds_dev_all = CIPatchDataset_v2(
        dev_all,
        "val",
        size=args.size,
        heatmap_size=args.heatmap_size,
        sigma=args.sigma,
        is_train=False,
    )
    ds_bench = CIPatchDataset_v2(
        bench,
        "val",
        size=args.size,
        heatmap_size=args.heatmap_size,
        sigma=args.sigma,
        is_train=False,
    )

    # 3. Setup Sampler
    if args.balanced_sampling:
        labels = ds_train.labels
        pos_mask = labels == 1.0
        w_pos = args.pos_ratio / max(1.0, float(np.sum(pos_mask)))
        w_neg = (1.0 - args.pos_ratio) / max(1.0, float(np.sum(~pos_mask)))
        weights = np.where(pos_mask, w_pos, w_neg)
        sampler = WeightedRandomSampler(
            weights=torch.as_tensor(weights, dtype=torch.double),
            num_samples=len(tr_patches),
            replacement=True,
        )
        train_shuffle = False
        print(
            f"[SAMPLER] WeightedRandomSampler enabled: pos_ratio={args.pos_ratio:.2f} (Pos weight: {w_pos:.3e}, Neg weight: {w_neg:.3e})"
        )
    else:
        sampler = None
        train_shuffle = True

    dl_train = DataLoader(
        ds_train,
        batch_size=args.batch,
        shuffle=train_shuffle,
        sampler=sampler,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=args.workers > 0,
        worker_init_fn=lambda _: _worker_init("train"),
    )
    dl_dev_small = DataLoader(
        ds_dev_small,
        batch_size=args.batch * 2,
        shuffle=False,
        num_workers=4,
        worker_init_fn=lambda _: _worker_init("val"),
    )
    dl_dev_all = DataLoader(
        ds_dev_all,
        batch_size=args.batch * 2,
        shuffle=False,
        num_workers=6,
        worker_init_fn=lambda _: _worker_init("val"),
    )
    dl_bench = DataLoader(
        ds_bench,
        batch_size=args.batch * 2,
        shuffle=False,
        num_workers=6,
        worker_init_fn=lambda _: _worker_init("val"),
    )

    # 4. Instantiate Model & Optimizer
    model = CINet_v2(
        in_ch=41, width=32, heatmap_size=args.heatmap_size
    ).to(device)
    if args.init:
        sd = torch.load(args.init, map_location=device)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(
            f"[INIT] Loaded checkpoint {args.init} (missing: {len(missing)}, unexpected: {len(unexpected)})",
            flush=True,
        )

    opt = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.epochs, eta_min=1e-5
    )

    # 5. Training Loop
    t_start = time.time()
    best_dev_auc = -1.0
    history = []

    print("[TRAIN] Starting training loop...")
    for ep in range(1, args.epochs + 1):
        model.train()
        tot_loss = 0.0
        tot_cls = 0.0
        tot_hm = 0.0
        n_samples = 0

        for x, y_cls, y_hm in dl_train:
            if time.time() - t_start > args.minutes * 60:
                print("[TRAIN] Time budget reached during epoch.", flush=True)
                break

            x = x.to(device, non_blocking=True)
            y_cls = y_cls.to(device, non_blocking=True)
            y_hm = y_hm.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            cls_logits, hm_logits = model(x, return_heatmap=True)

            loss_cls = F.binary_cross_entropy_with_logits(cls_logits, y_cls)
            loss_hm = gaussian_focal_loss(hm_logits, y_hm)
            loss_total = loss_cls + args.lambda_hm * loss_hm

            loss_total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            opt.step()

            bs = x.shape[0]
            tot_loss += loss_total.item() * bs
            tot_cls += loss_cls.item() * bs
            tot_hm += loss_hm.item() * bs
            n_samples += bs

        sched.step()

        # Validation on small Dev subset
        p_dev_s, hm_max_s = evaluate_probs(model, dl_dev_small, device)
        y_dev_s = ds_dev_small.labels
        rk_metrics = M.ranking(y_dev_s, p_dev_s)
        dev_auc = rk_metrics["ROC_AUC"]
        dev_pr = rk_metrics["PR_AUC"]

        elapsed_min = (time.time() - t_start) / 60.0
        avg_loss = tot_loss / max(1, n_samples)
        avg_cls = tot_cls / max(1, n_samples)
        avg_hm = tot_hm / max(1, n_samples)

        history.append(
            {
                "epoch": ep,
                "loss_total": avg_loss,
                "loss_cls": avg_cls,
                "loss_hm": avg_hm,
                "dev_auc": dev_auc,
                "dev_pr": dev_pr,
                "minutes": elapsed_min,
            }
        )
        print(
            f"Epoch {ep:2d}/{args.epochs} | Total: {avg_loss:.4f} (CLS: {avg_cls:.4f}, HM: {avg_hm:.4f}) | Dev ROC-AUC: {dev_auc:.4f}, PR-AUC: {dev_pr:.4f} | {elapsed_min:.1f} min",
            flush=True,
        )

        if dev_auc > best_dev_auc:
            best_dev_auc = dev_auc
            torch.save(model.state_dict(), out_dir / "best.pt")
            print(f"  --> Saved new best checkpoint (AUC {best_dev_auc:.4f})")

        if time.time() - t_start > args.minutes * 60:
            print("[TRAIN] Overall time budget reached.", flush=True)
            break

    # 6. Final Evaluation on full Dev and official Benchmark
    print("\n[EVAL] Loading best checkpoint for full evaluation...", flush=True)
    model.load_state_dict(torch.load(out_dir / "best.pt", map_location=device))

    p_dev_all, hm_dev_all = evaluate_probs(model, dl_dev_all, device)
    p_bench, hm_bench = evaluate_probs(model, dl_bench, device)

    y_dev_all = ds_dev_all.labels
    y_bench = ds_bench.labels

    res = {
        "dev": {
            "n": len(y_dev_all),
            "pos": int(y_dev_all.sum()),
            **M.ranking(y_dev_all, p_dev_all),
        },
        "bench": {
            "n": len(y_bench),
            "pos": int(y_bench.sum()),
            **M.ranking(y_bench, p_bench),
        },
        "history": history,
    }

    # Save output artifacts
    np.savez_compressed(
        out_dir / "probs.npz",
        p_dev=p_dev_all,
        y_dev=y_dev_all,
        hm_dev=hm_dev_all,
        p_bench=p_bench,
        y_bench=y_bench,
        hm_bench=hm_bench,
    )
    with open(out_dir / "metrics.json", "w") as fh:
        json.dump(res, fh, indent=2)

    print("\n[RESULTS SUMMARY]")
    print(
        json.dumps(
            {k: v for k, v in res.items() if k != "history"},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
