"""Dense Multi-GPU Full-Domain Continuous Perception Engine (tools/dense_inference_v2.py).

Features:
1. Loads 5-Seed CINet_v2 weights (experiments/p2_seed{0..4}/best.pt) and calibration parameters;
2. Implements sliding stride (default stride=56 px, ~62 km) with 2D Hanning (cosine) window blending;
3. Reconstructs smooth full-domain probability field P_full(y, x) and keypoint heatmap H_full(y, x) on native 2001x2101 grid;
4. Dispatches timestamps across RTX 4090 GPUs 1, 2, 3, 4 in parallel;
5. Extracts candidate peaks via 2D NMS and 32x32 local microphysical windows for targeted gating.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from scipy.ndimage import maximum_filter
import torch
import zarr

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from ewci import data
from ewci.data import IR_SLICE, BTD_PAIRS, IR7, SCALE, FILL_VALUE
from ewci.model_v2 import CINet_v2
from ewci.tensors import _MEAN, _STD, _IDX
import tools.eval_localization as el

FULL = Path("/data1/group3/DATASAT/experiments/ci_det_v2/annotations_full")
D = PROJECT / "experiments"
STORE_PATH = "/data1/group3/DATASAT/dataset/dataset.zarr"

LAT0, LON0 = 40.00448, 103.99570
H, W, S = 2001, 2101, 224
EARTH_RADIUS_KM = 6371.0


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2.0) ** 2
    return EARTH_RADIUS_KM * 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))


def build_full_domain_tensor(raw: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Convert raw [4, 7, H, W] int16 array to 41-channel full tensor and physical BT array."""
    bt = raw.astype(np.float32) * SCALE
    bt[raw == FILL_VALUE] = np.nan

    mask_valid = (raw != FILL_VALUE)
    normed = (np.nan_to_num(bt) - _MEAN[None, :, None, None]) / _STD[None, :, None, None]
    normed[~mask_valid] = 0.0

    last, first = bt[-1], bt[0]
    planes = [normed.reshape(28, *normed.shape[-2:])]
    planes.append(np.stack([(last[_IDX[a]] - last[_IDX[b]]) / 5.0 for a, b in BTD_PAIRS]))
    planes.append((last - first) / 2.0)

    full_tensor = np.concatenate(planes, axis=0).astype(np.float32)
    np.nan_to_num(full_tensor, copy=False)
    return full_tensor, bt


def _gpu_worker_dense_pipeline(
    gpu_id: int,
    uid_list: List[str],
    role: str,
    smp_dict: Dict[str, dict],
    stride: int,
    batch_size: int,
    tau_prob: float,
    tau_score: float,
    vis_uid: str,
    out_dir: Path,
    return_dict: dict,
):
    """Worker process running on a dedicated GPU for dense sliding inference."""
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"[GPU {gpu_id}] Initialized worker for {len(uid_list)} timestamps on {device}", flush=True)

        # 1. Load 5-seed CINet_v2 models
        ckpts = [D / f"p2_seed{s}/best.pt" for s in range(5)]
        models = []
        for c in ckpts:
            m = CINet_v2(in_ch=41, width=32, heatmap_size=28).to(device)
            m.load_state_dict(torch.load(c, map_location=device))
            m.eval()
            models.append(m)

        # 2. Precompute 2D Hann window
        w = np.sin(np.pi * (np.arange(S) + 0.5) / S) ** 2
        W2d = np.outer(w, w).astype(np.float32)

        # 3. Precompute grid slicing coordinates
        rows = sorted(set(list(range(0, H - S + 1, stride)) + [H - S]))
        cols = sorted(set(list(range(0, W - S + 1, stride)) + [W - S]))
        patches_coords = [(r, c) for r in rows for c in cols]
        n_patches = len(patches_coords)

        # 4. Open Zarr store per worker
        root = zarr.open(STORE_PATH, mode="r")
        z_obs = root[role]["obs"]

        idx_b13 = IR7.index("B13")
        idx_b10 = IR7.index("B10")
        idx_b11 = IR7.index("B11")

        results_by_uid = {}

        for u_idx, uid in enumerate(uid_list):
            t_start = time.time()
            oi = int(smp_dict[uid]["obs_index"])

            # Read 4 frames (235 MB)
            raw = np.asarray(z_obs[oi - 4 : oi, IR_SLICE, :, :])
            full_tensor, bt = build_full_domain_tensor(raw)

            b13_t10 = bt[3, idx_b13]
            b13_t40 = bt[0, idx_b13]
            b10_t10 = bt[3, idx_b10]
            b11_t10 = bt[3, idx_b11]

            # Batched model inference with 4-flip TTA
            all_p, all_hm = [], []
            with torch.no_grad():
                for i in range(0, n_patches, batch_size):
                    chunk_coords = patches_coords[i : i + batch_size]
                    B = len(chunk_coords)
                    bx = np.stack([full_tensor[:, r : r + S, c : c + S] for r, c in chunk_coords])
                    bx_t = torch.from_numpy(bx).to(device)

                    ens_p = torch.zeros(B, device=device)
                    ens_hm = torch.zeros((B, 1, 28, 28), device=device)
                    for m in models:
                        c_l, h_l = m(bx_t, return_heatmap=True)
                        p_acc = torch.sigmoid(c_l)
                        h_acc = torch.sigmoid(h_l)
                        for dims in ([3], [2], [2, 3]):
                            c_flip, h_flip = m(torch.flip(bx_t, dims=dims), return_heatmap=True)
                            p_acc = p_acc + torch.sigmoid(c_flip)
                            h_acc = h_acc + torch.flip(torch.sigmoid(h_flip), dims=dims)
                        ens_p = ens_p + p_acc / 4.0
                        ens_hm = ens_hm + h_acc / 4.0

                    all_p.append((ens_p / len(models)).cpu().numpy())
                    ens_hm = (ens_hm / len(models))
                    ens_hm_224 = torch.nn.functional.interpolate(
                        ens_hm, size=(S, S), mode="bilinear", align_corners=False
                    ).squeeze(1).cpu().numpy()
                    all_hm.append(ens_hm_224)

            all_p = np.concatenate(all_p)
            all_hm = np.concatenate(all_hm)

            # 2D Hann window blending
            P_acc = np.zeros((H, W), dtype=np.float32)
            H_acc = np.zeros((H, W), dtype=np.float32)
            W_acc = np.zeros((H, W), dtype=np.float32)

            for (r, c), p_val, hm_patch in zip(patches_coords, all_p, all_hm):
                P_acc[r : r + S, c : c + S] += W2d * p_val
                H_acc[r : r + S, c : c + S] += W2d * hm_patch
                W_acc[r : r + S, c : c + S] += W2d

            P_full = P_acc / np.maximum(W_acc, 1e-6)
            H_full = H_acc / np.maximum(W_acc, 1e-6)

            # Save full resolution fields for the representative visual timestamp
            if uid == vis_uid:
                vis_out = out_dir / "dense_vis_sample.npz"
                np.savez_compressed(
                    vis_out,
                    uid=uid,
                    P_full=P_full.astype(np.float16),
                    H_full=H_full.astype(np.float16),
                    bt13_t10=b13_t10.astype(np.float16),
                    bt13_t40=b13_t40.astype(np.float16),
                )
                print(f"[GPU {gpu_id}] Saved visualization sample {uid} to {vis_out}", flush=True)

            # Candidate peak extraction via score map
            H_max = float(H_full.max()) if H_full.size else 1.0
            H_norm = H_full / max(H_max, 1e-6)
            score_map = P_full * H_norm

            mf = maximum_filter(score_map, size=27, mode="constant", cval=0.0)
            peaks_mask = (score_map == mf) & (score_map >= tau_score) & (P_full >= tau_prob)
            ys, xs = np.nonzero(peaks_mask)
            vals = score_map[ys, xs]
            order = np.argsort(vals)[::-1]

            # Greedy Haversine NMS with 30 km radius
            selected_peaks = []
            for idx in order:
                y, x = int(ys[idx]), int(xs[idx])

                # Extract 32x32 local physical window
                r0, r1 = max(0, y - 16), min(H, y + 16)
                c0, c1 = max(0, x - 16), min(W, x + 16)

                w_b13_10 = b13_t10[r0:r1, c0:c1]
                w_b13_40 = b13_t40[r0:r1, c0:c1]
                w_b10_10 = b10_t10[r0:r1, c0:c1]
                w_b11_10 = b11_t10[r0:r1, c0:c1]

                min_b13_10 = float(np.nanmin(w_b13_10)) if np.isfinite(w_b13_10).any() else np.nan

                # Local cooling rate in 32x32 window
                diff_b13 = w_b13_10 - w_b13_40
                delta_bt13_local = float(np.nanmin(diff_b13)) if np.isfinite(diff_b13).any() else 0.0

                # Water vapor difference and glaciation index
                diff_10_13 = w_b10_10 - w_b13_10
                b10_b13_local = float(np.nanmax(diff_10_13)) if np.isfinite(diff_10_13).any() else -99.0

                diff_11_13 = w_b11_10 - w_b13_10
                b11_b13_local = float(np.nanmin(diff_11_13)) if np.isfinite(diff_11_13).any() else 0.0

                # Microphysical coldest centroid within 32x32 window
                if np.isfinite(w_b13_10).any():
                    ymin, xmin = np.unravel_index(np.nanargmin(w_b13_10), w_b13_10.shape)
                    y_phys = float(r0 + ymin)
                    x_phys = float(c0 + xmin)
                else:
                    y_phys, x_phys = float(y), float(x)

                # Dual coordinates (alpha = 0.5)
                y_dual = 0.5 * float(y) + 0.5 * y_phys
                x_dual = 0.5 * float(x) + 0.5 * x_phys

                lat = float(LAT0 - 0.01 * y_dual)
                lon = float(LON0 + 0.01 * x_dual)

                # Check conflict with existing selected peaks
                conflict = False
                for p in selected_peaks:
                    if haversine_km(lat, lon, p["lat"], p["lon"]) < 30.0:
                        conflict = True
                        break

                if not conflict:
                    selected_peaks.append(
                        {
                            "uid": uid,
                            "lat": lat,
                            "lon": lon,
                            "y_nn": int(y),
                            "x_nn": int(x),
                            "y_phys": float(y_phys),
                            "x_phys": float(x_phys),
                            "y_dual": float(y_dual),
                            "x_dual": float(x_dual),
                            "prob": float(P_full[y, x]),
                            "score": float(score_map[y, x]),
                            "hm_val": float(H_full[y, x]),
                            "bt13_local": min_b13_10,
                            "delta_bt13_local": delta_bt13_local,
                            "b10_b13_local": b10_b13_local,
                            "b11_b13_local": b11_b13_local,
                        }
                    )

            results_by_uid[uid] = selected_peaks
            print(
                f"[GPU {gpu_id}] [{u_idx + 1}/{len(uid_list)}] {uid} done in {time.time() - t_start:.2f}s | Peaks: {len(selected_peaks)}",
                flush=True,
            )

        return_dict[gpu_id] = results_by_uid
        print(f"[GPU {gpu_id}] All assigned timestamps completed successfully.", flush=True)

    except Exception as exc:
        print(f"[ERROR in GPU {gpu_id}]: {exc}", flush=True)
        import traceback
        traceback.print_exc()
        raise exc


def main():
    ap = argparse.ArgumentParser(description="Dense Multi-GPU Full-Domain Continuous Perception Engine")
    ap.add_argument("--split", choices=["dev", "benchmark", "test", "both", "all"], default="both")
    ap.add_argument("--gpus", default="1,2,3,4")
    ap.add_argument("--stride", type=int, default=56)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--tau-prob", type=float, default=0.20)
    ap.add_argument("--tau-score", type=float, default=0.10)
    ap.add_argument("--vis-uid", default="val/140/SCW2021-0065")
    ap.add_argument("--out-dir", default="experiments/final")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    gpu_list = [int(g.strip()) for g in args.gpus.split(",") if g.strip()]
    print(f"==================================================")
    print(f"Dense Continuous Perception Engine (Stride: {args.stride} px, GPUs: {gpu_list})")
    print(f"==================================================")

    if args.split == "all":
        splits_to_run = ["dev", "benchmark", "test"]
    elif args.split == "both":
        splits_to_run = ["benchmark", "test"]
    else:
        splits_to_run = [args.split]

    for s_name in splits_to_run:
        role = "test" if s_name == "test" else "val"
        if s_name == "benchmark":
            uids = sorted(list(dict.fromkeys([p["sample_uid"] for p in data.benchmark_patches()])))
        elif s_name == "dev":
            # Dev is needed to select the operating threshold without touching
            # benchmark/test (EXPERIMENTS E14 protocol breach).
            dev_uids = set(json.loads((PROJECT / "splits_v1.json").read_text())["dev"]["early_stop_uids"])
            uids = sorted(dev_uids)
        else:
            uids = sorted(list(dict.fromkeys(json.loads((PROJECT / "splits_test_eval.json").read_text())["uids"])))

        print(f"\n[{s_name.upper()}] Total timestamps to process: {len(uids)} on role '{role}'")
        # Samples must come from the same annotation source as the patches.
        # annotations_stage1_bg/val_samples.jsonl is a symlink to an older,
        # smaller set (950 records vs 1483) that misses 12 of the 100 dev
        # timestamps. obs_index agrees on all 950 shared uids, so
        # annotations_full is a safe superset.
        smp_dict = {}
        sample_file = FULL / ("test_samples.jsonl" if role == "test" else "val_samples.jsonl")
        with open(sample_file, "r", encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    rec = json.loads(line)
                    smp_dict[rec["sample_uid"]] = rec

        # Split across available GPUs
        chunks = np.array_split(uids, len(gpu_list))
        manager = mp.Manager()
        return_dict = manager.dict()
        processes = []

        t0_split = time.time()
        for gpu_id, chunk_uids in zip(gpu_list, chunks):
            if len(chunk_uids) == 0:
                continue
            p = mp.Process(
                target=_gpu_worker_dense_pipeline,
                args=(
                    gpu_id,
                    chunk_uids.tolist(),
                    role,
                    smp_dict,
                    args.stride,
                    args.batch_size,
                    args.tau_prob,
                    args.tau_score,
                    args.vis_uid if s_name == "benchmark" else "",
                    out_dir,
                    return_dict,
                ),
            )
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        # Check errors
        if len(return_dict) != len(processes):
            raise RuntimeError(f"Some GPU workers failed. Expected {len(processes)} results, got {len(return_dict)}")

        # Merge results
        merged_candidates = {}
        for gpu_id in return_dict:
            merged_candidates.update(return_dict[gpu_id])

        out_cache = out_dir / f"dense_candidates_{s_name}.json"
        with open(out_cache, "w", encoding="utf-8") as fh:
            json.dump(merged_candidates, fh, indent=2)

        tot_peaks = sum(len(peaks) for peaks in merged_candidates.values())
        print(f"[{s_name.upper()}] Done in {time.time() - t0_split:.1f}s | Saved {tot_peaks} candidate peaks to {out_cache}")

    print("\n==================================================")
    print("Dense Inference Engine completed successfully.")
    print("==================================================")


if __name__ == "__main__":
    main()
