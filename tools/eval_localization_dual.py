"""Neural-Physics Dual-Guidance Object Localization and Decaying Anvil Gating System.

Integrates:
1. CINet_v2 5-seed 4-flip TTA continuous 28x28 keypoint heatmap peak coordinates (y_nn, x_nn);
2. Satellite microphysical cold-top overshooting and rapid cooling centroid (y_phys, x_phys);
3. Posterior dual-guidance alignment: (y*, x*) = alpha * (y_nn, x_nn) + (1 - alpha) * (y_phys, x_phys);
4. Decaying anvil physical gating: Delta BT13 = BT13(T-10) - BT13(T-40) > -1.0 K false alarm suppression;
5. Operational evaluation at 30 km tolerance across Benchmark (30 timestamps) and Held-out Test (200 timestamps).
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import maximum_filter
import torch
from torch.utils.data import DataLoader, Dataset

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from ewci import data
from ewci.model_v2 import CINet_v2
from ewci.tensors import build_input
import tools.eval_localization as el
import tools.eval_localization_physics as elp

FULL = Path("/data1/group3/DATASAT/experiments/ci_det_v2/annotations_full")
D = PROJECT / "experiments"

LAT0, LON0 = el.LAT0, el.LON0
H, W, S = el.H, el.W, el.S
GRID_STEP = el.GRID_STEP


class ActivePatchDataset(Dataset):
    """Dataset for active convective candidate patches."""

    def __init__(self, patches: List[dict], role: str, size: int = 224):
        self.patches = patches
        self.role = role
        self.size = size
        self.reader: Optional[data.ObsReader] = None

    def __len__(self) -> int:
        return len(self.patches)

    def __getitem__(self, idx: int) -> torch.Tensor:
        if self.reader is None:
            self.reader = data.ObsReader(self.role)
        raw = self.reader.read_patch(self.patches[idx])
        return torch.from_numpy(build_input(raw, self.size))


def _gpu_worker_infer_heatmaps(
    gpu_id: int,
    patch_indices: List[int],
    all_patches_meta: List[dict],
    role: str,
    ckpt_paths: List[Path],
    return_dict: dict,
):
    """Worker process for 5-seed 4-flip TTA heatmap inference on a dedicated GPU."""
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        sub_patches = [all_patches_meta[i] for i in patch_indices]
        ds = ActivePatchDataset(sub_patches, role, size=224)
        dl = DataLoader(ds, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)

        models = []
        for ckpt in ckpt_paths:
            m = CINet_v2(in_ch=41, width=32, heatmap_size=28).to(device)
            m.load_state_dict(torch.load(ckpt, map_location=device))
            m.eval()
            models.append(m)

        nn_coords = []
        nn_max_vals = []

        with torch.no_grad():
            for bx in dl:
                bx = bx.to(device, non_blocking=True)
                B = bx.shape[0]
                ens_hm = torch.zeros((B, 1, 28, 28), device=device)
                for m in models:
                    _, hl = m(bx, return_heatmap=True)
                    acc = torch.sigmoid(hl)
                    # 4-flip TTA: [3] horizontal, [2] vertical, [2, 3] both
                    for dims in ([3], [2], [2, 3]):
                        x_flip = torch.flip(bx, dims=dims)
                        _, h_flip = m(x_flip, return_heatmap=True)
                        acc = acc + torch.flip(torch.sigmoid(h_flip), dims=dims)
                    ens_hm = ens_hm + (acc / 4.0)
                ens_hm = ens_hm / float(len(models))

                # Global argmax on 28x28 heatmap
                ens_flat = ens_hm.view(B, -1)
                max_vals, argmaxes = ens_flat.max(dim=1)
                y_hm = (argmaxes // 28).cpu().numpy()
                x_hm = (argmaxes % 28).cpu().numpy()

                # Scale to sub-window patch coordinates [0, 224)
                y_sub = (y_hm.astype(np.float32) + 0.5) * (224.0 / 28.0)
                x_sub = (x_hm.astype(np.float32) + 0.5) * (224.0 / 28.0)

                for i in range(B):
                    nn_coords.append((float(y_sub[i]), float(x_sub[i])))
                    nn_max_vals.append(float(max_vals[i].item()))

        return_dict[gpu_id] = (patch_indices, nn_coords, nn_max_vals)
    except Exception as exc:
        print(f"[ERROR in GPU worker {gpu_id}]: {exc}", flush=True)
        raise exc


def extract_active_patches_meta(
    split: str,
    tau: float,
) -> Tuple[List[str], Dict[str, Any], List[Tuple[str, int, int, float, dict]], Dict[str, set], Dict[str, List[dict]]]:
    """Load split predictions, ground truth, and locate active candidate patches."""
    role = "test" if split == "test" else "val"
    npz_name = {
        "dev": "final/cinet_v2_dev_predictions.npz",
        "benchmark": "final/cinet_v2_bench_predictions.npz",
        "test": "final/cinet_v2_test_predictions.npz",
    }[split]
    npz_file = D / npz_name
    if not npz_file.exists():
        raise FileNotFoundError(f"Predictions file {npz_file} not found.")

    pred_data = np.load(npz_file, allow_pickle=True)
    uids = [str(u) for u in pred_data["uid"]]
    rows = [int(r) for r in pred_data["row"]]
    cols = [int(c) for c in pred_data["col"]]
    probs = pred_data["p_cal"]

    unique_uids = sorted(list(dict.fromkeys(uids)))
    grid_rows, grid_cols = el.get_grid_coords()
    n_rows, n_cols = len(grid_rows), len(grid_cols)

    # Load ground truth objects
    gt_map = el.load_ground_truth_objects(role, set(unique_uids))

    # Probability grid per timestamp
    ts_data = {u: {} for u in unique_uids}
    for u, r, c, p in zip(uids, rows, cols, probs):
        ts_data[u][(r, c)] = float(p)

    # Detect local_max peaks (3x3 spatial NMS)
    local_max_peaks = {u: set() for u in unique_uids}
    for uid in unique_uids:
        grid = np.zeros((n_rows, n_cols), dtype=np.float32)
        for ri, r in enumerate(grid_rows):
            for cj, c in enumerate(grid_cols):
                grid[ri, cj] = ts_data[uid].get((r, c), 0.0)
        mf = maximum_filter(grid, size=3, mode="constant", cval=0.0)
        is_peak = (grid == mf) & (grid >= tau)
        visited = set()
        for i in range(n_rows):
            for j in range(n_cols):
                if is_peak[i, j] and (i, j) not in visited:
                    already = False
                    for di in (-1, 0, 1):
                        for dj in (-1, 0, 1):
                            ni, nj = i + di, j + dj
                            if (ni, nj) in visited and grid[ni, nj] == grid[i, j]:
                                already = True
                                break
                        if already:
                            break
                    if not already:
                        visited.add((i, j))
                        local_max_peaks[uid].add((grid_rows[i], grid_cols[j]))

    # Load patch dictionaries
    print(f"[{split.upper()}] Loading patches for role '{role}'...")
    if split == "benchmark":
        all_patches = data.benchmark_patches()
    elif split == "dev":
        dev_uids = set(json.loads((PROJECT / "splits_v1.json").read_text())["dev"]["early_stop_uids"])
        all_patches = [p for p in data.load_patches_from(FULL, "val") if p["sample_uid"] in dev_uids]
    else:
        test_uids = set(json.loads((PROJECT / "splits_test_eval.json").read_text())["uids"])
        all_patches = [p for p in data.load_patches_from(FULL, "test") if p["sample_uid"] in test_uids]

    patch_dict = {(p["sample_uid"], int(p["row"]), int(p["col"])): p for p in all_patches}

    active_meta = []
    for u in unique_uids:
        for (r, c), p in ts_data[u].items():
            if p >= tau:
                p_meta = patch_dict[(u, r, c)]
                active_meta.append((u, r, c, p, p_meta))

    return unique_uids, ts_data, active_meta, local_max_peaks, gt_map


def extract_features_and_coordinates(
    split: str,
    active_meta: List[Tuple[str, int, int, float, dict]],
    gpu_list: List[int],
) -> Dict[str, Any]:
    """Compute physics centroids and multi-GPU 5-seed TTA neural heatmaps."""
    role = "test" if split == "test" else "val"
    n_active = len(active_meta)
    print(f"[{split.upper()}] Total active candidate patches: {n_active}")

    # Load obs_index
    uid2obs = {}
    with open(FULL / f"{role}_samples.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                s = json.loads(line)
                uid2obs[s["sample_uid"]] = int(s["obs_index"])

    reader = data.ObsReader(role)
    b13_idx = data.STORE_BANDS.index("B13")

    # 1. Physics centroids & delta BT13
    print(f"[{split.upper()}] Computing satellite microphysics centroids and Delta BT13...")
    t0_phys = time.time()
    phys_coords = []
    cold_coords = []
    cool_coords = []
    d_min_bt13_list = []

    # Cache frames by timestamp to minimize I/O
    frame_cache = {}
    for u, r, c, p, p_meta in active_meta:
        oi = uid2obs[u]
        if oi not in frame_cache:
            f0 = reader.obs[oi - 4, b13_idx]
            f3 = reader.obs[oi - 1, b13_idx]
            frame_cache[oi] = (f0, f3)
        else:
            f0, f3 = frame_cache[oi]

        cd = elp.compute_patch_physics_coords(f0, f3, r, c)
        phys_coords.append(cd["physics"])
        cold_coords.append(cd["cold"])
        cool_coords.append(cd["cool"])

        # Delta BT13 over 30 min (T-10 minus T-40)
        b0_patch = f0[r : r + S, c : c + S]
        b3_patch = f3[r : r + S, c : c + S]
        fin3 = b3_patch != data.FILL_VALUE
        fin0 = b0_patch != data.FILL_VALUE
        min3 = float(b3_patch[fin3].min()) * data.SCALE if fin3.any() else np.nan
        min0 = float(b0_patch[fin0].min()) * data.SCALE if fin0.any() else np.nan
        d_min = (min3 - min0) if (np.isfinite(min3) and np.isfinite(min0)) else 0.0
        d_min_bt13_list.append(d_min)

    print(f"[{split.upper()}] Physics extraction finished in {time.time() - t0_phys:.1f}s")

    # 2. Multi-GPU 5-seed TTA Neural Heatmap Inference
    ckpt_paths = [D / f"p2_seed{s}/best.pt" for s in range(5)]
    all_patch_dicts = [p_meta for _, _, _, _, p_meta in active_meta]

    print(f"[{split.upper()}] Launching multi-GPU 5-seed TTA heatmap inference on GPUs {gpu_list}...")
    t0_nn = time.time()

    indices_splits = np.array_split(np.arange(n_active), len(gpu_list))
    manager = mp.Manager()
    return_dict = manager.dict()
    processes = []

    for gpu_id, idx_chunk in zip(gpu_list, indices_splits):
        if len(idx_chunk) == 0:
            continue
        p = mp.Process(
            target=_gpu_worker_infer_heatmaps,
            args=(
                gpu_id,
                idx_chunk.tolist(),
                all_patch_dicts,
                role,
                ckpt_paths,
                return_dict,
            ),
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    # Aggregate results back in order
    nn_coords = [None] * n_active
    nn_max_vals = [None] * n_active

    for gpu_id in return_dict:
        chunk_indices, chunk_coords, chunk_vals = return_dict[gpu_id]
        for orig_idx, coord, val in zip(chunk_indices, chunk_coords, chunk_vals):
            nn_coords[orig_idx] = coord
            nn_max_vals[orig_idx] = val

    print(f"[{split.upper()}] Neural heatmap inference completed in {time.time() - t0_nn:.1f}s")

    geom_coords = [(112.0, 112.0) for _ in range(n_active)]

    return {
        "geom": np.array(geom_coords, dtype=np.float32),
        "physics": np.array(phys_coords, dtype=np.float32),
        "cold": np.array(cold_coords, dtype=np.float32),
        "cool": np.array(cool_coords, dtype=np.float32),
        "nn": np.array(nn_coords, dtype=np.float32),
        "nn_max": np.array(nn_max_vals, dtype=np.float32),
        "d_min_bt13": np.array(d_min_bt13_list, dtype=np.float32),
        "probs": np.array([p for _, _, _, p, _ in active_meta], dtype=np.float32),
        "uids": [u for u, _, _, _, _ in active_meta],
        "rows": [r for _, r, _, _, _ in active_meta],
        "cols": [c for _, _, c, _, _ in active_meta],
    }


def evaluate_strategy_predictions(
    active_meta: List[Tuple[str, int, int, float, dict]],
    sub_coords: np.ndarray,
    probs: np.ndarray,
    d_min_bt13: np.ndarray,
    unique_uids: List[str],
    gt_map: Dict[str, List[dict]],
    local_max_peaks: Dict[str, set],
    peak_mode: str = "local_max",
    filter_decaying_anvil: bool = False,
    decaying_threshold_k: float = -1.0,
    tolerance_list: List[float] = (15.0, 30.0, 45.0, 60.0, 90.0, 112.0, 120.0),
    primary_tolerance_km: float = 30.0,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Run spatial bipartite matching for candidate predictions across tolerances."""
    preds_by_uid = {u: [] for u in unique_uids}
    for idx, (u, r, c, p, _) in enumerate(active_meta):
        if peak_mode == "local_max" and (r, c) not in local_max_peaks[u]:
            continue

        # Decaying anvil gate: Delta BT13 > -1.0 K (warming or cooling < 1 K)
        if filter_decaying_anvil and d_min_bt13[idx] > decaying_threshold_k:
            continue

        yr, xr = sub_coords[idx]
        lat = LAT0 - 0.01 * (r + yr)
        lon = LON0 + 0.01 * (c + xr)
        preds_by_uid[u].append({"lat": float(lat), "lon": float(lon), "prob": float(probs[idx])})

    curve_results = {}
    primary_result = {}
    total_gt = sum(len(gt_map[u]) for u in unique_uids)
    total_preds = sum(len(preds_by_uid[u]) for u in unique_uids)

    for tol in tolerance_list:
        tot_hits = 0
        tot_miss = 0
        tot_fa = 0
        all_dists: List[float] = []

        for uid in unique_uids:
            h, m, fa, dists, _ = el.spatial_bipartite_matching(
                preds_by_uid[uid], gt_map[uid], tolerance_km=tol
            )
            tot_hits += h
            tot_miss += m
            tot_fa += fa
            all_dists.extend(dists)

        pod = tot_hits / (tot_hits + tot_miss) if (tot_hits + tot_miss) > 0 else 0.0
        far = tot_fa / (tot_hits + tot_fa) if (tot_hits + tot_fa) > 0 else 0.0
        csi = tot_hits / (tot_hits + tot_miss + tot_fa) if (tot_hits + tot_miss + tot_fa) > 0 else 0.0

        mean_dist = float(np.mean(all_dists)) if all_dists else float("nan")
        median_dist = float(np.median(all_dists)) if all_dists else float("nan")
        p25_dist = float(np.percentile(all_dists, 25)) if all_dists else float("nan")
        p75_dist = float(np.percentile(all_dists, 75)) if all_dists else float("nan")

        res = {
            "tolerance_km": float(tol),
            "hits": int(tot_hits),
            "misses": int(tot_miss),
            "false_alarms": int(tot_fa),
            "total_gt": int(total_gt),
            "total_preds": int(total_preds),
            "POD_obj": float(pod),
            "FAR_obj": float(far),
            "CSI_obj": float(csi),
            "mean_distance_km": mean_dist,
            "median_distance_km": median_dist,
            "p25_distance_km": p25_dist,
            "p75_distance_km": p75_dist,
        }
        curve_results[f"{tol:.1f}km"] = res
        if abs(tol - primary_tolerance_km) < 1e-4:
            primary_result = res

    return primary_result, curve_results


def generate_benchmark_figure(
    results: Dict[str, Any],
    out_path: Path,
):
    """Render scientific publication-grade multi-panel figure for neural-physics dual localization."""
    print(f"[PLOT] Rendering comprehensive scientific figure to {out_path}...")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(18, 11), dpi=300)
    plt.subplots_adjust(hspace=0.32, wspace=0.28)

    colors = {
        "geom": "#7f8c8d",      # Gray
        "heatmap": "#e67e22",   # Orange
        "physics": "#2980b9",   # Blue
        "dual": "#27ae60",      # Green (best)
        "gated": "#8e44ad",     # Purple
    }

    # Determine available splits
    has_bench = "benchmark" in results["splits"]
    has_test = "test" in results["splits"]
    ref_split = "test" if has_test else "benchmark"
    ref_p = results["splits"][ref_split]["modes"]["local_max"]

    # -------------------------------------------------------------
    # Subplot 1: Benchmark Strategy Comparison (30 km tolerance)
    # -------------------------------------------------------------
    ax = axes[0, 0]
    s1_split = "benchmark" if has_bench else "test"
    bench_p = results["splits"][s1_split]["modes"]["local_max"]
    bench_strats = [
        ("Strat A (Geom)", bench_p["strategy_a_geom"]["primary_30km"]),
        ("Strat B (Heatmap)", bench_p["strategy_b_heatmap"]["primary_30km"]),
        ("Strat C (Physics)", bench_p["strategy_c_physics"]["primary_30km"]),
        (f"Strat D (Dual a={bench_p['strategy_d_dual_optimal']['optimal_alpha']:.1f})", bench_p["strategy_d_dual_optimal"]["primary_30km"]),
    ]
    names = [s[0] for s in bench_strats]
    csis = [s[1]["CSI_obj"] * 100 for s in bench_strats]
    pods = [s[1]["POD_obj"] * 100 for s in bench_strats]
    x = np.arange(len(names))
    width = 0.35

    rects1 = ax.bar(x - width/2, csis, width, label="CSI_obj (%)", color=["#95a5a6", "#f39c12", "#3498db", "#2ecc71"])
    rects2 = ax.bar(x + width/2, pods, width, label="POD_obj (%)", color=["#7f8c8d", "#d35400", "#2980b9", "#27ae60"], alpha=0.85)

    ax.set_ylabel("Percentage (%)", fontsize=11, fontweight="bold")
    ax.set_title(f"(a) {s1_split.capitalize()}: Strategy Comparison (30 km)", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=18, ha="right", fontsize=9.5)
    ax.legend(loc="upper left", frameon=True)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    for r in rects1:
        h = r.get_height()
        ax.annotate(f"{h:.2f}%", xy=(r.get_x() + r.get_width() / 2, h), xytext=(0, 3),
                    textcoords="offset points", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    # -------------------------------------------------------------
    # Subplot 2: Held-out Test Set Strategy Comparison (30 km tolerance)
    # -------------------------------------------------------------
    ax = axes[0, 1]
    s2_split = "test" if has_test else "benchmark"
    test_p = results["splits"][s2_split]["modes"]["local_max"]
    test_strats = [
        ("Strat A (Geom)", test_p["strategy_a_geom"]["primary_30km"]),
        ("Strat B (Heatmap)", test_p["strategy_b_heatmap"]["primary_30km"]),
        ("Strat C (Physics)", test_p["strategy_c_physics"]["primary_30km"]),
        (f"Strat D (Dual a={test_p['strategy_d_dual_optimal']['optimal_alpha']:.1f})", test_p["strategy_d_dual_optimal"]["primary_30km"]),
    ]
    tnames = [s[0] for s in test_strats]
    t_csis = [s[1]["CSI_obj"] * 100 for s in test_strats]
    t_pods = [s[1]["POD_obj"] * 100 for s in test_strats]
    x_t = np.arange(len(tnames))

    r1 = ax.bar(x_t - width/2, t_csis, width, label="CSI_obj (%)", color=["#95a5a6", "#f39c12", "#3498db", "#2ecc71"])
    r2 = ax.bar(x_t + width/2, t_pods, width, label="POD_obj (%)", color=["#7f8c8d", "#d35400", "#2980b9", "#27ae60"], alpha=0.85)

    ax.set_ylabel("Percentage (%)", fontsize=11, fontweight="bold")
    ax.set_title(f"(b) {s2_split.capitalize()}: Object-Level Performance", fontsize=12, fontweight="bold")
    ax.set_xticks(x_t)
    ax.set_xticklabels(tnames, rotation=18, ha="right", fontsize=9.5)
    ax.legend(loc="upper left", frameon=True)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    for r in r1:
        h = r.get_height()
        ax.annotate(f"{h:.2f}%", xy=(r.get_x() + r.get_width() / 2, h), xytext=(0, 3),
                    textcoords="offset points", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    # -------------------------------------------------------------
    # Subplot 3: Tolerance Sensitivity Curve
    # -------------------------------------------------------------
    ax = axes[0, 2]
    tols = [15.0, 30.0, 45.0, 60.0, 90.0, 112.0, 120.0]
    tol_keys = [f"{t:.1f}km" for t in tols]

    geom_curve = [ref_p["strategy_a_geom"]["tolerance_curve"][k]["CSI_obj"] * 100 for k in tol_keys]
    hm_curve = [ref_p["strategy_b_heatmap"]["tolerance_curve"][k]["CSI_obj"] * 100 for k in tol_keys]
    phys_curve = [ref_p["strategy_c_physics"]["tolerance_curve"][k]["CSI_obj"] * 100 for k in tol_keys]
    dual_curve = [ref_p["strategy_d_dual_optimal"]["tolerance_curve"][k]["CSI_obj"] * 100 for k in tol_keys]

    ax.plot(tols, geom_curve, "o--", color=colors["geom"], label="Strat A: Geom Center", linewidth=1.8, markersize=5)
    ax.plot(tols, hm_curve, "s--", color=colors["heatmap"], label="Strat B: Neural Heatmap", linewidth=1.8, markersize=5)
    ax.plot(tols, phys_curve, "^-.", color=colors["physics"], label="Strat C: Satellite Physics", linewidth=1.8, markersize=5)
    ax.plot(tols, dual_curve, "D-", color=colors["dual"], label="Strat D: Dual Fusion (Ours)", linewidth=2.4, markersize=6)

    ax.axvline(30.0, color="crimson", linestyle=":", alpha=0.7, label="Operational Tol (30 km)")
    ax.set_xlabel("Spatial Matching Tolerance Radius (km)", fontsize=11, fontweight="bold")
    ax.set_ylabel("CSI_obj (%)", fontsize=11, fontweight="bold")
    ax.set_title(f"(c) Operational Tolerance Curve ({ref_split.capitalize()})", fontsize=12, fontweight="bold")
    ax.legend(loc="lower right", frameon=True, fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.5)

    # -------------------------------------------------------------
    # Subplot 4: Alpha Sensitivity Optimization Curve
    # -------------------------------------------------------------
    ax = axes[1, 0]
    # The alpha synergy curve must come from dev: benchmark/test are scored at a
    # single frozen alpha, so their "sweep" has only one point.
    alpha_split = "dev" if "dev" in results["splits"] else ref_split
    alpha_sweep_ref = results["splits"][alpha_split]["modes"]["local_max"]["alpha_sweep"]
    alpha_vals = [item["alpha"] for item in alpha_sweep_ref]
    csi_alphas = [item["CSI_obj"] * 100 for item in alpha_sweep_ref]
    dist_alphas = [item["mean_distance_km"] for item in alpha_sweep_ref]

    ax2 = ax.twinx()
    l1 = ax.plot(alpha_vals, csi_alphas, "o-", color="#27ae60", linewidth=2.2, label="CSI_obj (%)")
    l2 = ax2.plot(alpha_vals, dist_alphas, "s--", color="#c0392b", linewidth=1.8, label="Mean Dist (km)")

    ax.set_xlabel(r"Neural Heatmap Weight $\alpha$ (Physics Weight = $1 - \alpha$)", fontsize=11, fontweight="bold")
    ax.set_ylabel("CSI_obj (%)", color="#27ae60", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Mean Distance Error (km)", color="#c0392b", fontsize=11, fontweight="bold")
    ax.set_title(rf"(d) $\alpha$ Synergy Analysis ({alpha_split.capitalize()}, selection set)", fontsize=12, fontweight="bold")

    lines = l1 + l2
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, loc="center left", frameon=True)
    ax.grid(True, linestyle="--", alpha=0.5)

    # -------------------------------------------------------------
    # Subplot 5: Decaying Anvil Gating False Alarm Suppression
    # -------------------------------------------------------------
    ax = axes[1, 1]
    gating_ref = ref_p["decaying_anvil_ablation"]
    g_labels = ["Ungated\n(All Peaks)", "Decaying Anvil\nGated (dBT13<-1K)"]
    fas = [gating_ref["ungated"]["false_alarms"], gating_ref["gated"]["false_alarms"]]
    fars = [gating_ref["ungated"]["FAR_obj"] * 100, gating_ref["gated"]["FAR_obj"] * 100]

    x_g = np.arange(2)
    b_fa = ax.bar(x_g - width/2, fas, width, label="False Alarms (Counts)", color=["#e74c3c", "#8e44ad"])
    ax_far = ax.twinx()
    b_far = ax_far.plot(x_g + width/2, fars, "D--", color="#f39c12", label="FAR_obj (%)", linewidth=2.2, markersize=7)

    ax.set_xticks(x_g)
    ax.set_xticklabels(g_labels, fontsize=10, fontweight="bold")
    ax.set_ylabel("False Alarm Count", color="#c0392b", fontsize=11, fontweight="bold")
    ax_far.set_ylabel("FAR_obj (%)", color="#f39c12", fontsize=11, fontweight="bold")
    ax.set_title(rf"(e) Decaying Anvil Gate ({ref_split.capitalize()})", fontsize=12, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    for r in b_fa:
        h = r.get_height()
        ax.annotate(f"{int(h)}", xy=(r.get_x() + r.get_width() / 2, h), xytext=(0, 3),
                    textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")

    # -------------------------------------------------------------
    # Subplot 6: Localization Distance Error Distribution
    # -------------------------------------------------------------
    ax = axes[1, 2]
    strategies_dist = [
        ("Strat A: Geom", ref_p["strategy_a_geom"]["primary_30km"]),
        ("Strat B: Heatmap", ref_p["strategy_b_heatmap"]["primary_30km"]),
        ("Strat C: Physics", ref_p["strategy_c_physics"]["primary_30km"]),
        ("Strat D: Dual Fusion", ref_p["strategy_d_dual_optimal"]["primary_30km"]),
    ]
    bar_names = [s[0].split(":")[1].strip() for s in strategies_dist]
    means = [s[1]["mean_distance_km"] for s in strategies_dist]
    medians = [s[1]["median_distance_km"] for s in strategies_dist]

    x_d = np.arange(len(bar_names))
    ax.bar(x_d - width/2, means, width, label="Mean Dist Error (km)", color="#34495e")
    ax.bar(x_d + width/2, medians, width, label="Median Dist Error (km)", color="#16a085")

    ax.set_ylabel("Distance to Ground Truth (km)", fontsize=11, fontweight="bold")
    ax.set_title(f"(f) Distance to Ground Truth CI ({ref_split.capitalize()})", fontsize=12, fontweight="bold")
    ax.set_xticks(x_d)
    ax.set_xticklabels(bar_names, rotation=15, ha="right", fontsize=9.5)
    ax.legend(loc="upper right", frameon=True)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[PLOT] Successfully saved scientific figure to: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Neural-Physics Dual Centroiding & Decaying Anvil Gating System.")
    parser.add_argument("--split", choices=["dev", "benchmark", "test", "all"], default="all")
    parser.add_argument("--tolerance", type=float, default=30.0)
    parser.add_argument("--gpus", type=str, default="1,2,3,4")
    parser.add_argument("--cache", type=str, default="experiments/final/cinet_v2_localization_dual_cache.npz")
    parser.add_argument("--out-metrics", type=str, default="experiments/final/cinet_v2_localization_dual_metrics.json")
    parser.add_argument("--out-fig", type=str, default="reports/figures/localization_dual_benchmark.png")
    parser.add_argument("--force-recompute", action="store_true")
    args = parser.parse_args()

    gpu_list = [int(g.strip()) for g in args.gpus.split(",") if g.strip()]
    # Dev always runs first: alpha is selected there and frozen for the others.
    requested = ["dev", "benchmark", "test"] if args.split == "all" else [args.split]
    splits = ["dev"] + [s for s in requested if s != "dev"]
    frozen_alpha: Dict[str, float] = {}
    tolerances = [15.0, 30.0, 45.0, 60.0, 90.0, 112.0, 120.0]

    # Load ensemble metrics to get tau_dev
    ensemble_meta = json.load(open(D / "final/cinet_v2_ensemble_metrics.json"))
    tau_dev = float(ensemble_meta["tau_dev"])
    print(f"[INIT] CINet_v2 Ensemble optimal threshold tau_dev: {tau_dev:.4f}")

    cache_path = PROJECT / args.cache
    cache_data = {}
    if cache_path.exists() and not args.force_recompute:
        print(f"[CACHE] Loading cached features and coordinates from: {cache_path}")
        cache_data = dict(np.load(cache_path, allow_pickle=True))

    full_results: Dict[str, Any] = {
        "title": "CINet_v2 Neural-Physics Dual-Guidance Object Localization System",
        "description": (
            "Sub-window continuous keypoint heatmap regression fused with satellite microphysical "
            "cold-top overshooting and rapid cooling centroids, with decaying anvil physical gating."
        ),
        "evaluation_protocol": {
            "primary_tolerance_km": args.tolerance,
            "matching": "greedy nearest-neighbor spatial bipartite matching (Haversine)",
            "dual_formula": "(y*, x*) = alpha * (y_nn, x_nn) + (1 - alpha) * (y_phys, x_phys)",
            "gating_formula": "Delta BT13 = BT13(T-10) - BT13(T-40) <= -1.0 K (reject if > -1.0 K)",
            "alpha_selection": "alpha swept on dev only, then frozen per peak mode for benchmark/test",
        },
        "splits": {},
    }

    t0_total = time.time()
    for sp in splits:
        print(f"\n{'='*80}\nPROCESSING SPLIT: {sp.upper()}\n{'='*80}")
        unique_uids, ts_data, active_meta, local_max_peaks, gt_map = extract_active_patches_meta(sp, tau_dev)

        # Check or compute coordinates
        cache_prefix = f"{sp}_"
        if f"{cache_prefix}nn" in cache_data and not args.force_recompute:
            feat_dict = {
                "geom": cache_data[f"{cache_prefix}geom"],
                "physics": cache_data[f"{cache_prefix}physics"],
                "cold": cache_data[f"{cache_prefix}cold"],
                "cool": cache_data[f"{cache_prefix}cool"],
                "nn": cache_data[f"{cache_prefix}nn"],
                "nn_max": cache_data[f"{cache_prefix}nn_max"],
                "d_min_bt13": cache_data[f"{cache_prefix}d_min_bt13"],
                "probs": cache_data[f"{cache_prefix}probs"],
            }
        else:
            feat_dict = extract_features_and_coordinates(sp, active_meta, gpu_list)
            for k, v in feat_dict.items():
                if isinstance(v, np.ndarray):
                    cache_data[f"{cache_prefix}{k}"] = v

        # Evaluate modes: local_max (primary) and all_active
        split_res: Dict[str, Any] = {
            "n_timestamps": len(unique_uids),
            "total_gt_objects": sum(len(gt_map[u]) for u in unique_uids),
            "total_active_windows": len(active_meta),
            "tau_dev": tau_dev,
            "modes": {},
        }

        for p_mode in ["local_max", "all_active"]:
            print(f"\n--- Evaluating Split: {sp.upper()} | Peak Mode: {p_mode} ---")

            # 1. Strategy A: Crude Window Geometric Center
            a_prim, a_curve = evaluate_strategy_predictions(
                active_meta, feat_dict["geom"], feat_dict["probs"], feat_dict["d_min_bt13"],
                unique_uids, gt_map, local_max_peaks, peak_mode=p_mode, tolerance_list=tolerances, primary_tolerance_km=args.tolerance
            )

            # 2. Strategy B: Pure Neural Attention Heatmap Peak
            b_prim, b_curve = evaluate_strategy_predictions(
                active_meta, feat_dict["nn"], feat_dict["probs"], feat_dict["d_min_bt13"],
                unique_uids, gt_map, local_max_peaks, peak_mode=p_mode, tolerance_list=tolerances, primary_tolerance_km=args.tolerance
            )

            # 3. Strategy C: Pure Satellite Microphysics Centroid
            c_prim, c_curve = evaluate_strategy_predictions(
                active_meta, feat_dict["physics"], feat_dict["probs"], feat_dict["d_min_bt13"],
                unique_uids, gt_map, local_max_peaks, peak_mode=p_mode, tolerance_list=tolerances, primary_tolerance_km=args.tolerance
            )

            # 4. Strategy D: dual guidance.
            # Alpha is swept on DEV ONLY and frozen per peak mode; benchmark and
            # test are scored once at that frozen value (EXPERIMENTS E14: the
            # original code swept alpha inside each evaluation split, i.e. it
            # tuned on the frozen test set).
            alpha_grid = np.linspace(0.0, 1.0, 11)
            alpha_sweep_results = []

            def _eval_alpha(a):
                c_dual = a * feat_dict["nn"] + (1.0 - a) * feat_dict["physics"]
                return evaluate_strategy_predictions(
                    active_meta, c_dual, feat_dict["probs"], feat_dict["d_min_bt13"],
                    unique_uids, gt_map, local_max_peaks, peak_mode=p_mode,
                    tolerance_list=tolerances, primary_tolerance_km=args.tolerance
                )

            if sp == "dev":
                best_alpha, best_alpha_csi = 0.5, -1.0
                best_dual_prim = None
                best_dual_curve = None
                for a in alpha_grid:
                    prim_d, curve_d = _eval_alpha(a)
                    alpha_sweep_results.append({
                        "alpha": round(float(a), 2),
                        "hits": prim_d["hits"],
                        "false_alarms": prim_d["false_alarms"],
                        "POD_obj": prim_d["POD_obj"],
                        "FAR_obj": prim_d["FAR_obj"],
                        "CSI_obj": prim_d["CSI_obj"],
                        "mean_distance_km": prim_d["mean_distance_km"],
                    })
                    if prim_d["CSI_obj"] > best_alpha_csi:
                        best_alpha_csi = prim_d["CSI_obj"]
                        best_alpha = float(a)
                        best_dual_prim = prim_d
                        best_dual_curve = curve_d
                frozen_alpha[p_mode] = float(best_alpha)
                print(f"[DEV] frozen alpha[{p_mode}] = {best_alpha:.2f} (dev CSI {best_alpha_csi:.4f})")
            else:
                best_alpha = frozen_alpha[p_mode]
                best_dual_prim, best_dual_curve = _eval_alpha(best_alpha)
                alpha_sweep_results.append({
                    "alpha": round(float(best_alpha), 2),
                    "frozen_from_dev": True,
                    "hits": best_dual_prim["hits"],
                    "false_alarms": best_dual_prim["false_alarms"],
                    "POD_obj": best_dual_prim["POD_obj"],
                    "FAR_obj": best_dual_prim["FAR_obj"],
                    "CSI_obj": best_dual_prim["CSI_obj"],
                    "mean_distance_km": best_dual_prim["mean_distance_km"],
                })

            # 5. Decaying Anvil Physical Gating Ablation on the best dual configuration
            c_opt = best_alpha * feat_dict["nn"] + (1.0 - best_alpha) * feat_dict["physics"]
            gated_prim, gated_curve = evaluate_strategy_predictions(
                active_meta, c_opt, feat_dict["probs"], feat_dict["d_min_bt13"],
                unique_uids, gt_map, local_max_peaks, peak_mode=p_mode,
                filter_decaying_anvil=True, decaying_threshold_k=-1.0,
                tolerance_list=tolerances, primary_tolerance_km=args.tolerance
            )

            # Performance delta analysis
            delta_strat_d = {
                "hits_gain": best_dual_prim["hits"] - a_prim["hits"],
                "hits_relative_pct": round(((best_dual_prim["hits"] - a_prim["hits"]) / max(1, a_prim["hits"])) * 100.0, 2),
                "POD_gain_points": round((best_dual_prim["POD_obj"] - a_prim["POD_obj"]) * 100.0, 3),
                "FAR_reduction_points": round((a_prim["FAR_obj"] - best_dual_prim["FAR_obj"]) * 100.0, 3),
                "CSI_gain_points": round((best_dual_prim["CSI_obj"] - a_prim["CSI_obj"]) * 100.0, 3),
                "CSI_relative_improvement_pct": round(((best_dual_prim["CSI_obj"] - a_prim["CSI_obj"]) / max(1e-5, a_prim["CSI_obj"])) * 100.0, 2),
                "distance_reduction_km": round(a_prim["mean_distance_km"] - best_dual_prim["mean_distance_km"], 2),
            }

            gating_impact = {
                "ungated": best_dual_prim,
                "gated": gated_prim,
                "fa_suppressed_count": best_dual_prim["false_alarms"] - gated_prim["false_alarms"],
                "fa_suppressed_pct": round(((best_dual_prim["false_alarms"] - gated_prim["false_alarms"]) / max(1, best_dual_prim["false_alarms"])) * 100.0, 2),
                "FAR_reduction_points": round((best_dual_prim["FAR_obj"] - gated_prim["FAR_obj"]) * 100.0, 3),
            }

            split_res["modes"][p_mode] = {
                "strategy_a_geom": {"primary_30km": a_prim, "tolerance_curve": a_curve},
                "strategy_b_heatmap": {"primary_30km": b_prim, "tolerance_curve": b_curve},
                "strategy_c_physics": {"primary_30km": c_prim, "tolerance_curve": c_curve},
                "strategy_d_dual_optimal": {
                    "optimal_alpha": best_alpha,
                    "primary_30km": best_dual_prim,
                    "tolerance_curve": best_dual_curve,
                    "improvement_over_geom": delta_strat_d,
                },
                "alpha_sweep": alpha_sweep_results,
                "decaying_anvil_ablation": gating_impact,
            }

        full_results["splits"][sp] = split_res

    full_results["alpha_selection"] = {
        "selected_on": "dev",
        "frozen_alpha_by_peak_mode": dict(frozen_alpha),
        "note": "benchmark and test are scored once at the dev-selected alpha; no sweep on evaluation splits",
    }

    # Save cache
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, **cache_data)
    print(f"\n[CACHE] Successfully cached all features and coordinates to {cache_path}")

    # Save metrics JSON
    metrics_path = PROJECT / args.out_metrics
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(full_results, indent=2))
    print(f"[METRICS] Successfully saved quantitative metrics to: {metrics_path}")

    # Generate publication figure
    fig_path = PROJECT / args.out_fig
    generate_benchmark_figure(full_results, fig_path)

    # Print comprehensive summary table
    print("\n" + "=" * 115)
    print("NEURAL-PHYSICS DUAL LOCALIZATION: COMPREHENSIVE BENCHMARK TABLE (30 km TOLERANCE)")
    print("=" * 115)
    print(
        f"{'Split':<10} | {'Mode':<10} | {'Strategy':<22} | {'Hits':<6} | {'Miss':<6} | {'FA':<6} | "
        f"{'POD_obj':<8} | {'FAR_obj':<8} | {'CSI_obj':<8} | {'MeanDist (km)':<13}"
    )
    print("-" * 115)

    for sp in splits:
        s_data = full_results["splits"][sp]
        for p_mode in ["local_max", "all_active"]:
            m_data = s_data["modes"][p_mode]
            strats = [
                ("Strat A: Geom Center", m_data["strategy_a_geom"]["primary_30km"]),
                ("Strat B: Pure Heatmap", m_data["strategy_b_heatmap"]["primary_30km"]),
                ("Strat C: Pure Physics", m_data["strategy_c_physics"]["primary_30km"]),
                (f"Strat D: Dual (a={m_data['strategy_d_dual_optimal']['optimal_alpha']:.2f})", m_data["strategy_d_dual_optimal"]["primary_30km"]),
                ("Strat D + Gating", m_data["decaying_anvil_ablation"]["gated"]),
            ]
            for s_name, p in strats:
                print(
                    f"{sp:<10} | {p_mode:<10} | {s_name:<22} | {p['hits']:<6} | {p['misses']:<6} | {p['false_alarms']:<6} | "
                    f"{p['POD_obj']*100:6.2f}%  | {p['FAR_obj']*100:6.2f}%  | {p['CSI_obj']*100:6.2f}%  | {p['mean_distance_km']:8.2f} km"
                )
            imp = m_data["strategy_d_dual_optimal"]["improvement_over_geom"]
            gat = m_data["decaying_anvil_ablation"]
            print(
                f"  ==> [GAIN Strat D vs A]: Hits +{imp['hits_gain']} (+{imp['hits_relative_pct']}%), "
                f"CSI +{imp['CSI_gain_points']} pts (+{imp['CSI_relative_improvement_pct']}%), "
                f"Mean Dist -{imp['distance_reduction_km']} km | [Gating FA Drop]: -{gat['fa_suppressed_count']} (-{gat['fa_suppressed_pct']}%)"
            )
            print("-" * 115)

    print(f"\nAll operations completed successfully in {time.time() - t0_total:.1f} s.")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
