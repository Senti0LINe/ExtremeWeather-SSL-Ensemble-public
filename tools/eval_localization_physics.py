"""Physics Centroiding Object Localization Evaluation for ExtremeWeather CI.

Replaces crude window geometric center (row + 112, col + 112) with sub-window
physics-guided convective centroid (y_phys, x_phys) derived from:
1. BT13(T-10) minimum temperature overshooting core (y_cold, x_cold);
2. 30-min cloud-top cooling rate field Delta BT13 = BT13(T-40) - BT13(T-10)
   weighted geometric centroid (y_cool, x_cool) of the top 10% cooling pixels;
3. Physical blend of overshooting core and rapid cooling centroid.

Evaluates on the official Benchmark (30 timestamps) and Held-out Test Set (200 timestamps)
with 30 km operational tolerance spatial bipartite matching.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from scipy.ndimage import maximum_filter

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from ewci import data
import tools.eval_localization as el

FULL = Path("/data1/group3/DATASAT/experiments/ci_det_v2/annotations_full")
D = PROJECT / "experiments"

LAT0, LON0 = el.LAT0, el.LON0
H, W, S = el.H, el.W, el.S
GRID_STEP = el.GRID_STEP


def compute_patch_physics_coords(
    f0_b13: np.ndarray,
    f3_b13: np.ndarray,
    row: int,
    col: int,
) -> Dict[str, Tuple[float, float]]:
    """Compute sub-window physics centroid coordinates for a 224x224 patch.

    Args:
        f0_b13: Full grid B13 frame at T-40 (int16).
        f3_b13: Full grid B13 frame at T-10 (int16).
        row: Top-left row index in full grid.
        col: Top-left col index in full grid.

    Returns:
        Dict mapping centroid mode to (y_sub, x_sub) within [0, 224).
    """
    b0_patch = f0_b13[row : row + S, col : col + S]
    b3_patch = f3_b13[row : row + S, col : col + S]

    # Convert to float Kelvin, masking invalid fill values
    fin3 = b3_patch != data.FILL_VALUE
    fin0 = b0_patch != data.FILL_VALUE
    b3_k = np.where(fin3, b3_patch.astype(np.float32) * data.SCALE, np.nan)
    b0_k = np.where(fin0, b0_patch.astype(np.float32) * data.SCALE, np.nan)

    # 1. Geometric center (baseline)
    c_geom = (112.0, 112.0)

    # 2. Coldest pixel (T-10 overshooting core)
    if fin3.any():
        yc, xc = np.unravel_index(np.nanargmin(b3_k), b3_k.shape)
        c_cold = (float(yc), float(xc))
    else:
        c_cold = c_geom

    # 3. Cooling rate field: Delta BT13 = BT13(T-40) - BT13(T-10) (cooling > 0)
    fin_both = fin0 & fin3
    cooling = np.where(fin_both, b0_k - b3_k, np.nan)
    c_pos = cooling[fin_both & (cooling > 0)]

    if c_pos.size > 0:
        q90 = np.percentile(c_pos, 90)
        mask = fin_both & (cooling >= q90) & (cooling > 0)
        ys, xs = np.nonzero(mask)
        ws = cooling[mask]
        w_sum = ws.sum()
        if w_sum > 0:
            c_cool = (float((ys * ws).sum() / w_sum), float((xs * ws).sum() / w_sum))
        else:
            c_cool = c_cold
    else:
        c_cool = c_cold

    # 4. Physics Blend: 50% Cold Core + 50% Cooling Updraft Centroid
    c_blend = (0.5 * (c_cold[0] + c_cool[0]), 0.5 * (c_cold[1] + c_cool[1]))

    return {
        "geom": c_geom,
        "cold": c_cold,
        "cool": c_cool,
        "physics": c_blend,
    }


def extract_predictions_for_split(
    split: str,
    role: str,
    uids: List[str],
    coords: List[Tuple[int, int]],
    probs: np.ndarray,
    tau: float,
    uid2obs: Dict[str, int],
    reader: data.ObsReader,
) -> Dict[str, Dict[str, Dict[str, List[Dict[str, Any]]]]]:
    """Extract geometric and physics predictions for all timestamps and peak modes.

    Returns:
        nested dict: [peak_mode][method][uid] -> list of prediction dicts
    """
    rows, cols = el.get_grid_coords()
    n_rows, n_cols = len(rows), len(cols)
    unique_uids = sorted(list(dict.fromkeys(uids)))

    # Map probability grid per timestamp
    ts_data: Dict[str, Dict[Tuple[int, int], float]] = {u: {} for u in unique_uids}
    for u, (r, c), p in zip(uids, coords, probs):
        ts_data[u][(r, c)] = float(p)

    # Pre-extract patch physics coordinates by reading full frames once per timestamp
    methods = ["geom", "cold", "cool", "physics"]
    ts_all_preds: Dict[str, Dict[str, List[Dict[str, Any]]]] = {
        m: {u: [] for u in unique_uids} for m in methods
    }

    b13_idx = data.STORE_BANDS.index("B13")
    t0 = time.time()
    n_total_active = 0

    print(f"[{split.upper()}] Computing physics centroids across {len(unique_uids)} timestamps...")
    for idx, uid in enumerate(unique_uids):
        # Active windows in this timestamp
        active_rcp = [
            (r, c, p) for (r, c), p in ts_data[uid].items() if p >= tau
        ]
        if not active_rcp:
            continue

        n_total_active += len(active_rcp)
        oi = uid2obs[uid]
        # Read full B13 frames: T-40 and T-10
        f0 = reader.obs[oi - 4, b13_idx]
        f3 = reader.obs[oi - 1, b13_idx]

        for r, c, p in active_rcp:
            c_dict = compute_patch_physics_coords(f0, f3, r, c)
            for m in methods:
                yr, xr = c_dict[m]
                lat = el.LAT0 - 0.01 * (r + yr)
                lon = el.LON0 + 0.01 * (c + xr)
                ts_all_preds[m][uid].append(
                    {
                        "lat": float(lat),
                        "lon": float(lon),
                        "prob": float(p),
                        "row": r,
                        "col": c,
                        "y_sub": float(yr),
                        "x_sub": float(xr),
                    }
                )

        if (idx + 1) % 50 == 0 or (idx + 1) == len(unique_uids):
            print(f"  Processed {idx + 1}/{len(unique_uids)} timestamps ({time.time() - t0:.1f}s)")

    print(f"[{split.upper()}] Total active windows (p >= {tau:.4f}): {n_total_active}")

    # Now filter by peak_mode: "local_max" and "all_active"
    out: Dict[str, Dict[str, Dict[str, List[Dict[str, Any]]]]] = {
        "local_max": {m: {u: [] for u in unique_uids} for m in methods},
        "all_active": {m: {u: [] for u in unique_uids} for m in methods},
    }

    for uid in unique_uids:
        grid = np.zeros((n_rows, n_cols), dtype=np.float32)
        for ri, r in enumerate(rows):
            for cj, c in enumerate(cols):
                grid[ri, cj] = ts_data[uid].get((r, c), 0.0)

        # 1. all_active: all active windows
        for m in methods:
            out["all_active"][m][uid] = list(ts_all_preds[m][uid])

        # 2. local_max: 3x3 maximum filter
        mf = maximum_filter(grid, size=3, mode="constant", cval=0.0)
        is_peak = (grid == mf) & (grid >= tau)

        visited = set()
        active_peaks: List[Tuple[int, int]] = []
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
                        active_peaks.append((rows[i], cols[j]))

        peak_set = set(active_peaks)
        for m in methods:
            out["local_max"][m][uid] = [
                pred for pred in ts_all_preds[m][uid] if (pred["row"], pred["col"]) in peak_set
            ]

    return out


def evaluate_metrics(
    ts_preds: Dict[str, List[Dict[str, Any]]],
    gt_map: Dict[str, List[Dict[str, Any]]],
    unique_uids: List[str],
    tolerance_list: List[float],
    primary_tolerance_km: float = 30.0,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Compute bipartite matching metrics across multiple tolerance radii."""
    curve = {}
    primary = {}
    total_gt = sum(len(gt_map[u]) for u in unique_uids)
    total_preds = sum(len(ts_preds[u]) for u in unique_uids)

    for tol in tolerance_list:
        tot_hits = 0
        tot_miss = 0
        tot_fa = 0
        all_dists: List[float] = []

        for uid in unique_uids:
            h, m, fa, dists, _ = el.spatial_bipartite_matching(
                ts_preds[uid], gt_map[uid], tolerance_km=tol
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
        curve[f"{tol:.1f}km"] = res
        if abs(tol - primary_tolerance_km) < 1e-4:
            primary = res

    return primary, curve


def run_split_evaluation(
    split: str,
    primary_tolerance_km: float = 30.0,
    tolerance_list: List[float] = (15.0, 30.0, 45.0, 60.0, 90.0, 112.0, 120.0),
) -> Dict[str, Any]:
    """Run full object-level localization evaluation comparing geometric vs physics centroiding."""
    role = "val" if split == "benchmark" else "test"
    uids, coords, probs, w, tau = el.load_calibrated_predictions(split)
    unique_uids = sorted(list(dict.fromkeys(uids)))

    gt_map = el.load_ground_truth_objects(role, set(unique_uids))
    total_gt = sum(len(gt_map[u]) for u in unique_uids)

    # Load obs_index mapping
    uid2obs = {}
    with open(FULL / f"{role}_samples.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                s = json.loads(line)
                uid2obs[s["sample_uid"]] = int(s["obs_index"])

    reader = data.ObsReader(role)

    # Extract all predictions
    pred_hierarchy = extract_predictions_for_split(
        split=split,
        role=role,
        uids=uids,
        coords=coords,
        probs=probs,
        tau=tau,
        uid2obs=uid2obs,
        reader=reader,
    )

    split_res: Dict[str, Any] = {
        "split": split,
        "n_timestamps": len(unique_uids),
        "total_gt_objects": total_gt,
        "ensemble_cnn_weight": float(w),
        "tau": float(tau),
        "modes": {},
    }

    for p_mode in ["local_max", "all_active"]:
        mode_data: Dict[str, Any] = {}
        for method in ["geom", "cold", "cool", "physics"]:
            primary, curve = evaluate_metrics(
                ts_preds=pred_hierarchy[p_mode][method],
                gt_map=gt_map,
                unique_uids=unique_uids,
                tolerance_list=tolerance_list,
                primary_tolerance_km=primary_tolerance_km,
            )
            mode_data[method] = {
                "primary_metric_30km": primary,
                "tolerance_sensitivity_curve": curve,
            }

        # Calculate deltas (physics vs geometric center)
        geom_p = mode_data["geom"]["primary_metric_30km"]
        phys_p = mode_data["physics"]["primary_metric_30km"]
        cold_p = mode_data["cold"]["primary_metric_30km"]
        cool_p = mode_data["cool"]["primary_metric_30km"]

        delta_info = {
            "geom_hits": geom_p["hits"],
            "physics_hits": phys_p["hits"],
            "hits_increase": phys_p["hits"] - geom_p["hits"],
            "hits_relative_pct": round(
                ((phys_p["hits"] - geom_p["hits"]) / max(1, geom_p["hits"])) * 100.0, 2
            ),
            "geom_POD_obj": geom_p["POD_obj"],
            "physics_POD_obj": phys_p["POD_obj"],
            "POD_obj_gain_pct_points": round((phys_p["POD_obj"] - geom_p["POD_obj"]) * 100.0, 3),
            "geom_CSI_obj": geom_p["CSI_obj"],
            "physics_CSI_obj": phys_p["CSI_obj"],
            "CSI_obj_gain_pct_points": round((phys_p["CSI_obj"] - geom_p["CSI_obj"]) * 100.0, 3),
            "geom_mean_distance_km": geom_p["mean_distance_km"],
            "physics_mean_distance_km": phys_p["mean_distance_km"],
            "distance_reduction_km": round(
                geom_p["mean_distance_km"] - phys_p["mean_distance_km"], 2
            ),
        }

        mode_data["comparison_summary"] = delta_info
        split_res["modes"][p_mode] = mode_data

    return split_res


def main():
    parser = argparse.ArgumentParser(
        description="Physics Centroiding Engine: Object-Level CI Localization."
    )
    parser.add_argument(
        "--split",
        choices=["benchmark", "test", "all"],
        default="all",
        help="Dataset split to evaluate (default: all)",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=30.0,
        help="Primary spatial matching tolerance in km (default: 30.0)",
    )
    parser.add_argument(
        "--out",
        default="experiments/final/localization_physics_metrics.json",
        help="Output path for evaluation JSON metrics",
    )
    args = parser.parse_args()

    tolerances = [15.0, 30.0, 45.0, 60.0, 90.0, 112.0, 120.0]
    splits = ["benchmark", "test"] if args.split == "all" else [args.split]

    full_results: Dict[str, Any] = {
        "title": "ExtremeWeather CI Physics Centroiding Localization Verification",
        "description": (
            "Comparative evaluation of satellite microphysical centroiding (cold-top overshooting "
            "+ cloud-top cooling rate field) versus crude window geometric center."
        ),
        "spatial_reference": {
            "LAT0": LAT0,
            "LON0": LON0,
            "resolution_deg": 0.01,
            "approx_px_km": 1.11,
        },
        "evaluation_protocol": {
            "primary_tolerance_km": args.tolerance,
            "matching": "greedy nearest-neighbor spatial bipartite matching (Haversine)",
            "physics_centroid_formula": (
                "0.5 * (y_cold, x_cold) + 0.5 * (y_cool, x_cool) where cold is BT13(T-10) "
                "minimum pixel and cool is weighted centroid of top 10% cooling pixels "
                "Delta BT13 = BT13(T-40) - BT13(T-10)."
            ),
        },
        "splits": {},
    }

    t0_total = time.time()
    for sp in splits:
        print(f"\n{'='*80}\nRUNNING EVALUATION ON SPLIT: {sp.upper()}\n{'='*80}")
        res = run_split_evaluation(
            split=sp,
            primary_tolerance_km=args.tolerance,
            tolerance_list=tolerances,
        )
        full_results["splits"][sp] = res

    out_path = PROJECT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(full_results, indent=2))
    print(f"\n[SUCCESS] Quantitative physics localization metrics saved to: {out_path}")

    # Print comprehensive comparison summary tables
    print("\n" + "=" * 110)
    print("PHYSICS CENTROIDING LOCALIZATION VERIFICATION: QUANTITATIVE BENCHMARK (30 km TOLERANCE)")
    print("=" * 110)
    print(
        f"{'Split':<10} | {'Mode':<10} | {'Method':<10} | {'Hits':<6} | {'Miss':<6} | {'FA':<6} | "
        f"{'POD_obj':<8} | {'FAR_obj':<8} | {'CSI_obj':<8} | {'MeanDist (km)':<13}"
    )
    print("-" * 110)

    for sp, s_data in full_results["splits"].items():
        for p_mode, m_data in s_data["modes"].items():
            for meth in ["geom", "cold", "cool", "physics"]:
                p = m_data[meth]["primary_metric_30km"]
                meth_name = "Geom (Old)" if meth == "geom" else ("Phys (New)" if meth == "physics" else meth)
                print(
                    f"{sp:<10} | {p_mode:<10} | {meth_name:<10} | {p['hits']:<6} | {p['misses']:<6} | {p['false_alarms']:<6} | "
                    f"{p['POD_obj']*100:6.2f}%  | {p['FAR_obj']*100:6.2f}%  | {p['CSI_obj']*100:6.2f}%  | {p['mean_distance_km']:8.2f} km"
                )
            # Print delta line
            cmp = m_data["comparison_summary"]
            print(
                f"   ==> [IMPROVEMENT: {sp}/{p_mode}] Hits: {cmp['geom_hits']} -> {cmp['physics_hits']} "
                f"(+{cmp['hits_relative_pct']}%), POD: {cmp['geom_POD_obj']*100:.2f}% -> {cmp['physics_POD_obj']*100:.2f}% "
                f"(+{cmp['POD_obj_gain_pct_points']} pts), CSI: {cmp['geom_CSI_obj']*100:.2f}% -> {cmp['physics_CSI_obj']*100:.2f}% "
                f"(+{cmp['CSI_obj_gain_pct_points']} pts), Dist Reduc: {cmp['distance_reduction_km']} km"
            )
            print("-" * 110)

    print(f"\nAll evaluations completed successfully in {time.time() - t0_total:.1f} s.")


if __name__ == "__main__":
    main()
