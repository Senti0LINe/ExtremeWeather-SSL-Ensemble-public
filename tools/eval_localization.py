"""Object Localization Evaluation System for ExtremeWeather CI Nowcasting.

Evaluates 5-seed TTA CNN + GBDT ensemble predictions against ground truth CI objects
using spatial bipartite matching (Haversine distance <= 30 km operational tolerance)
on the official Benchmark (30 timestamps, 9,180 windows) and Held-out Test Set (200 timestamps, 61,200 windows).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
from scipy.ndimage import maximum_filter

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

FULL = Path("/data1/group3/DATASAT/experiments/ci_det_v2/annotations_full")
D = PROJECT / "experiments"

LAT0, LON0 = 40.00448, 103.99570
H, W, S = 2001, 2101, 224
GRID_STEP = 112
EARTH_RADIUS_KM = 6371.0


def get_grid_coords() -> Tuple[List[int], List[int]]:
    """Return sorted unique row and column start positions for the 17x18 grid."""
    rows = sorted(set(list(range(0, H - S + 1, GRID_STEP)) + [H - S]))
    cols = sorted(set(list(range(0, W - S + 1, GRID_STEP)) + [W - S]))
    return rows, cols


def haversine_km(
    lat1: float | np.ndarray,
    lon1: float | np.ndarray,
    lat2: float | np.ndarray,
    lon2: float | np.ndarray,
) -> float | np.ndarray:
    """Calculate Great Circle distance in kilometers between two WGS84 points."""
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2.0) ** 2
    return EARTH_RADIUS_KM * 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))


def bbox_to_latlon(bbox_xyxy: List[float]) -> Tuple[float, float]:
    """Convert full-grid bounding box [x1, y1, x2, y2] to geographic (lat, lon)."""
    x1, y1, x2, y2 = bbox_xyxy
    xc = (x1 + x2) / 2.0
    yc = (y1 + y2) / 2.0
    lat = LAT0 - 0.01 * yc
    lon = LON0 + 0.01 * xc
    return float(lat), float(lon)


def load_ground_truth_objects(role: str, uids: set[str]) -> Dict[str, List[dict]]:
    """Extract ground truth CI convective cell objects from samples.jsonl."""
    path = FULL / f"{role}_samples.jsonl"
    out: Dict[str, List[dict]] = {u: [] for u in uids}
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            rec = json.loads(line)
            uid = rec["sample_uid"]
            if uid in uids:
                objs = []
                for o in rec.get("objects", []):
                    # Check CI category or positive CI label
                    if (
                        o.get("category", "") == "CI"
                        or o.get("ci_label", 0) == 1
                        or o.get("label", 0) == 1
                    ):
                        lat, lon = bbox_to_latlon(o["bbox_xyxy"])
                        x1, y1, x2, y2 = o["bbox_xyxy"]
                        area = max(1, (x2 - x1) * (y2 - y1))
                        objs.append(
                            {
                                "object_uid": o["object_uid"],
                                "event_id": o.get("event_id", ""),
                                "bbox_xyxy": o["bbox_xyxy"],
                                "area_px": area,
                                "lat": lat,
                                "lon": lon,
                            }
                        )
                out[uid] = objs
    return out


def load_calibrated_predictions(
    split: str,
) -> Tuple[List[str], List[Tuple[int, int]], np.ndarray, float, float]:
    """Load 5-seed TTA CNN + GBDT ensemble predictions and calibrate on dev.

    Returns:
        uids: list of timestamp uids
        coords: list of (row, col) for each window
        probs_cal: calibrated probability array
        w: CNN ensemble weight
        tau: optimal decision threshold
    """
    calib_file = D / "final/calib.json"
    if not calib_file.exists():
        raise FileNotFoundError(f"Calibration file {calib_file} not found.")
    calib = json.loads(calib_file.read_text())
    w = calib.get("w", 0.9)
    tau = calib.get("tau", 0.26050420168067234)

    gbdt_cache = D / "final/gbdt_cache.pkl"
    if not gbdt_cache.exists():
        raise FileNotFoundError(f"GBDT cache {gbdt_cache} not found.")
    gb_models = joblib.load(gbdt_cache)

    if split == "benchmark":
        f_feat = np.load(D / "p0_gbdt/features_val.npz", allow_pickle=True)
        keys = [(str(u), int(r), int(c)) for u, r, c in zip(f_feat["uid"], f_feat["row"], f_feat["col"])]

        cnn_preds = []
        for s in range(5):
            npz_path = D / f"p1_seed{s}_bench_tta.npz"
            z = np.load(npz_path, allow_pickle=True)
            pos = {(str(u), int(r), int(c)): i for i, (u, r, c) in enumerate(zip(z["uid"], z["row"], z["col"]))}
            cnn_preds.append(z["p"][np.array([pos[k] for k in keys])])
        p_cnn = np.mean(cnn_preds, axis=0)
        p_gbdt = np.mean([m.predict_proba(f_feat["X"])[:, 1] for m in gb_models], axis=0)

    elif split == "test":
        f_feat = np.load(D / "final/features_test_eval.npz", allow_pickle=True)
        keys = [(str(u), int(r), int(c)) for u, r, c in zip(f_feat["uid"], f_feat["row"], f_feat["col"])]

        cnn_preds = []
        for s in range(5):
            npz_path = D / f"final/p1_seed{s}_test_tta.npz"
            z = np.load(npz_path, allow_pickle=True)
            pos = {(str(u), int(r), int(c)): i for i, (u, r, c) in enumerate(zip(z["uid"], z["row"], z["col"]))}
            cnn_preds.append(z["p"][np.array([pos[k] for k in keys])])
        p_cnn = np.mean(cnn_preds, axis=0)
        p_gbdt = np.mean([m.predict_proba(f_feat["X"])[:, 1] for m in gb_models], axis=0)
    else:
        raise ValueError(f"Unknown split: {split}")

    p_raw = w * p_cnn + (1.0 - w) * p_gbdt
    p_cal = np.interp(p_raw, calib["x"], calib["y"])

    uids = [k[0] for k in keys]
    coords = [(k[1], k[2]) for k in keys]
    return uids, coords, p_cal, w, tau


def extract_peaks_from_timestamp(
    p_grid: np.ndarray,
    rows: List[int],
    cols: List[int],
    tau: float,
    peak_mode: str = "local_max",
) -> List[Dict[str, Any]]:
    """Detect local peaks or active windows on the 17x18 probability grid."""
    peaks = []
    n_rows, n_cols = len(rows), len(cols)

    if peak_mode == "local_max":
        # 3x3 maximum filter for spatial non-maximum suppression
        mf = maximum_filter(p_grid, size=3, mode="constant", cval=0.0)
        is_peak = (p_grid == mf) & (p_grid >= tau)

        # Handle flat plateaus: if adjacent cells have identical max, suppress duplicates
        labeled_peaks = []
        visited = set()
        for i in range(n_rows):
            for j in range(n_cols):
                if is_peak[i, j] and (i, j) not in visited:
                    # Check if any neighbor in 3x3 has already been added with equal value
                    already_represented = False
                    for di in (-1, 0, 1):
                        for dj in (-1, 0, 1):
                            ni, nj = i + di, j + dj
                            if (ni, nj) in visited and p_grid[ni, nj] == p_grid[i, j]:
                                already_represented = True
                                break
                        if already_represented:
                            break
                    if not already_represented:
                        visited.add((i, j))
                        r_center = rows[i] + S // 2
                        c_center = cols[j] + S // 2
                        lat = LAT0 - 0.01 * r_center
                        lon = LON0 + 0.01 * c_center
                        labeled_peaks.append(
                            {
                                "lat": float(lat),
                                "lon": float(lon),
                                "prob": float(p_grid[i, j]),
                                "row": rows[i],
                                "col": cols[j],
                                "grid_idx": (i, j),
                            }
                        )
        peaks = labeled_peaks

    elif peak_mode == "all_active":
        # Every window above tau is evaluated
        for i in range(n_rows):
            for j in range(n_cols):
                if p_grid[i, j] >= tau:
                    r_center = rows[i] + S // 2
                    c_center = cols[j] + S // 2
                    lat = LAT0 - 0.01 * r_center
                    lon = LON0 + 0.01 * c_center
                    peaks.append(
                        {
                            "lat": float(lat),
                            "lon": float(lon),
                            "prob": float(p_grid[i, j]),
                            "row": rows[i],
                            "col": cols[j],
                            "grid_idx": (i, j),
                        }
                    )
    else:
        raise ValueError(f"Unknown peak_mode: {peak_mode}")

    return peaks


def spatial_bipartite_matching(
    pred_peaks: List[Dict[str, Any]],
    true_objects: List[Dict[str, Any]],
    tolerance_km: float = 30.0,
) -> Tuple[int, int, int, List[float], List[Dict[str, Any]]]:
    """Perform greedy nearest-neighbor bipartite matching between predicted peaks and true CI objects.

    Returns:
        hits: Number of matched pairs within tolerance_km
        misses: Number of unmatched ground truth objects
        false_alarms: Number of unmatched predicted peaks
        distances: List of matching distances (km) for all Hits
        matched_pairs: List of dicts describing each matched hit
    """
    m = len(pred_peaks)
    n = len(true_objects)

    if m == 0 and n == 0:
        return 0, 0, 0, [], []
    if m == 0:
        return 0, n, 0, [], []
    if n == 0:
        return 0, 0, m, [], []

    # Compute candidate pairs within tolerance
    candidates = []
    for i, p in enumerate(pred_peaks):
        for j, t in enumerate(true_objects):
            d = haversine_km(p["lat"], p["lon"], t["lat"], t["lon"])
            if d <= tolerance_km:
                candidates.append((float(d), i, j))

    # Sort ascending by distance (nearest-neighbor greedy matching)
    candidates.sort(key=lambda x: x[0])

    matched_p = set()
    matched_t = set()
    matched_distances = []
    matched_details = []

    for d, i, j in candidates:
        if i not in matched_p and j not in matched_t:
            matched_p.add(i)
            matched_t.add(j)
            matched_distances.append(d)
            matched_details.append(
                {
                    "distance_km": round(d, 2),
                    "pred_lat": pred_peaks[i]["lat"],
                    "pred_lon": pred_peaks[i]["lon"],
                    "pred_prob": pred_peaks[i]["prob"],
                    "true_lat": true_objects[j]["lat"],
                    "true_lon": true_objects[j]["lon"],
                    "true_uid": true_objects[j]["object_uid"],
                }
            )

    hits = len(matched_p)
    misses = n - hits
    false_alarms = m - hits

    return hits, misses, false_alarms, matched_distances, matched_details


def evaluate_split(
    split: str,
    peak_mode: str = "local_max",
    primary_tolerance_km: float = 30.0,
    tolerance_list: List[float] = (15.0, 30.0, 45.0, 60.0, 90.0, 112.0, 120.0),
) -> Dict[str, Any]:
    """Run object-level localization evaluation for a given dataset split."""
    print(f"\n{'='*70}")
    print(f"Evaluating Split: {split.upper()} (Peak Mode: {peak_mode})")
    print(f"{'='*70}")

    uids, coords, probs, w, tau = load_calibrated_predictions(split)
    unique_uids = sorted(list(dict.fromkeys(uids)))
    rows, cols = get_grid_coords()
    n_rows, n_cols = len(rows), len(cols)

    # Load ground truth objects
    role = "val" if split == "benchmark" else "test"
    gt_map = load_ground_truth_objects(role, set(unique_uids))

    # Group predictions by timestamp
    ts_data: Dict[str, Dict[Tuple[int, int], float]] = {u: {} for u in unique_uids}
    for u, (r, c), p in zip(uids, coords, probs):
        ts_data[u][(r, c)] = p

    # Pre-extract peaks for each timestamp
    ts_peaks: Dict[str, List[Dict[str, Any]]] = {}
    for uid in unique_uids:
        grid = np.zeros((n_rows, n_cols), dtype=np.float32)
        for ri, r in enumerate(rows):
            for cj, c in enumerate(cols):
                grid[ri, cj] = ts_data[uid].get((r, c), 0.0)
        ts_peaks[uid] = extract_peaks_from_timestamp(grid, rows, cols, tau, peak_mode)

    total_gt_objects = sum(len(gt_map[u]) for u in unique_uids)
    total_pred_peaks = sum(len(ts_peaks[u]) for u in unique_uids)
    print(f"Timestamps: {len(unique_uids)}")
    print(f"Total True CI Objects: {total_gt_objects}")
    print(f"Total Predicted Peaks (p >= {tau:.4f}): {total_pred_peaks}")

    # Evaluate across tolerance list
    curve_results = {}
    primary_result = {}

    for tol in tolerance_list:
        tot_hits = 0
        tot_miss = 0
        tot_fa = 0
        all_dists: List[float] = []

        for uid in unique_uids:
            h, m, fa, dists, _ = spatial_bipartite_matching(
                ts_peaks[uid], gt_map[uid], tolerance_km=tol
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

        metrics = {
            "tolerance_km": float(tol),
            "hits": int(tot_hits),
            "misses": int(tot_miss),
            "false_alarms": int(tot_fa),
            "total_gt": int(total_gt_objects),
            "total_peaks": int(total_pred_peaks),
            "POD_obj": float(pod),
            "FAR_obj": float(far),
            "CSI_obj": float(csi),
            "mean_distance_km": mean_dist,
            "median_distance_km": median_dist,
            "p25_distance_km": p25_dist,
            "p75_distance_km": p75_dist,
        }
        curve_results[f"{tol:.1f}km"] = metrics

        if abs(tol - primary_tolerance_km) < 1e-4:
            primary_result = metrics

    # Format output dictionary
    return {
        "split": split,
        "peak_mode": peak_mode,
        "n_timestamps": len(unique_uids),
        "total_gt_objects": total_gt_objects,
        "total_pred_peaks": total_pred_peaks,
        "ensemble_cnn_weight": float(w),
        "tau": float(tau),
        "primary_metric_30km": primary_result,
        "tolerance_sensitivity_curve": curve_results,
    }


def main():
    parser = argparse.ArgumentParser(description="Object-level CI Localization Evaluation.")
    parser.add_argument("--split", choices=["benchmark", "test", "all"], default="all",
                        help="Dataset split to evaluate (default: all)")
    parser.add_argument("--peak-mode", choices=["local_max", "all_active"], default="local_max",
                        help="Peak detection mode: local_max (3x3 filter) or all_active (all p >= tau)")
    parser.add_argument("--tolerance", type=float, default=30.0,
                        help="Primary spatial matching tolerance in km (default: 30.0)")
    parser.add_argument("--out", default="experiments/final/localization_metrics.json",
                        help="Output path for evaluation JSON metrics")
    args = parser.parse_args()

    tolerances = [15.0, 30.0, 45.0, 60.0, 90.0, 112.0, 120.0]

    splits_to_run = ["benchmark", "test"] if args.split == "all" else [args.split]
    results: Dict[str, Any] = {
        "description": "ExtremeWeather CI Object-Level Localization Verification",
        "spatial_reference": {"LAT0": LAT0, "LON0": LON0, "resolution_deg": 0.01, "approx_px_km": 1.11},
        "evaluation_protocol": {
            "primary_tolerance_km": args.tolerance,
            "peak_mode": args.peak_mode,
            "matching": "greedy nearest-neighbor spatial bipartite matching",
        },
        "results": {},
    }

    for sp in splits_to_run:
        res = evaluate_split(
            split=sp,
            peak_mode=args.peak_mode,
            primary_tolerance_km=args.tolerance,
            tolerance_list=tolerances,
        )
        results["results"][sp] = res

        # Also run sensitivity with all_active if running local_max to give complete diagnostic view
        if args.peak_mode == "local_max":
            res_active = evaluate_split(
                split=sp,
                peak_mode="all_active",
                primary_tolerance_km=args.tolerance,
                tolerance_list=tolerances,
            )
            results["results"][f"{sp}_all_active"] = res_active

    out_path = PROJECT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\n[SUCCESS] Quantitative localization metrics saved to: {out_path}")

    # Print summary table
    print("\n" + "=" * 95)
    print("OBJECT-LEVEL LOCALIZATION VERIFICATION SUMMARY (PRIMARY: 30 km)")
    print("=" * 95)
    print(f"{'Split':<12} | {'Mode':<10} | {'Hits':<6} | {'Miss':<6} | {'FA':<6} | {'POD_obj':<8} | {'FAR_obj':<8} | {'CSI_obj':<8} | {'MeanDist (km)':<13}")
    print("-" * 95)
    for key, val in results["results"].items():
        p = val["primary_metric_30km"]
        mode_str = val["peak_mode"]
        print(f"{val['split']:<12} | {mode_str:<10} | {p['hits']:<6} | {p['misses']:<6} | {p['false_alarms']:<6} | {p['POD_obj']*100:6.2f}%  | {p['FAR_obj']*100:6.2f}%  | {p['CSI_obj']*100:6.2f}%  | {p['mean_distance_km']:8.2f} km")
    print("=" * 95)


if __name__ == "__main__":
    main()
