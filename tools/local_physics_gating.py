"""Object-Centric Local Microphysics Gating & Full-Domain Benchmark Evaluation (tools/local_physics_gating.py).

Functions:
1. Implements targeted decaying-anvil microphysics gating on dense candidate peaks:
   - Condition: Delta_BT13_local > -0.5 K AND (BT10 - BT13)_local < 0 K -> suppressed as decaying anvil.
2. Computes strict 30 km bipartite matching across Benchmark (463 CI objects) and Held-out Test (6,806 CI objects).
3. Cross-compares:
   - Coarse grid 112-stride baseline
   - Dense continuous 56-stride perception (pre-gating vs post-gating)
4. Saves comprehensive metrics to experiments/final/dense_perception_metrics.json.
5. Renders native full-domain visualization to reports/figures/full_domain_dense_perception.png.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from ewci import data
import tools.eval_localization as el

FULL = Path("/data1/group3/DATASAT/experiments/ci_det_v2/annotations_full")
D = PROJECT / "experiments"
REPORTS_FIG = PROJECT / "reports/figures"

LAT0, LON0 = 40.00448, 103.99570
H, W = 2001, 2101
TOLERANCE_LIST = [15.0, 30.0, 45.0, 60.0, 90.0, 112.0, 120.0]


def evaluate_candidate_dict(
    candidates_by_uid: Dict[str, List[dict]],
    gt_map: Dict[str, List[dict]],
    filter_decaying: bool = False,
    decay_delta_thr: float = -0.5,
    coord_key: str = "dual",  # "dual", "nn", "phys"
    tolerances: List[float] = TOLERANCE_LIST,
    primary_tol: float = 30.0,
) -> Tuple[Dict[str, Any], Dict[str, Any], int, int]:
    """Run spatial bipartite matching on candidate predictions."""
    uids = sorted(list(candidates_by_uid.keys()))
    preds_by_uid = {u: [] for u in uids}

    total_input_candidates = 0
    suppressed_count = 0

    for u in uids:
        for p in candidates_by_uid[u]:
            total_input_candidates += 1
            if filter_decaying:
                d_cooling = p.get("delta_bt13_local", 0.0)
                b10_b13 = p.get("b10_b13_local", -99.0)
                # Targeted decaying anvil rule: warming/stagnant AND no overshooting
                if d_cooling > decay_delta_thr and b10_b13 < 0.0:
                    suppressed_count += 1
                    continue

            # Coordinate selection. Each branch recomputes lat/lon from its own
            # pixel coordinates so the three ablation arms are genuinely
            # different (E14 P7: the old `else` branch was a copy of `dual`).
            if coord_key == "nn":
                lat = float(LAT0 - 0.01 * p["y_nn"])
                lon = float(LON0 + 0.01 * p["x_nn"])
            elif coord_key == "phys":
                lat = float(LAT0 - 0.01 * p.get("y_phys", p["y_nn"]))
                lon = float(LON0 + 0.01 * p.get("x_phys", p["x_nn"]))
            else:
                lat = float(LAT0 - 0.01 * p.get("y_dual", p["y_nn"]))
                lon = float(LON0 + 0.01 * p.get("x_dual", p["x_nn"]))

            preds_by_uid[u].append(
                {
                    "lat": lat,
                    "lon": lon,
                    "prob": float(p.get("prob", 1.0)),
                    "score": float(p.get("score", 1.0)),
                    "meta": p,
                }
            )

    total_gt = sum(len(gt_map.get(u, [])) for u in uids)
    total_preds = sum(len(preds_by_uid[u]) for u in uids)

    curve_results = {}
    primary_result = {}

    for tol in tolerances:
        tot_hits, tot_miss, tot_fa = 0, 0, 0
        all_dists: List[float] = []

        for uid in uids:
            gt_objs = gt_map.get(uid, [])
            h, m, fa, dists, _ = el.spatial_bipartite_matching(
                preds_by_uid[uid], gt_objs, tolerance_km=tol
            )
            tot_hits += h
            tot_miss += m
            tot_fa += fa
            all_dists.extend(dists)

        pod = tot_hits / (tot_hits + tot_miss) if (tot_hits + tot_miss) > 0 else 0.0
        far = tot_fa / (tot_hits + tot_fa) if (tot_hits + tot_fa) > 0 else 0.0
        csi = tot_hits / (tot_hits + tot_miss + tot_fa) if (tot_hits + tot_miss + tot_fa) > 0 else 0.0

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
            "mean_distance_km": float(np.mean(all_dists)) if all_dists else float("nan"),
            "median_distance_km": float(np.median(all_dists)) if all_dists else float("nan"),
            "p25_distance_km": float(np.percentile(all_dists, 25)) if all_dists else float("nan"),
            "p75_distance_km": float(np.percentile(all_dists, 75)) if all_dists else float("nan"),
        }
        curve_results[f"tolerance_{int(tol)}km"] = res
        if abs(tol - primary_tol) < 1e-3:
            primary_result = res

    return primary_result, curve_results, total_input_candidates, suppressed_count


def render_full_domain_visualization(
    sample_npz_path: Path,
    candidates_by_uid: Dict[str, List[dict]],
    gt_map: Dict[str, List[dict]],
    out_png: Path,
):
    """Render a 4-panel publication figure of the full 2001x2101 continuous perception state."""
    if not sample_npz_path.exists():
        print(f"[WARN] Visual sample {sample_npz_path} not found. Skipping plot rendering.", flush=True)
        return

    print(f"[PLOT] Loading visualization field from {sample_npz_path}...")
    z = np.load(sample_npz_path, allow_pickle=True)
    uid = str(z["uid"])
    P_full = z["P_full"].astype(np.float32)
    H_full = z["H_full"].astype(np.float32)
    bt13 = z["bt13_t10"].astype(np.float32)

    gt_objs = gt_map.get(uid, [])
    cand_peaks = candidates_by_uid.get(uid, [])

    # Classify candidate peaks into kept vs gated
    kept_peaks = []
    gated_peaks = []
    for p in cand_peaks:
        d_cooling = p.get("delta_bt13_local", 0.0)
        b10_b13 = p.get("b10_b13_local", -99.0)
        if d_cooling > -0.5 and b10_b13 < 0.0:
            gated_peaks.append(p)
        else:
            kept_peaks.append(p)

    # Perform matching for visual annotation
    h, m, fa, dists, matched_pairs = el.spatial_bipartite_matching(
        kept_peaks, gt_objs, tolerance_km=30.0
    )

    print(f"[PLOT] Rendering figure for {uid} (GT: {len(gt_objs)}, Kept: {len(kept_peaks)}, Gated: {len(gated_peaks)}, Hits: {h})...")
    fig, axes = plt.subplots(2, 2, figsize=(18, 16), dpi=150)

    # 1. Native IR BT13 (T-10)
    ax0 = axes[0, 0]
    im0 = ax0.imshow(bt13, cmap="gist_ncar_r", vmin=190, vmax=300, origin="upper")
    ax0.set_title(f"(a) Native Himawari-8 Infrared B13 (10.4 μm) T-10 min\n[{uid}]", fontsize=13, fontweight="bold")
    plt.colorbar(im0, ax=ax0, fraction=0.046, pad=0.04, label="Brightness Temperature (K)")

    # 2. Dense Continuous CI Probability Field P_full
    ax1 = axes[0, 1]
    im1 = ax1.imshow(P_full, cmap="plasma", vmin=0.0, vmax=1.0, origin="upper")
    ax1.set_title("(b) 2D Hann-Blended Full-Domain Probability P_full(y, x)\n[Stride 56 px Overlap Fusion]", fontsize=13, fontweight="bold")
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04, label="P(CI >= 1)")

    # 3. Dense Neural Heatmap H_full
    ax2 = axes[1, 0]
    im2 = ax2.imshow(H_full, cmap="inferno", vmin=0.0, vmax=max(0.1, float(H_full.max())), origin="upper")
    ax2.set_title("(c) Multi-Task Convective Core Keypoint Heatmap H_full(y, x)\n[CINet_v2 Sub-window Head]", fontsize=13, fontweight="bold")
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04, label="Heatmap Intensity")

    # 4. Operational CI Perception & Targeted Gating
    ax3 = axes[1, 1]
    ax3.imshow(bt13, cmap="gray_r", vmin=200, vmax=295, origin="upper")

    # Plot Ground Truth CI objects (Red open circles)
    for g in gt_objs:
        y_g = (LAT0 - g["lat"]) / 0.01
        x_g = (g["lon"] - LON0) / 0.01
        circle = plt.Circle((x_g, y_g), 27.0, color="red", fill=False, linewidth=1.5, linestyle="--", alpha=0.8)
        ax3.add_patch(circle)
        ax3.plot(x_g, y_g, "ro", markersize=6, label="Ground Truth CI" if g == gt_objs[0] else "")

    # Plot Kept Predictions (Cyan stars)
    for p in kept_peaks:
        x_p = p.get("x_dual", p["x_nn"])
        y_p = p.get("y_dual", p["y_nn"])
        ax3.plot(x_p, y_p, "c*", markersize=9, markeredgecolor="black", markeredgewidth=0.8,
                 label="Active Predicted CI (Kept)" if p == kept_peaks[0] else "")

    # Plot Gated Decaying Anvils (Yellow 'x' marks)
    for p in gated_peaks:
        x_p = p.get("x_dual", p["x_nn"])
        y_p = p.get("y_dual", p["y_nn"])
        ax3.plot(x_p, y_p, "yx", markersize=8, markeredgewidth=1.8,
                 label="Suppressed Decaying Anvil" if p == gated_peaks[0] else "")

    # Annotations
    ax3.set_title(
        f"(d) Dense Continuous Perception & Local Physics Gating\n"
        f"[30 km Matches: {h}/{len(gt_objs)} (POD={h/max(1, len(gt_objs)):.1%}) | Gated Anvils: {len(gated_peaks)}]",
        fontsize=13, fontweight="bold"
    )
    handles, labels = ax3.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax3.legend(by_label.values(), by_label.keys(), loc="lower left", fontsize=10, framealpha=0.9)

    for ax in axes.flat:
        ax.set_xlim(0, W)
        ax.set_ylim(H, 0)
        ax.set_xlabel("Grid X (0.01° / px)")
        ax.set_ylabel("Grid Y (0.01° / px)")

    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] High-resolution visualization saved to {out_png}")


def main():
    ap = argparse.ArgumentParser(description="Object-Centric Local Microphysics Gating & Full Evaluation")
    ap.add_argument("--candidates-dir", default="experiments/final")
    ap.add_argument("--out-json", default="experiments/final/dense_perception_metrics.json")
    ap.add_argument("--vis-sample", default="experiments/final/dense_vis_sample.npz")
    ap.add_argument("--out-fig", default="reports/figures/full_domain_dense_perception.png")
    args = ap.parse_args()

    cand_dir = Path(args.candidates_dir)
    out_json = Path(args.out_json)
    out_fig = Path(args.out_fig)

    bench_cand_file = cand_dir / "dense_candidates_benchmark.json"
    test_cand_file = cand_dir / "dense_candidates_test.json"

    # Load baseline dual metrics for comparison
    dual_file = D / "final/cinet_v2_localization_dual_metrics.json"
    coarse_baseline = {}
    if dual_file.exists():
        coarse_baseline = json.loads(dual_file.read_text())

    results: Dict[str, Any] = {
        "title": "Dense Grid Continuous Perception & Object-Centric Physics Gating Evaluation",
        "description": "Full-domain 56-stride Hann-reconstructed CI perception with 32x32 local microphysical decaying-anvil gating.",
        "splits": {},
    }

    available_splits = []
    bench_candidates = {}
    gt_bench = {}
    if bench_cand_file.exists():
        bench_candidates = json.loads(bench_cand_file.read_text(encoding="utf-8"))
        gt_bench = el.load_ground_truth_objects("val", set(bench_candidates.keys()))
        available_splits.append(("benchmark", bench_candidates, gt_bench))
    if test_cand_file.exists():
        test_candidates = json.loads(test_cand_file.read_text(encoding="utf-8"))
        gt_test = el.load_ground_truth_objects("test", set(test_candidates.keys()))
        available_splits.append(("test", test_candidates, gt_test))

    if not available_splits:
        raise FileNotFoundError(f"No candidate files found in {cand_dir}. Please run dense_inference_v2.py first.")

    for s_name, candidates, gt_map in available_splits:
        print(f"\n{'='*60}")
        print(f"EVALUATING SPLIT: {s_name.upper()}")
        print(f"{'='*60}")

        # 1. Pre-Gating (Raw Dense Candidates with Dual Coordinates)
        pre_prim, pre_curve, n_cand, _ = evaluate_candidate_dict(
            candidates, gt_map, filter_decaying=False, coord_key="dual"
        )
        print(f"[{s_name.upper()}] Pre-Gating (30 km): Hits={pre_prim['hits']}/{pre_prim['total_gt']} | POD={pre_prim['POD_obj']:.4f} | FAR={pre_prim['FAR_obj']:.4f} | CSI={pre_prim['CSI_obj']:.4f}")

        # 2. Post-Gating (Targeted Local Microphysical Decaying Anvil Gating)
        post_prim, post_curve, _, n_gated = evaluate_candidate_dict(
            candidates, gt_map, filter_decaying=True, decay_delta_thr=-0.5, coord_key="dual"
        )
        print(f"[{s_name.upper()}] Post-Gating (30 km): Hits={post_prim['hits']}/{post_prim['total_gt']} | POD={post_prim['POD_obj']:.4f} | FAR={post_prim['FAR_obj']:.4f} | CSI={post_prim['CSI_obj']:.4f}")

        hit_retention = post_prim["hits"] / max(1, pre_prim["hits"])
        fa_reduction = (pre_prim["false_alarms"] - post_prim["false_alarms"]) / max(1, pre_prim["false_alarms"])
        print(f"[{s_name.upper()}] Gating Verification: Hit Retention = {hit_retention:.1%} (Target > 75%) | FA Suppressed = {n_gated} ({fa_reduction:.1%})")

        # 3. Coordinate Refinement Ablation (Neural Alone vs Dual vs Physics Alone)
        nn_prim, _, _, _ = evaluate_candidate_dict(candidates, gt_map, filter_decaying=True, coord_key="nn")
        phys_prim, _, _, _ = evaluate_candidate_dict(candidates, gt_map, filter_decaying=True, coord_key="phys")

        # Extract baseline numbers for clean cross-comparison
        b_data = coarse_baseline.get("splits", {}).get(s_name, {}).get("modes", {})
        coarse_lm_geom = b_data.get("local_max", {}).get("strategy_a_geom", {}).get("primary_30km", {})
        coarse_lm_dual = b_data.get("local_max", {}).get("strategy_d_dual_optimal", {}).get("primary_30km", {})
        coarse_aa_geom = b_data.get("all_active", {}).get("strategy_a_geom", {}).get("primary_30km", {})
        coarse_aa_dual = b_data.get("all_active", {}).get("strategy_d_dual_optimal", {}).get("primary_30km", {})

        results["splits"][s_name] = {
            "n_timestamps": len(candidates),
            "total_gt_objects": pre_prim["total_gt"],
            "total_raw_candidates": n_cand,
            "suppressed_decaying_anvils": n_gated,
            "hit_retention_pct": round(hit_retention * 100, 2),
            "fa_suppressed_pct": round(fa_reduction * 100, 2),
            "coarse_grid_baseline": {
                "local_max_geom": coarse_lm_geom,
                "local_max_dual": coarse_lm_dual,
                "all_active_geom": coarse_aa_geom,
                "all_active_dual": coarse_aa_dual,
            },
            "dense_continuous_perception": {
                "pre_gating": {
                    "primary_30km": pre_prim,
                    "tolerance_curve": pre_curve,
                },
                "post_gating_targeted_anvil": {
                    "primary_30km": post_prim,
                    "tolerance_curve": post_curve,
                },
                "coordinate_ablation_post_gating": {
                    "pure_neural": nn_prim,
                    "dual_optimal": post_prim,
                    "pure_physics": phys_prim,
                },
            },
        }

    # Save summary JSON
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    print(f"\n[OUTPUT] Saved detailed perception metrics to {out_json}")

    # Render publication visualization figure
    render_full_domain_visualization(
        Path(args.vis_sample),
        bench_candidates,
        gt_bench,
        out_fig,
    )

    print("\n==================================================")
    print("Local Physics Gating & Evaluation completed successfully.")
    print("==================================================")


if __name__ == "__main__":
    main()
