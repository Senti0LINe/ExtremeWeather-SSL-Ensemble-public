"""Multi-Tier Operational Warning System Evaluation & Object-Centric PR Curve.

Tools:
    tools/eval_operational_tiers.py

Specification:
1. Loads dense candidates for dev, benchmark and test from
   experiments/final/dense_candidates_{dev,benchmark,test}.json.
2. Selects the operating thresholds on DEV ONLY by sweeping tau_obj over
   [0.30 .. 0.80], then freezes them. Benchmark and test are scored once at the
   frozen values and are never used for selection.
3. Computes 30 km spatial bipartite matching: POD_obj, FAR_obj, CSI_obj,
   Precision, Recall, and average precision over ALL candidates ranked by score.
4. Evaluates three operational warning tiers:
   - Red (Level 1 - Extreme Warning): tau_obj >= dev-selected red threshold
     AND Delta_BT13_local <= -2.0 K.
   - Orange (Level 2 - Severe Alert): dev-selected max-CSI balance point.
   - Yellow (Level 3 - Surveillance Watch): Full valid candidate set (max POD).
5. Plots publication-quality 3-panel figure: reports/figures/operational_tiers_pr_curve.png.
6. Saves metrics plus the threshold-selection provenance to
   experiments/final/operational_tiers_metrics.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

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
SWEEP_TAUS = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]


def is_decaying_anvil(cand: dict, threshold: float = -0.5) -> bool:
    """Microphysics rule to detect decaying anvils."""
    d_cooling = cand.get("delta_bt13_local", 0.0)
    b10_b13 = cand.get("b10_b13_local", -99.0)
    return d_cooling > threshold and b10_b13 < 0.0


def evaluate_tier_condition(
    candidates_by_uid: Dict[str, List[dict]],
    gt_map: Dict[str, List[dict]],
    filter_fn: Callable[[dict], bool],
    tier_name: str,
    tolerance_km: float = 30.0,
) -> Dict[str, Any]:
    """Evaluate candidate subset under specific filtering function using bipartite matching."""
    uids = sorted(list(candidates_by_uid.keys()))
    preds_by_uid = {u: [] for u in uids}
    total_preds = 0

    for u in uids:
        for p in candidates_by_uid[u]:
            if filter_fn(p):
                total_preds += 1
                lat = float(p.get("lat", LAT0 - 0.01 * p.get("y_dual", p["y_nn"])))
                lon = float(p.get("lon", LON0 + 0.01 * p.get("x_dual", p["x_nn"])))
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
    tot_hits, tot_miss, tot_fa = 0, 0, 0
    all_dists: List[float] = []

    for u in uids:
        gt_objs = gt_map.get(u, [])
        h, m, fa, dists, _ = el.spatial_bipartite_matching(
            preds_by_uid[u], gt_objs, tolerance_km=tolerance_km
        )
        tot_hits += h
        tot_miss += m
        tot_fa += fa
        all_dists.extend(dists)

    pod = tot_hits / (tot_hits + tot_miss) if (tot_hits + tot_miss) > 0 else 0.0
    far = tot_fa / (tot_hits + tot_fa) if (tot_hits + tot_fa) > 0 else 0.0
    csi = tot_hits / (tot_hits + tot_miss + tot_fa) if (tot_hits + tot_miss + tot_fa) > 0 else 0.0
    precision = 1.0 - far
    recall = pod

    res = {
        "tier_name": tier_name,
        "tolerance_km": float(tolerance_km),
        "total_preds": int(total_preds),
        "total_gt": int(total_gt),
        "hits": int(tot_hits),
        "misses": int(tot_miss),
        "false_alarms": int(tot_fa),
        "POD_obj": float(pod),
        "FAR_obj": float(far),
        "CSI_obj": float(csi),
        "Precision_obj": float(precision),
        "Recall_obj": float(recall),
        "mean_dist_km": float(np.mean(all_dists)) if all_dists else float("nan"),
        "median_dist_km": float(np.median(all_dists)) if all_dists else float("nan"),
    }
    return res


def compute_pr_auc(recalls: List[float], precisions: List[float]) -> float:
    """Compute Area Under Precision-Recall Curve (PR-AUC) using trapezoidal integration."""
    points = sorted(zip(recalls, precisions), key=lambda x: x[0])
    rec_sorted = [p[0] for p in points]
    prec_sorted = [p[1] for p in points]
    if len(rec_sorted) < 2:
        return 0.0
    if hasattr(np, "trapezoid"):
        auc = float(np.trapezoid(prec_sorted, rec_sorted))
    else:
        try:
            from scipy.integrate import trapezoid
            auc = float(trapezoid(prec_sorted, rec_sorted))
        except Exception:
            auc = float(sum((rec_sorted[i] - rec_sorted[i - 1]) * (prec_sorted[i] + prec_sorted[i - 1]) / 2.0 for i in range(1, len(rec_sorted))))
    return max(0.0, auc)


def compute_object_ap(
    candidates_by_uid: Dict[str, List[dict]],
    gt_map: Dict[str, List[dict]],
    filter_fn: Callable[[dict], bool],
    tolerance_km: float = 30.0,
) -> Tuple[float, int, int]:
    """Average precision over ALL candidates ranked by score.

    Replaces the 11-point truncated trapezoid area as the headline metric
    (EXPERIMENTS E14 P6): that estimator only covered recall 13.8%-35.4% and
    used a different rule from the window-level sklearn average_precision_score,
    so the two were not comparable. Here every candidate participates, ranked
    by score descending, with one-to-one nearest-free-GT matching per timestamp.

    Returns (AP, true_positives, false_positives).
    """
    ranked: List[Tuple[float, str, int]] = []
    total_gt = 0
    for uid, cands in candidates_by_uid.items():
        total_gt += len(gt_map.get(uid, []))
        for i, c in enumerate(cands):
            if filter_fn(c):
                ranked.append((float(c.get("score", c.get("prob", 0.0))), uid, i))
    if total_gt == 0 or not ranked:
        return 0.0, 0, 0

    ranked.sort(key=lambda t: -t[0])
    matched: Dict[str, set] = {uid: set() for uid in candidates_by_uid}
    tp = 0
    fp = 0
    prec_at_tp: List[float] = []

    for _score, uid, i in ranked:
        c = candidates_by_uid[uid][i]
        lat = float(c.get("lat", LAT0 - 0.01 * c.get("y_dual", c["y_nn"])))
        lon = float(c.get("lon", LON0 + 0.01 * c.get("x_dual", c["x_nn"])))
        best_j, best_d = -1, tolerance_km
        for j, g in enumerate(gt_map.get(uid, [])):
            if j in matched[uid]:
                continue
            d = el.haversine_km(lat, lon, g["lat"], g["lon"])
            if d <= best_d:
                best_d, best_j = d, j
        if best_j >= 0:
            matched[uid].add(best_j)
            tp += 1
            prec_at_tp.append(tp / (tp + fp))
        else:
            fp += 1

    return float(sum(prec_at_tp) / total_gt), tp, fp


def plot_operational_tiers_pr_curve(
    sweep_results: Dict[str, List[Dict[str, Any]]],
    tier_results: Dict[str, Dict[str, Dict[str, Any]]],
    out_path: Path,
    ap_by_split: Dict[str, float] | None = None,
):
    """Plot publication-quality 3-panel figure showing PR curve, CSI trade-off, and forecaster workload reduction."""
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(1, 3, figsize=(21, 6.5), dpi=150)
    fig.subplots_adjust(wspace=0.28, top=0.88, bottom=0.14, left=0.06, right=0.96)

    # Focus on Held-out Test as primary, Benchmark as secondary
    test_sweep = sweep_results["test"]
    test_tiers = tier_results["test"]

    test_recalls = [r["Recall_obj"] for r in test_sweep]
    test_precs = [r["Precision_obj"] for r in test_sweep]
    test_csis = [r["CSI_obj"] for r in test_sweep]
    test_pods = [r["POD_obj"] for r in test_sweep]
    test_fars = [r["FAR_obj"] for r in test_sweep]
    test_preds = [r["total_preds"] for r in test_sweep]
    taus = [r["tau"] for r in test_sweep]

    test_pr_auc = (ap_by_split or {}).get("test", compute_pr_auc(test_recalls, test_precs))

    bench_sweep = sweep_results.get("benchmark", [])
    bench_recalls = [r["Recall_obj"] for r in bench_sweep] if bench_sweep else []
    bench_precs = [r["Precision_obj"] for r in bench_sweep] if bench_sweep else []
    bench_pr_auc = (ap_by_split or {}).get("benchmark", compute_pr_auc(bench_recalls, bench_precs)) if bench_sweep else 0.0

    # -------------------------------------------------------------
    # Panel 1: Single-cell Precision-Recall Curve
    # -------------------------------------------------------------
    ax0 = axes[0]
    ax0.plot(test_recalls, test_precs, "o-", color="#1f77b4", lw=2.4, ms=6, label=f"Held-out Test (AP={test_pr_auc:.4f})")
    if bench_sweep:
        ax0.plot(bench_recalls, bench_precs, "s--", color="#7f7f7f", lw=1.6, ms=5, alpha=0.75, label=f"Benchmark (AP={bench_pr_auc:.4f})")

    # Mark Operational Tiers on Test
    y_tier = test_tiers["yellow_level3"]
    o_tier = test_tiers["orange_level2"]
    r_tier = test_tiers["red_level1"]

    ax0.scatter([y_tier["Recall_obj"]], [y_tier["Precision_obj"]], s=170, color="#f1c40f", edgecolors="black", lw=1.5, zorder=5, label=f"Level 3 Watch (POD={y_tier['POD_obj']:.1%})")
    ax0.scatter([o_tier["Recall_obj"]], [o_tier["Precision_obj"]], s=200, marker="D", color="#e67e22", edgecolors="black", lw=1.5, zorder=5, label=f"Level 2 Alert (CSI={o_tier['CSI_obj']:.1%})")
    ax0.scatter([r_tier["Recall_obj"]], [r_tier["Precision_obj"]], s=240, marker="*", color="#e74c3c", edgecolors="black", lw=1.5, zorder=5, label=f"Level 1 Warning (ΔBT<=-2K)")

    # Annotate threshold values along the curve
    for r in test_sweep:
        if r["tau"] in [0.40, 0.60, 0.70, 0.80]:
            ax0.annotate(
                f"τ={r['tau']:.2f}",
                xy=(r["Recall_obj"], r["Precision_obj"]),
                xytext=(r["Recall_obj"] + 0.015, r["Precision_obj"] + 0.004),
                fontsize=9,
                color="#0b3c5d",
                fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#0b3c5d", lw=0.8),
            )

    ax0.set_title("(a) Object-Centric Precision-Recall Curve (30 km)", fontsize=13, fontweight="bold", pad=10)
    ax0.set_xlabel("Recall / POD (Detection Rate)", fontsize=11, fontweight="bold")
    ax0.set_ylabel("Precision (1 - FAR)", fontsize=11, fontweight="bold")
    ax0.set_xlim(0.08, 0.42)
    ax0.set_ylim(0.02, 0.13)
    ax0.grid(True, linestyle="--", alpha=0.5)
    ax0.legend(loc="upper right", fontsize=9.5, framealpha=0.92)

    # -------------------------------------------------------------
    # Panel 2: Operational CSI, POD, and FAR trade-off vs tau
    # -------------------------------------------------------------
    ax1 = axes[1]
    ax1.plot(taus, test_csis, "d-", color="#2980b9", lw=2.5, ms=7, label="CSI_obj (Threat Score)")
    ax1.plot(taus, test_pods, "^--", color="#27ae60", lw=2.0, ms=6, label="POD_obj (Hit Rate)")
    
    # Secondary y-axis for FAR
    ax1_r = ax1.twinx()
    ax1_r.plot(taus, test_fars, "v:", color="#c0392b", lw=2.0, ms=6, label="FAR_obj (False Alarm Ratio)")

    # Highlight optimal CSI operating threshold
    opt_tau = o_tier["tau_operating"]
    ax1.axvline(x=opt_tau, color="#e67e22", linestyle="-.", lw=1.8, alpha=0.85, label=f"Orange Opt τ={opt_tau:.2f}")

    ax1.set_title("(b) Operational Detection Metrics vs. Confidence Threshold", fontsize=13, fontweight="bold", pad=10)
    ax1.set_xlabel("Candidate Confidence Threshold (τ_obj)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("CSI / POD Score", fontsize=11, fontweight="bold", color="#2980b9")
    ax1_r.set_ylabel("FAR Score", fontsize=11, fontweight="bold", color="#c0392b")
    ax1.set_xlim(0.28, 0.82)
    ax1.set_ylim(0.0, 0.40)
    ax1_r.set_ylim(0.88, 0.98)
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Combine legends from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_r.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center left", fontsize=9.5, framealpha=0.92)

    # -------------------------------------------------------------
    # Panel 3: Workload & Alert Reduction vs. CSI Improvement
    # -------------------------------------------------------------
    ax2 = axes[2]
    total_raw_preds = y_tier["total_preds"]
    reductions = [100.0 * (1.0 - p / total_raw_preds) for p in test_preds]

    bars = ax2.bar(
        [t - 0.008 for t in taus],
        [p / 1000.0 for p in test_preds],
        width=0.024,
        color="#34495e",
        alpha=0.78,
        label="Alert Volume (×1,000 cells)",
    )

    ax2_r = ax2.twinx()
    ax2_r.plot(taus, reductions, "s-", color="#e67e22", lw=2.2, ms=6, label="Workload Reduction (%)")
    ax2_r.plot(taus, [c * 100.0 for c in test_csis], "o--", color="#2980b9", lw=2.0, ms=6, label="CSI Score (%)")

    # Annotate Orange Tier benefit
    orange_idx = taus.index(opt_tau)
    orange_reduction = reductions[orange_idx]
    orange_csi_gain = 100.0 * (o_tier["CSI_obj"] - y_tier["CSI_obj"]) / max(1e-9, y_tier["CSI_obj"])
    ax2.annotate(
        f"Orange Tier:\n-{orange_reduction:.1f}% Alerts\n+{orange_csi_gain:.1f}% CSI",
        xy=(opt_tau, test_preds[orange_idx] / 1000.0),
        xytext=(opt_tau - 0.12, (test_preds[orange_idx] / 1000.0) + 12),
        fontsize=9.5,
        fontweight="bold",
        color="#b95300",
        bbox=dict(boxstyle="round,pad=0.3", fc="#fef5e7", ec="#e67e22", lw=1.2),
        arrowprops=dict(arrowstyle="->", color="#e67e22", lw=1.2),
    )

    ax2.set_title("(c) Forecaster Workload Reduction & Operational Gain", fontsize=13, fontweight="bold", pad=10)
    ax2.set_xlabel("Candidate Confidence Threshold (τ_obj)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Alert Volume Issued (×1,000 cells)", fontsize=11, fontweight="bold", color="#34495e")
    ax2_r.set_ylabel("Reduction / CSI Score (%)", fontsize=11, fontweight="bold", color="#e67e22")
    ax2.set_xlim(0.26, 0.84)
    ax2.set_ylim(0, 60)
    ax2_r.set_ylim(0, 100)
    ax2.grid(True, linestyle="--", alpha=0.5)

    lines3, labels3 = ax2.get_legend_handles_labels()
    lines4, labels4 = ax2_r.get_legend_handles_labels()
    ax2.legend(lines3 + lines4, labels4 + labels3, loc="upper right", fontsize=9.5, framealpha=0.92)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[PLOT] High-quality PR curve and operational trade-off plot saved to: {out_path}")


def main():
    ap = argparse.ArgumentParser(description="Multi-Tier Operational Warning System Evaluation")
    ap.add_argument("--candidates-dir", default="experiments/final")
    ap.add_argument("--out-json", default="experiments/final/operational_tiers_metrics.json")
    ap.add_argument("--out-fig", default="reports/figures/operational_tiers_pr_curve.png")
    ap.add_argument("--tolerance", type=float, default=30.0)
    ap.add_argument("--plot-only", action="store_true", help="Plot directly from existing JSON")
    args = ap.parse_args()

    cand_dir = Path(args.candidates_dir)
    out_json = Path(args.out_json)
    out_fig = Path(args.out_fig)
    tol = args.tolerance

    if args.plot_only and out_json.exists():
        data_json = json.loads(out_json.read_text(encoding="utf-8"))
        sweep_res = {}
        tier_res = {}
        ap_res = {}
        for s in data_json["splits"]:
            sweep_res[s] = data_json["splits"][s]["threshold_sweep"]
            tier_res[s] = data_json["splits"][s]["operational_tiers"]
            ap_res[s] = data_json["splits"][s].get("pr_auc", 0.0)
        plot_operational_tiers_pr_curve(sweep_res, tier_res, out_fig, ap_res)
        print(f"[FAST] Re-rendered plot from {out_json} to {out_fig}")
        return

    bench_file = cand_dir / "dense_candidates_benchmark.json"
    test_file = cand_dir / "dense_candidates_test.json"
    dev_file = cand_dir / "dense_candidates_dev.json"

    # Operating thresholds are selected on DEV ONLY, then frozen for
    # benchmark/test (EXPERIMENTS E14: they used to be hardcoded per split at
    # each split's own best-CSI point, which is tuning on the evaluation set).
    if not dev_file.exists():
        raise FileNotFoundError(
            f"{dev_file} not found. Run tools/dense_inference_v2.py --split dev first: "
            "the operating thresholds must be selected on dev, never on benchmark/test."
        )
    dev_cands = json.loads(dev_file.read_text(encoding="utf-8"))
    gt_dev = el.load_ground_truth_objects("val", set(dev_cands.keys()))

    dev_sweep = []
    for tau in SWEEP_TAUS:
        r = evaluate_tier_condition(
            dev_cands, gt_dev,
            lambda p, t=tau: (p.get("prob", 0.0) >= t) and (not is_decaying_anvil(p)),
            f"dev_sweep_{tau:.2f}", tolerance_km=tol,
        )
        r["tau"] = float(tau)
        dev_sweep.append(r)
    tau_orange = max(dev_sweep, key=lambda r: r["CSI_obj"])["tau"]

    dev_sweep_cool = []
    for tau in SWEEP_TAUS:
        r = evaluate_tier_condition(
            dev_cands, gt_dev,
            lambda p, t=tau: (p.get("prob", 0.0) >= t)
            and (p.get("delta_bt13_local", 0.0) <= -2.0)
            and (not is_decaying_anvil(p)),
            f"dev_sweep_cool_{tau:.2f}", tolerance_km=tol,
        )
        r["tau"] = float(tau)
        dev_sweep_cool.append(r)
    tau_red = max(dev_sweep_cool, key=lambda r: r["CSI_obj"])["tau"]
    # Red strict = the dev-optimal overall confidence point AND deep cooling.
    tau_red_strict = tau_orange

    print(f"[DEV] frozen operating thresholds: orange={tau_orange:.2f} "
          f"red={tau_red:.2f} red_strict={tau_red_strict:.2f}")
    tier_selection = {
        "selected_on": "dev",
        "dev_timestamps": len(dev_cands),
        "tau_orange": float(tau_orange),
        "tau_red": float(tau_red),
        "tau_red_strict": float(tau_red_strict),
        "dev_sweep": dev_sweep,
        "dev_sweep_with_cooling": dev_sweep_cool,
    }

    available_splits = []
    if bench_file.exists():
        bench_cands = json.loads(bench_file.read_text(encoding="utf-8"))
        gt_bench = el.load_ground_truth_objects("val", set(bench_cands.keys()))
        available_splits.append(("benchmark", bench_cands, gt_bench, tau_orange, tau_red, tau_red_strict))
    if test_file.exists():
        test_cands = json.loads(test_file.read_text(encoding="utf-8"))
        gt_test = el.load_ground_truth_objects("test", set(test_cands.keys()))
        available_splits.append(("test", test_cands, gt_test, tau_orange, tau_red, tau_red_strict))

    if not available_splits:
        raise FileNotFoundError(f"No candidate files found in {cand_dir}.")

    sweep_results: Dict[str, List[Dict[str, Any]]] = {}
    tier_results: Dict[str, Dict[str, Dict[str, Any]]] = {}
    ap_by_split: Dict[str, float] = {}

    for s_name, cands_map, gt_map, opt_tau, red_tau, red_strict_tau in available_splits:
        print(f"\n{'='*70}")
        print(f"EVALUATING OPERATIONAL TIERS & PR CURVE: {s_name.upper()}")
        print(f"{'='*70}")

        # 1. Sweep Thresholds
        split_sweep = []
        for tau in SWEEP_TAUS:
            fn = lambda p, t=tau: (p.get("prob", 0.0) >= t) and (not is_decaying_anvil(p))
            res = evaluate_tier_condition(
                cands_map, gt_map, fn, f"Sweep_tau_{tau:.2f}", tolerance_km=tol
            )
            res["tau"] = float(tau)
            split_sweep.append(res)
            print(
                f"[{s_name.upper()}] tau={tau:.2f} | Preds={res['total_preds']:5d} | "
                f"Hits={res['hits']:4d}/{res['total_gt']} | "
                f"POD={res['POD_obj']:.4f} | FAR={res['FAR_obj']:.4f} | CSI={res['CSI_obj']:.4f} | "
                f"Precision={res['Precision_obj']:.4f}"
            )
        sweep_results[s_name] = split_sweep

        # Compute PR-AUC
        recs = [r["Recall_obj"] for r in split_sweep]
        precs = [r["Precision_obj"] for r in split_sweep]
        pr_auc_trunc = compute_pr_auc(recs, precs)
        ap_full, ap_tp, ap_fp = compute_object_ap(
            cands_map, gt_map, lambda p: not is_decaying_anvil(p), tol
        )
        ap_by_split[s_name] = ap_full
        print(f"[{s_name.upper()}] Object AP (all candidates ranked by score, 30 km): {ap_full:.5f}  TP={ap_tp} FP={ap_fp}")
        print(f"[{s_name.upper()}] Legacy truncated 11-point PR area (recall {min(recs):.3f}-{max(recs):.3f}): {pr_auc_trunc:.5f}")

        # 2. Three Operational Tiers
        # Level 3: Yellow Watch (all valid post-gating candidates, preserving max detection rate)
        fn_yellow = lambda p: not is_decaying_anvil(p)
        yellow_res = evaluate_tier_condition(
            cands_map, gt_map, fn_yellow, "Level 3 - Surveillance Watch (Yellow)", tolerance_km=tol
        )
        yellow_res["description"] = "Full set of valid post-gating candidate peaks. Preserves maximum detection recall for early monitoring."

        # Level 2: Orange Severe Alert (optimal CSI operating balance point)
        fn_orange = lambda p, t=opt_tau: (p.get("prob", 0.0) >= t) and (not is_decaying_anvil(p))
        orange_res = evaluate_tier_condition(
            cands_map, gt_map, fn_orange, "Level 2 - Severe Alert (Orange)", tolerance_km=tol
        )
        orange_res["tau_operating"] = float(opt_tau)
        orange_res["description"] = f"Operational optimal CSI balance point (tau_obj={opt_tau:.2f}), selected on dev and frozen. Substantially reduces forecaster false alarm fatigue."

        # Level 1: Red Extreme Warning (dev-frozen confidence AND deep cooling Delta_BT13 <= -2.0 K)
        fn_red = lambda p, t=red_tau: (p.get("prob", 0.0) >= t) and (p.get("delta_bt13_local", 0.0) <= -2.0) and (not is_decaying_anvil(p))
        red_res = evaluate_tier_condition(
            cands_map, gt_map, fn_red, "Level 1 - Extreme Warning (Red)", tolerance_km=tol
        )
        red_res["tau_operating"] = float(red_tau)
        red_res["delta_bt13_max"] = -2.0
        red_res["description"] = f"High-confidence rapid overshooting convective cores (tau_obj >= {red_tau:.2f} & Delta_BT13 <= -2.0 K), threshold frozen on dev. Actionable emergency warning."

        # Red strict: the dev-optimal overall confidence point AND deep cooling.
        fn_red_strict = lambda p, t=red_strict_tau: (p.get("prob", 0.0) >= t) and (p.get("delta_bt13_local", 0.0) <= -2.0) and (not is_decaying_anvil(p))
        red_strict_res = evaluate_tier_condition(
            cands_map, gt_map, fn_red_strict, "Level 1 Strict - Extreme Warning (Red Strict)", tolerance_km=tol
        )
        red_strict_res["tau_operating"] = float(red_strict_tau)
        red_strict_res["delta_bt13_max"] = -2.0

        tier_results[s_name] = {
            "yellow_level3": yellow_res,
            "orange_level2": orange_res,
            "red_level1": red_res,
            "red_level1_strict": red_strict_res,
        }

        print(f"\n--- {s_name.upper()} OPERATIONAL TIERS SUMMARY ---")
        print(f"  🟡 Level 3 (Watch):   Preds={yellow_res['total_preds']:5d} | POD={yellow_res['POD_obj']:.1%} | FAR={yellow_res['FAR_obj']:.1%} | CSI={yellow_res['CSI_obj']:.2%}")
        print(f"  🟠 Level 2 (Alert):   Preds={orange_res['total_preds']:5d} | POD={orange_res['POD_obj']:.1%} | FAR={orange_res['FAR_obj']:.1%} | CSI={orange_res['CSI_obj']:.2%}")
        print(f"  🔴 Level 1 (Warning): Preds={red_res['total_preds']:5d} | POD={red_res['POD_obj']:.1%} | FAR={red_res['FAR_obj']:.1%} | CSI={red_res['CSI_obj']:.2%}")

    # Build final dictionary
    output_dict: Dict[str, Any] = {
        "title": "Operational Multi-Tier Warning System & Object-Centric PR Evaluation",
        "tolerance_km": tol,
        "tier_selection": tier_selection,
        "splits": {},
    }

    for s_name in sweep_results:
        st = tier_results[s_name]
        sw = sweep_results[s_name]
        recs = [r["Recall_obj"] for r in sw]
        precs = [r["Precision_obj"] for r in sw]
        output_dict["splits"][s_name] = {
            "pr_auc": float(ap_by_split.get(s_name, 0.0)),
            "pr_auc_estimator": "average_precision over all candidates ranked by score, one-to-one 30 km matching",
            "pr_auc_truncated_11pt_legacy": float(compute_pr_auc(recs, precs)),
            "pr_auc_legacy_recall_range": [float(min(recs)), float(max(recs))],
            "threshold_sweep": sw,
            "operational_tiers": st,
            "workload_reduction": {
                "orange_vs_yellow_alert_reduction_pct": round(
                    100.0 * (1.0 - st["orange_level2"]["total_preds"] / st["yellow_level3"]["total_preds"]), 2
                ),
                "orange_vs_yellow_csi_gain_pct": round(
                    100.0 * (st["orange_level2"]["CSI_obj"] - st["yellow_level3"]["CSI_obj"]) / st["yellow_level3"]["CSI_obj"], 2
                ),
            },
        }

    # Save metrics JSON
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump(output_dict, fh, indent=2)
    print(f"\n[OUTPUT] Saved operational tier metrics to: {out_json}")

    # Plot 3-panel figure
    plot_operational_tiers_pr_curve(sweep_results, tier_results, out_fig, ap_by_split)

    print("\n=======================================================")
    print("Multi-Tier Operational Evaluation completed successfully.")
    print("=======================================================")


if __name__ == "__main__":
    main()
