"""Object-level false-alarm root-cause diagnosis (tools/diagnose_far.py).

Read-only diagnostic. Consumes the cached dense candidates and the CIDS
annotations; trains nothing, changes no operating point, writes only
experiments/final/far_diagnosis.json.

Separates three candidate causes of the 90%+ object-level FAR:
  A. emission budget  - how many alerts are issued per real object;
  B. post-processing  - per-timestamp score normalisation, neighbourhood-extremum
                        gating, NMS radius, uncalibrated probability;
  C. model ranking    - does any stored confidence actually order true objects
                        above false ones?
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import numpy as np

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

import tools.eval_localization as el
import tools.eval_operational_tiers as ot

FULL = Path("/data1/group3/DATASAT/experiments/ci_det_v2/annotations_full")
D = PROJECT / "experiments"
LAT0, LON0 = 40.00448, 103.99570
R_KM = 6371.0
TOL = 30.0
TAU_DEV = 0.3143534994068802


def dist_matrix(plat, plon, glat, glon) -> np.ndarray:
    phi1 = np.radians(plat)[:, None]
    phi2 = np.radians(glat)[None, :]
    dphi = phi2 - phi1
    dlam = np.radians(glon)[None, :] - np.radians(plon)[:, None]
    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2.0) ** 2
    return R_KM * 2.0 * np.arctan2(np.sqrt(a), np.sqrt(np.maximum(0.0, 1.0 - a)))


def greedy_from_D(Dm: np.ndarray, tol: float) -> Tuple[np.ndarray, np.ndarray]:
    """Bilateral unique greedy nearest-neighbour matching, identical to
    el.spatial_bipartite_matching (stable distance sort, pred-major tie order)."""
    m, n = Dm.shape
    mp = np.zeros(m, dtype=bool)
    mg = np.zeros(n, dtype=bool)
    if m == 0 or n == 0:
        return mp, mg
    ii, jj = np.nonzero(Dm <= tol)
    if len(ii) == 0:
        return mp, mg
    dd = Dm[ii, jj]
    for k in np.argsort(dd, kind="stable"):
        i, j = ii[k], jj[k]
        if not mp[i] and not mg[j]:
            mp[i] = True
            mg[j] = True
    return mp, mg


def cand_coords(c: dict) -> Tuple[float, float]:
    lat = float(c.get("lat", LAT0 - 0.01 * c.get("y_dual", c["y_nn"])))
    lon = float(c.get("lon", LON0 + 0.01 * c.get("x_dual", c["x_nn"])))
    return lat, lon


def rank_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Mann-Whitney AUC with tie correction; 0.5 = no discriminative power."""
    n1 = int((labels == 1).sum())
    n0 = int((labels == 0).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(scores, kind="stable")
    s = scores[order]
    ranks = np.empty(len(scores), dtype=np.float64)
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks[order[i : j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return float((ranks[labels == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def ap_global(per_ts: List[dict], key_vals: List[np.ndarray], tol: float = TOL) -> Tuple[float, int, int]:
    """Mirror of ot.compute_object_ap with a pluggable ranking key.

    Candidates are ranked globally by descending key; each takes the nearest
    still-free GT within tol (last index wins an exact tie, as in the original).
    """
    ranked: List[Tuple[float, int, int]] = []
    total_gt = 0
    for ti, t in enumerate(per_ts):
        total_gt += len(t["gt"])
        for i in range(len(t["cands"])):
            ranked.append((float(key_vals[ti][i]), ti, i))
    if total_gt == 0 or not ranked:
        return 0.0, 0, 0
    ranked.sort(key=lambda z: -z[0])
    free = [np.ones(len(t["gt"]), dtype=bool) for t in per_ts]
    tp = fp = 0
    prec_at_tp: List[float] = []
    for _s, ti, i in ranked:
        Dm = per_ts[ti]["D"]
        if Dm.shape[1] == 0:
            fp += 1
            continue
        row = np.where(free[ti], Dm[i], np.inf)
        j = int(len(row) - 1 - np.argmin(row[::-1]))
        if row[j] <= tol:
            free[ti][j] = False
            tp += 1
            prec_at_tp.append(tp / (tp + fp))
        else:
            fp += 1
    return float(sum(prec_at_tp) / total_gt), tp, fp


def build_split(name: str) -> Tuple[List[dict], List[dict]]:
    """Return (raw_per_ts, gated_per_ts) sharing the same candidate dicts."""
    f = D / "final" / f"dense_candidates_{name}.json"
    raw = json.loads(f.read_text(encoding="utf-8"))
    role = "test" if name == "test" else "val"
    gt_map = el.load_ground_truth_objects(role, set(raw.keys()))
    raw_ts: List[dict] = []
    gated_ts: List[dict] = []
    for uid in sorted(raw):
        cands = raw[uid]
        gt = gt_map.get(uid, [])
        plat = np.array([cand_coords(c)[0] for c in cands])
        plon = np.array([cand_coords(c)[1] for c in cands])
        glat = np.array([g["lat"] for g in gt])
        glon = np.array([g["lon"] for g in gt])
        Dm = dist_matrix(plat, plon, glat, glon) if len(cands) and len(gt) else np.zeros((len(cands), len(gt)))
        raw_ts.append({"uid": uid, "cands": cands, "gt": gt, "D": Dm})
        keep = np.array([not ot.is_decaying_anvil(c) for c in cands], dtype=bool)
        gated_ts.append(
            {
                "uid": uid,
                "cands": [c for c, k in zip(cands, keep) if k],
                "gt": gt,
                "D": Dm[keep],
                "keep": keep,
            }
        )
    return raw_ts, gated_ts


def agg(per_ts: List[dict], keeps: List[np.ndarray], tol: float = TOL) -> Dict[str, Any]:
    H = M = F = P = G = 0
    for t, k in zip(per_ts, keeps):
        mp, _ = greedy_from_D(t["D"][k], tol)
        h = int(mp.sum())
        H += h
        M += len(t["gt"]) - h
        F += int(k.sum()) - h
        P += int(k.sum())
        G += len(t["gt"])
    return {
        "preds": P, "gt": G, "hits": H, "misses": M, "false_alarms": F,
        "POD": H / max(1, H + M), "FAR": F / max(1, H + F),
        "CSI": H / max(1, H + M + F), "precision": H / max(1, P),
    }


def keys_all(t: dict) -> np.ndarray:
    return np.ones(len(t["cands"]), dtype=bool)


def field(t: dict, name: str, default: float = 0.0) -> np.ndarray:
    return np.array([float(c.get(name, default)) for c in t["cands"]])


def topn_mask(vals: np.ndarray, N: int) -> np.ndarray:
    k = np.zeros(len(vals), dtype=bool)
    if len(vals) and N > 0:
        k[np.argsort(-vals, kind="stable")[:N]] = True
    return k


def dedup_mask(t: dict, vals: np.ndarray, radius_km: float) -> np.ndarray:
    """Post-hoc spatial redundancy suppression: keep the highest-scoring alert
    within radius_km of an already-kept one."""
    n = len(vals)
    keep = np.zeros(n, dtype=bool)
    if n == 0:
        return keep
    lat = np.array([cand_coords(c)[0] for c in t["cands"]])
    lon = np.array([cand_coords(c)[1] for c in t["cands"]])
    DD = dist_matrix(lat, lon, lat, lon)
    kept_idx: List[int] = []
    for i in np.argsort(-vals, kind="stable"):
        if kept_idx and np.any(DD[i, kept_idx] < radius_km):
            continue
        keep[i] = True
        kept_idx.append(int(i))
    return keep


def validate_matcher(per_ts: List[dict], n: int = 8) -> Dict[str, Any]:
    """Prove the vectorised matcher reproduces el.spatial_bipartite_matching."""
    checked = agree = 0
    for t in per_ts[:n]:
        preds = [{"lat": cand_coords(c)[0], "lon": cand_coords(c)[1], "prob": c.get("prob", 0.0)} for c in t["cands"]]
        h, m, f, _, _ = el.spatial_bipartite_matching(preds, t["gt"], tolerance_km=TOL)
        mp, _ = greedy_from_D(t["D"], TOL)
        hh = int(mp.sum())
        checked += 1
        if hh == h and (len(t["gt"]) - hh) == m and (len(preds) - hh) == f:
            agree += 1
    return {"timestamps_checked": checked, "identical": agree, "all_agree": checked == agree}


def probe_annotations() -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for role in ("val", "test"):
        p = FULL / f"{role}_samples.jsonl"
        n_obj = n_full = 0
        offsets: List[float] = []
        obj_keys: set = set()
        with open(p, "r", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                rec = json.loads(line)
                for o in rec.get("objects", []):
                    obj_keys.update(o.keys())
                    n_obj += 1
                    fg, bb = o.get("full_grid_bbox_xyxy"), o.get("bbox_xyxy")
                    if fg and bb and len(fg) == 4 and len(bb) == 4:
                        n_full += 1
                        dx = ((fg[0] + fg[2]) - (bb[0] + bb[2])) / 2.0
                        dy = ((fg[1] + fg[3]) - (bb[1] + bb[3])) / 2.0
                        offsets.append(float(np.hypot(dx, dy) * 1.11))
        out[role] = {
            "objects": n_obj,
            "objects_with_both_bboxes": n_full,
            "object_keys": sorted(obj_keys),
            "centre_offset_km_mean": float(np.mean(offsets)) if offsets else None,
            "centre_offset_km_p99": float(np.percentile(offsets, 99)) if offsets else None,
            "centre_offset_km_max": float(np.max(offsets)) if offsets else None,
        }
    return out


RANK_KEYS: Dict[str, Callable[[dict], float]] = {
    "score": lambda c: float(c.get("score", 0.0)),
    "prob": lambda c: float(c.get("prob", 0.0)),
    "hm_val": lambda c: float(c.get("hm_val", 0.0)),
    "prob_x_hm_unnormalised": lambda c: float(c.get("prob", 0.0)) * float(c.get("hm_val", 0.0)),
}


def main():
    ap = argparse.ArgumentParser(description="Object-level FAR root-cause diagnosis")
    ap.add_argument("--splits", default="dev,benchmark,test")
    ap.add_argument("--out", default="experiments/final/far_diagnosis.json")
    ap.add_argument("--null-reps", type=int, default=20)
    ap.add_argument("--skip-reference-ap", action="store_true")
    args = ap.parse_args()

    report: Dict[str, Any] = {
        "title": "Object-level false-alarm root-cause diagnosis",
        "tolerance_km": TOL,
        "note": "Read-only. No model, threshold or coordinate is changed by this script.",
        "annotation_probe": probe_annotations(),
        "splits": {},
    }

    print("=" * 78)
    print("ANNOTATION / GT-CENTRE PROBE")
    print("=" * 78)
    for role, v in report["annotation_probe"].items():
        print(f"[{role}] objects={v['objects']} with_both_bboxes={v['objects_with_both_bboxes']}")
        print(f"        keys={v['object_keys']}")
        if v["centre_offset_km_mean"] is not None:
            print(f"        bbox_xyxy vs full_grid centre offset: mean={v['centre_offset_km_mean']:.2f} km "
                  f"p99={v['centre_offset_km_p99']:.2f} km max={v['centre_offset_km_max']:.2f} km")

    rng = np.random.default_rng(20261116)

    for name in [s.strip() for s in args.splits.split(",") if s.strip()]:
        if not (D / "final" / f"dense_candidates_{name}.json").exists():
            print(f"[SKIP] {name}: candidate cache missing")
            continue
        raw_ts, per_ts = build_split(name)
        if "matcher_validation" not in report:
            report["matcher_validation"] = validate_matcher(per_ts)
            print(f"\n[VALIDATE] vectorised matcher vs el.spatial_bipartite_matching: "
                  f"{report['matcher_validation']}")
            if not report["matcher_validation"]["all_agree"]:
                raise RuntimeError("matcher mismatch - diagnosis would not be comparable to the published metrics")
        n_ts = len(per_ts)
        tot_gt = sum(len(t["gt"]) for t in per_ts)
        tot_raw = sum(len(t["cands"]) for t in raw_ts)
        tot_gated = sum(len(t["cands"]) for t in per_ts)
        res: Dict[str, Any] = {"n_timestamps": n_ts}

        print("\n" + "=" * 78)
        print(f"SPLIT {name.upper()}  ({n_ts} timestamps)")
        print("=" * 78)

        # ---------------- A. emission budget ----------------
        cnt = np.array([len(t["cands"]) for t in per_ts])
        gcnt = np.array([len(t["gt"]) for t in per_ts])
        base = agg(per_ts, [keys_all(t) for t in per_ts])
        floor = 1.0 - min(tot_gt, tot_gated) / max(1, tot_gated)
        res["emission_budget"] = {
            "total_gt_objects": tot_gt,
            "raw_candidates": tot_raw,
            "post_gating_candidates": tot_gated,
            "gt_per_ts_mean": float(gcnt.mean()),
            "candidates_per_ts_mean": float(cnt.mean()),
            "candidates_per_ts_median": float(np.median(cnt)),
            "candidates_per_ts_p90": float(np.percentile(cnt, 90)),
            "candidates_per_ts_max": int(cnt.max()),
            "emission_ratio": tot_gated / max(1, tot_gt),
            "structural_FAR_floor_perfect_discrimination": float(floor),
            "actual_FAR": base["FAR"],
            "actual_POD": base["POD"],
            "actual_CSI": base["CSI"],
            "preds_needed_for_FAR_50_at_current_hits": int(np.ceil(base["hits"] / 0.5)),
            "preds_needed_for_FAR_70_at_current_hits": int(np.ceil(base["hits"] / 0.3)),
            "zero_gt_timestamps": int((gcnt == 0).sum()),
            "candidates_on_zero_gt_timestamps": int(cnt[gcnt == 0].sum()),
        }
        eb = res["emission_budget"]
        print(f"[A] GT={tot_gt}  raw_cand={tot_raw}  post_gate={tot_gated}  emission ratio={eb['emission_ratio']:.2f}x")
        print(f"    cand/ts mean={eb['candidates_per_ts_mean']:.1f} median={eb['candidates_per_ts_median']:.0f} "
              f"p90={eb['candidates_per_ts_p90']:.0f} max={eb['candidates_per_ts_max']}  |  GT/ts mean={eb['gt_per_ts_mean']:.1f}")
        print(f"    FAR floor even with PERFECT discrimination = {floor:.4f}   actual FAR = {base['FAR']:.4f}")
        print(f"    zero-GT timestamps = {eb['zero_gt_timestamps']}, alerts issued there = {eb['candidates_on_zero_gt_timestamps']}")
        cut = 100 * (1 - eb["preds_needed_for_FAR_50_at_current_hits"] / max(1, tot_gated))
        print(f"    FAR 50% at current hits needs alerts <= {eb['preds_needed_for_FAR_50_at_current_hits']} ({cut:.1f}% cut)")

        # ---------------- B1. tolerance sensitivity ----------------
        tol_curve = {}
        for tol in (15.0, 30.0, 45.0, 60.0, 90.0, 120.0):
            tol_curve[f"{int(tol)}km"] = agg(per_ts, [keys_all(t) for t in per_ts], tol)
        res["tolerance_sensitivity"] = tol_curve
        print("[B1] tolerance sweep (post-gating, dual coords):")
        for k, v in tol_curve.items():
            print(f"     {k:>6}  POD={v['POD']:.4f}  FAR={v['FAR']:.4f}  CSI={v['CSI']:.4f}")

        # ---------------- B2. top-N emission budget, per ranking key ----------------
        budgets: Dict[str, Any] = {}
        for kname, kfn in RANK_KEYS.items():
            rows = {}
            for N in (5, 10, 20, 34, 50, 100, 150, 280):
                keeps = [topn_mask(np.array([kfn(c) for c in t["cands"]]), N) for t in per_ts]
                rows[f"top{N}"] = agg(per_ts, keeps)
            budgets[kname] = rows
        oracle = {}
        for mult in (1.0, 2.0, 3.0):
            keeps = [
                topn_mask(np.array([RANK_KEYS["score"](c) for c in t["cands"]]), int(round(mult * len(t["gt"]))))
                for t in per_ts
            ]
            oracle[f"{mult:g}x_gt_count"] = agg(per_ts, keeps)
        res["topN_budget_by_ranking_key"] = budgets
        res["topN_oracle_uses_split_gt_NOT_an_operating_point"] = oracle
        print("[B2] top-N alerts per timestamp (no GT used):")
        for kname, rows in budgets.items():
            print(f"     ranking key = {kname}")
            for N, v in rows.items():
                print(f"       {N:>6}  preds={v['preds']:6d}  POD={v['POD']:.4f}  FAR={v['FAR']:.4f}  CSI={v['CSI']:.4f}")
        print("     ORACLE N = k x true object count (diagnostic upper bound only):")
        for k, v in oracle.items():
            print(f"       {k:>12}  preds={v['preds']:6d}  POD={v['POD']:.4f}  FAR={v['FAR']:.4f}  CSI={v['CSI']:.4f}")

        # ---------------- B3. fine probability threshold ----------------
        fine = {}
        for tau in np.arange(0.20, 1.001, 0.02):
            keeps = [field(t, "prob") >= tau for t in per_ts]
            fine[f"{tau:.2f}"] = agg(per_ts, keeps)
        res["fine_prob_threshold_sweep"] = fine
        best_tau = max(fine, key=lambda k: fine[k]["CSI"])
        res["fine_prob_threshold_best_csi_tau_on_this_split"] = float(best_tau)
        print(f"[B3] fine prob sweep (step 0.02): best-CSI tau here = {best_tau} "
              f"CSI={fine[best_tau]['CSI']:.4f} FAR={fine[best_tau]['FAR']:.4f} POD={fine[best_tau]['POD']:.4f}")
        for tau in ("0.50", "0.64", "0.80", "0.90", "0.96", "1.00"):
            if tau in fine:
                v = fine[tau]
                print(f"     tau={tau}  preds={v['preds']:6d}  POD={v['POD']:.4f}  FAR={v['FAR']:.4f}  CSI={v['CSI']:.4f}")

        # ---------------- C1. false alarm geometry ----------------
        all_c = [c for t in per_ts for c in t["cands"]]
        labels = np.zeros(len(all_c), dtype=np.int8)
        off = 0
        fa_nearest: List[float] = []
        fa_zero_gt = 0
        for t in per_ts:
            mp, _ = greedy_from_D(t["D"], TOL)
            labels[off : off + len(t["cands"])] = mp.astype(np.int8)
            Dm = t["D"]
            for i in np.nonzero(~mp)[0]:
                if Dm.shape[1] == 0:
                    fa_zero_gt += 1
                    fa_nearest.append(np.inf)
                else:
                    fa_nearest.append(float(Dm[i].min()))
            off += len(t["cands"])
        fa_nearest_arr = np.asarray(fa_nearest)
        finite = fa_nearest_arr[np.isfinite(fa_nearest_arr)]
        buckets = {
            "le_30km_lost_bilateral_contest": int((finite <= 30).sum()),
            "30_45km": int(((finite > 30) & (finite <= 45)).sum()),
            "45_60km": int(((finite > 45) & (finite <= 60)).sum()),
            "60_90km": int(((finite > 60) & (finite <= 90)).sum()),
            "90_150km": int(((finite > 90) & (finite <= 150)).sum()),
            "gt_150km": int((finite > 150).sum()),
            "no_gt_in_timestamp": fa_zero_gt,
        }
        res["false_alarm_geometry"] = {
            "total_false_alarms": int((labels == 0).sum()),
            "nearest_gt_distance_buckets": buckets,
            "median_nearest_gt_km": float(np.median(finite)) if len(finite) else None,
            "mean_nearest_gt_km": float(np.mean(finite)) if len(finite) else None,
        }
        print(f"[C1] false alarms = {int((labels == 0).sum())}; distance to nearest GT:")
        for k, v in buckets.items():
            print(f"     {k:>34}: {v:6d}  ({100 * v / max(1, len(fa_nearest_arr)):.1f}%)")

        # ---------------- C2. field discriminative power ----------------
        fields = ["prob", "score", "hm_val", "bt13_local", "delta_bt13_local", "b10_b13_local", "b11_b13_local"]
        aucs = {}
        for fld in fields:
            v = np.array([float(c.get(fld, np.nan)) for c in all_c])
            ok = np.isfinite(v)
            aucs[fld] = rank_auc(v[ok], labels[ok]) if ok.any() else float("nan")
        res["tp_vs_fa_auc_by_field"] = aucs
        print("[C2] TP-vs-FA AUC of every stored field (0.5 = no information):")
        for k, v in aucs.items():
            print(f"     {k:>20}: {v:.4f}")

        # ---------------- C3. AP by ranking key + null ----------------
        key_vals = {kn: [np.array([kf(c) for c in t["cands"]]) for t in per_ts] for kn, kf in RANK_KEYS.items()}
        key_vals["minus_bt13_local"] = [-field(t, "bt13_local") for t in per_ts]
        key_vals["minus_delta_bt13_local"] = [-field(t, "delta_bt13_local") for t in per_ts]
        ap_variants = {}
        for kn, kv in key_vals.items():
            a, tp, fp = ap_global(per_ts, kv)
            ap_variants[kn] = {"AP": a, "TP": tp, "FP": fp}
        res["ap_by_ranking_key"] = ap_variants
        print("[C3] AP over all candidates (one-to-one 30 km):")
        for k, v in ap_variants.items():
            print(f"     {k:>28}: AP={v['AP']:.5f}  TP={v['TP']}")

        nulls = []
        for _ in range(args.null_reps):
            kv = []
            for t in per_ts:
                s = field(t, "score").copy()
                rng.shuffle(s)
                kv.append(s)
            nulls.append(ap_global(per_ts, kv)[0])
        res["ap_null_shuffled_within_timestamp"] = {
            "reps": args.null_reps, "mean": float(np.mean(nulls)), "std": float(np.std(nulls)),
            "min": float(np.min(nulls)), "max": float(np.max(nulls)),
        }
        print(f"     {'NULL shuffled within ts':>28}: AP={np.mean(nulls):.5f} +/- {np.std(nulls):.5f}")

        if not args.skip_reference_ap:
            ref = ot.compute_object_ap(
                {t["uid"]: t["cands"] for t in per_ts},
                {t["uid"]: t["gt"] for t in per_ts},
                lambda p: True, TOL,
            )
            res["ap_reference_eval_operational_tiers"] = {"AP": ref[0], "TP": ref[1], "FP": ref[2]}
            res["ap_implementation_agrees"] = bool(abs(ap_variants["score"]["AP"] - ref[0]) < 1e-9)
            print(f"     reference ot.compute_object_ap: AP={ref[0]:.5f}  agrees={res['ap_implementation_agrees']}")

        # ---------------- B4. per-timestamp normalisation ----------------
        hmax = []
        for t in per_ts:
            p, s, h = field(t, "prob"), field(t, "score"), field(t, "hm_val")
            ok = (s > 1e-9) & (p > 1e-9)
            hmax.append(float(np.median(h[ok] * p[ok] / s[ok])) if ok.any() else np.nan)
        hmax = np.asarray(hmax)
        okh = np.isfinite(hmax)
        res["per_timestamp_heatmap_max"] = {
            "recovered_Hmax_median": float(np.nanmedian(hmax)),
            "recovered_Hmax_p10": float(np.nanpercentile(hmax, 10)),
            "recovered_Hmax_p90": float(np.nanpercentile(hmax, 90)),
            "corr_Hmax_vs_candidate_count": float(np.corrcoef(hmax[okh], cnt[okh])[0, 1]) if okh.sum() > 2 else None,
            "corr_Hmax_vs_gt_count": float(np.corrcoef(hmax[okh], gcnt[okh])[0, 1]) if okh.sum() > 2 else None,
            "Hmax_median_on_zero_gt_ts": float(np.nanmedian(hmax[gcnt == 0])) if (gcnt == 0).any() else None,
            "Hmax_median_on_positive_gt_ts": float(np.nanmedian(hmax[gcnt > 0])) if (gcnt > 0).any() else None,
        }
        print("[B4] per-timestamp heatmap normalisation (P4):")
        for k, v in res["per_timestamp_heatmap_max"].items():
            print(f"     {k:>40}: {v}")

        # ---------------- B5. gate strength ----------------
        n_sup = tot_raw - tot_gated
        sup_tp = 0
        for tr, tg in zip(raw_ts, per_ts):
            mp, _ = greedy_from_D(tr["D"], TOL)
            sup_tp += int((mp & ~tg["keep"]).sum())
        gate_fields = {}
        for fld in ("delta_bt13_local", "b10_b13_local", "bt13_local"):
            v = np.array([float(c.get(fld, np.nan)) for c in all_c])
            okf = np.isfinite(v)
            tp_m, fa_m = okf & (labels == 1), okf & (labels == 0)
            gate_fields[fld] = {
                "TP_median": float(np.median(v[tp_m])) if tp_m.any() else None,
                "FA_median": float(np.median(v[fa_m])) if fa_m.any() else None,
                "TP_p10": float(np.percentile(v[tp_m], 10)) if tp_m.any() else None,
                "FA_p10": float(np.percentile(v[fa_m], 10)) if fa_m.any() else None,
            }
        res["gate_strength"] = {
            "suppressed": n_sup,
            "suppressed_pct_of_raw": 100.0 * n_sup / max(1, tot_raw),
            "true_objects_lost_to_gate": sup_tp,
            "field_separation": gate_fields,
        }
        print(f"[B5] gate suppressed {n_sup}/{tot_raw} = {100 * n_sup / max(1, tot_raw):.2f}%, true objects lost = {sup_tp}")
        for k, v in gate_fields.items():
            print(f"     {k:>20}: TP median={v['TP_median']}  FA median={v['FA_median']}")

        # ---------------- B7. spatial redundancy / NMS radius ----------------
        dedup = {}
        for radius in (30.0, 45.0, 60.0, 90.0):
            keeps = [dedup_mask(t, field(t, "score"), radius) for t in per_ts]
            dedup[f"{int(radius)}km"] = agg(per_ts, keeps)
        res["post_hoc_dedup_radius"] = dedup
        print("[B7] post-hoc dedup radius (current NMS already enforces 30 km):")
        for k, v in dedup.items():
            print(f"     {k:>6}  preds={v['preds']:6d}  POD={v['POD']:.4f}  FAR={v['FAR']:.4f}  CSI={v['CSI']:.4f}")

        # ---------------- B6. calibration bypass ----------------
        tag = "test" if name == "test" else ("dev" if name == "dev" else "bench")
        npz = D / "final" / f"cinet_v2_{tag}_predictions.npz"
        dense_prob = np.array([float(c.get("prob", 0.0)) for c in all_c])
        cal: Dict[str, Any] = {
            "dense_candidate_prob_mean": float(dense_prob.mean()),
            "dense_candidate_prob_quantiles": {str(q): float(np.percentile(dense_prob, q)) for q in (5, 25, 50, 75, 95)},
        }
        if npz.exists():
            z = np.load(npz, allow_pickle=True)
            p_raw = np.asarray(z["p_raw"], dtype=np.float64)
            p_cal = np.asarray(z["p_cal"], dtype=np.float64)
            y = np.asarray(z["y"])
            cal["window_level_n"] = int(len(p_raw))
            cal["window_level_pos_rate"] = float(y.mean())
            cal["window_p_raw_quantiles"] = {str(q): float(np.percentile(p_raw, q)) for q in (5, 25, 50, 75, 95)}
            cal["window_p_cal_quantiles"] = {str(q): float(np.percentile(p_cal, q)) for q in (5, 25, 50, 75, 95)}
            cal["frac_windows_p_raw_ge_0.20"] = float((p_raw >= 0.20).mean())
            cal["frac_windows_p_cal_ge_tau_dev"] = float((p_cal >= TAU_DEV).mean())
            o = np.argsort(p_raw)
            xs, idx = np.unique(p_raw[o], return_index=True)
            ys = p_cal[o][idx]
            mapped = np.interp(dense_prob, xs, ys)
            cal["dense_prob_mapped_through_window_isotonic_quantiles"] = {
                str(q): float(np.percentile(mapped, q)) for q in (5, 25, 50, 75, 95)
            }
            cal["frac_dense_candidates_surviving_calibrated_tau_dev"] = float((mapped >= TAU_DEV).mean())
            cal["caveat"] = (
                "Dense prob is the Hann-blended CNN-only sigmoid; the isotonic map was fitted on the "
                "CNN+GBDT window blend. Indicative, not an exact recalibration."
            )
        res["calibration_bypass"] = cal
        print("[B6] calibration bypass (dense path uses RAW sigmoid, never the dev isotonic map):")
        for k, v in cal.items():
            print(f"     {k}: {v}")

        report["splits"][name] = res

    out = PROJECT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, allow_nan=True))
    print(f"\n[OUTPUT] {out}")


if __name__ == "__main__":
    main()