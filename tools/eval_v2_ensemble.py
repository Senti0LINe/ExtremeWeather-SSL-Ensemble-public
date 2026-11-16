"""Ensemble evaluation for CINet_v2 5-Seed TTA + GBDT on Dev, Benchmark, and Held-out Test Set.

Includes calibration via isotonic regression and export of 28x28 convective cell center heatmaps.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
from sklearn.isotonic import IsotonicRegression

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))
from ewci import data, metrics as M  # noqa: E402

D = PROJECT / "experiments"
FULL = Path("/data1/group3/DATASAT/experiments/ci_det_v2/annotations_full")


def load_npz(path: Path):
    z = np.load(path, allow_pickle=True)
    p = z["p"]
    hm_max = z["hm_max"] if "hm_max" in z else None
    keys = [
        (str(u), int(r), int(c))
        for u, r, c in zip(z["uid"], z["row"], z["col"])
    ]
    return p, hm_max, keys


def main():
    print("==================================================")
    print("Evaluating 5-Seed CINet_v2 TTA + GBDT Heterogeneous Ensemble")
    print("==================================================")

    # 1. Load Dev Ground Truth & Keys
    uids_dev = set(
        json.loads((PROJECT / "splits_v1.json").read_text())["dev"][
            "early_stop_uids"
        ]
    )
    dev_patches = [
        p
        for p in data.load_patches_from(FULL, "val")
        if p["sample_uid"] in uids_dev
    ]
    dk = [(p["sample_uid"], int(p["row"]), int(p["col"])) for p in dev_patches]
    y_dev = np.array([data.has_ci(p) for p in dev_patches], dtype=np.float32)

    # 2. Load 5-Seed Dev TTA Predictions
    dev_preds = []
    dev_hms = []
    for s in range(5):
        p, hm_max, keys = load_npz(D / f"p2_seed{s}_dev_tta.npz")
        pos = {k: i for i, k in enumerate(keys)}
        idx = [pos[k] for k in dk]
        dev_preds.append(p[idx])
        if hm_max is not None:
            dev_hms.append(hm_max[idx])
    p_cnn_dev = np.mean(dev_preds, axis=0)
    hm_cnn_dev = np.mean(dev_hms, axis=0) if dev_hms else None

    # 3. Load 5-Seed Benchmark TTA Predictions
    bench_patches = data.benchmark_patches()
    bk = [
        (p["sample_uid"], int(p["row"]), int(p["col"])) for p in bench_patches
    ]
    y_bench = np.array(
        [data.has_ci(p) for p in bench_patches], dtype=np.float32
    )

    bench_preds = []
    bench_hms = []
    for s in range(5):
        p, hm_max, keys = load_npz(D / f"p2_seed{s}_bench_tta.npz")
        pos = {k: i for i, k in enumerate(keys)}
        idx = [pos[k] for k in bk]
        bench_preds.append(p[idx])
        if hm_max is not None:
            bench_hms.append(hm_max[idx])
    p_cnn_bench = np.mean(bench_preds, axis=0)
    hm_cnn_bench = np.mean(bench_hms, axis=0) if bench_hms else None

    # 4. Load 5-Seed Test TTA Predictions
    f_te = np.load(D / "final/features_test_eval.npz", allow_pickle=True)
    te_keys = [
        (str(u), int(r), int(c))
        for u, r, c in zip(f_te["uid"], f_te["row"], f_te["col"])
    ]
    y_test = f_te["y"]

    test_preds = []
    test_hms = []
    for s in range(5):
        p, hm_max, keys = load_npz(D / f"final/p2_seed{s}_test_tta.npz")
        pos = {k: i for i, k in enumerate(keys)}
        idx = [pos[k] for k in te_keys]
        test_preds.append(p[idx])
        if hm_max is not None:
            test_hms.append(hm_max[idx])
    p_cnn_test = np.mean(test_preds, axis=0)
    hm_cnn_test = np.mean(test_hms, axis=0) if test_hms else None

    # 5. Fit 3-Seed GBDT Baseline on Dev and Predict
    print("[GBDT] Fitting LightGBM models...")
    f_dev = np.load(D / "p0_gbdt/features_dev.npz", allow_pickle=True)
    f_val = np.load(D / "p0_gbdt/features_val.npz", allow_pickle=True)
    tr = np.load(D / "p0_gbdt/features_train.npz", allow_pickle=True)

    posd = {
        (str(u), int(r), int(c)): i
        for i, (u, r, c) in enumerate(
            zip(f_dev["uid"], f_dev["row"], f_dev["col"])
        )
    }
    idx_dev_gbdt = np.array([posd[k] for k in dk])

    lgb_params = dict(
        n_estimators=700,
        learning_rate=0.03,
        num_leaves=63,
        min_child_samples=40,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        class_weight="balanced",
        n_jobs=8,
        verbose=-1,
    )
    gb_models = [
        lgb.LGBMClassifier(**lgb_params, random_state=s).fit(tr["X"], tr["y"])
        for s in (0, 1, 2)
    ]
    p_gbdt_dev = np.mean(
        [m.predict_proba(f_dev["X"][idx_dev_gbdt])[:, 1] for m in gb_models],
        axis=0,
    )
    p_gbdt_bench = np.mean(
        [m.predict_proba(f_val["X"])[:, 1] for m in gb_models], axis=0
    )
    p_gbdt_test = np.mean(
        [m.predict_proba(f_te["X"])[:, 1] for m in gb_models], axis=0
    )

    # 6. Optimize Ensemble Weight w strictly on Dev
    print("[ENSEMBLE] Finding optimal weight w on Dev set...")
    best_w, best_dev_auc = 0.9, -1.0
    for w in np.linspace(0.0, 1.0, 41):
        p_blend = w * p_cnn_dev + (1.0 - w) * p_gbdt_dev
        auc = M.ranking(y_dev, p_blend)["ROC_AUC"]
        if auc > best_dev_auc:
            best_w, best_dev_auc = float(w), float(auc)
    print(f"  --> Optimal w_cnn: {best_w:.2f} (Dev ROC-AUC: {best_dev_auc:.4f})")

    p_dev_blend = best_w * p_cnn_dev + (1.0 - best_w) * p_gbdt_dev
    p_bench_blend = best_w * p_cnn_bench + (1.0 - best_w) * p_gbdt_bench
    p_test_blend = best_w * p_cnn_test + (1.0 - best_w) * p_gbdt_test

    # 7. Probability Calibration via Isotonic Regression on Dev strictly
    print("[CALIBRATION] Fitting Isotonic Regression on Dev...")
    iso = IsotonicRegression(out_of_bounds="clip").fit(p_dev_blend, y_dev)
    p_dev_cal = iso.predict(p_dev_blend)
    p_bench_cal = iso.predict(p_bench_blend)
    p_test_cal = iso.predict(p_test_blend)

    # 8. Optimal Threshold Selection on Dev maximizing CSI
    tau_dev, _ = M.best_threshold(y_dev, p_dev_cal, "max_csi")
    print(f"  --> Optimal decision threshold tau (Dev max CSI): {tau_dev:.4f}")

    # 9. Evaluate Benchmark Metrics
    bench_rk = M.ranking(y_bench, p_bench_cal)
    bench_sk = M.skill(y_bench, p_bench_cal, tau_dev)
    bench_rel = M.reliability(y_bench, p_bench_cal)

    # 10. Evaluate Held-out Test Set Metrics
    test_rk = M.ranking(y_test, p_test_cal)
    test_sk = M.skill(y_test, p_test_cal, tau_dev)
    test_rel = M.reliability(y_test, p_test_cal)

    # Compute operational fixed-POFD points on Test Set
    order = np.argsort(p_test_cal)[::-1]
    P = y_test[order]
    tp = np.cumsum(P)
    fp = np.cumsum(1.0 - P)
    pos_n = float(y_test.sum())
    neg_n = float(len(y_test) - pos_n)
    pofd = fp / neg_n
    at_pofd = {}
    for tgt in (0.10, 0.20):
        k = min(int(np.searchsorted(pofd, tgt)), len(tp) - 1)
        at_pofd[f"pofd_{int(tgt * 100)}"] = {
            "POD": float(tp[k] / pos_n),
            "CSI": float(tp[k] / (tp[k] + fp[k] + (pos_n - tp[k]))),
        }

    results = {
        "model": "CINet_v2_5seed_TTA_GBDT",
        "w_cnn": best_w,
        "tau_dev": float(tau_dev),
        "dev": {
            "n": len(y_dev),
            "pos": int(y_dev.sum()),
            "ROC_AUC": float(best_dev_auc),
            "PR_AUC": float(M.ranking(y_dev, p_dev_cal)["PR_AUC"]),
            "CSI": float(M.skill(y_dev, p_dev_cal, tau_dev)["CSI"]),
        },
        "benchmark": {
            "n": len(y_bench),
            "pos": int(y_bench.sum()),
            "ROC_AUC": float(bench_rk["ROC_AUC"]),
            "PR_AUC": float(bench_rk["PR_AUC"]),
            "CSI": float(bench_sk["CSI"]),
            "POD": float(bench_sk["POD"]),
            "FAR": float(bench_sk["FAR"]),
            "ECE": float(bench_rel[3]),
            "Brier": float(bench_sk["Brier"]),
        },
        "test": {
            "n": len(y_test),
            "pos": int(y_test.sum()),
            "ROC_AUC": float(test_rk["ROC_AUC"]),
            "PR_AUC": float(test_rk["PR_AUC"]),
            "CSI": float(test_sk["CSI"]),
            "POD": float(test_sk["POD"]),
            "FAR": float(test_sk["FAR"]),
            "ECE": float(test_rel[3]),
            "Brier": float(test_sk["Brier"]),
            "at_fixed_pofd": at_pofd,
        },
    }

    out_json = D / "final/cinet_v2_ensemble_metrics.json"
    with open(out_json, "w") as fh:
        json.dump(results, fh, indent=2)

    # Save calibrated predictions & convective cell heatmaps.
    # Dev and benchmark archives are required so that the object-level operating
    # points (alpha, tau_obj) can be selected on dev alone (EXPERIMENTS E14).
    np.savez_compressed(
        D / "final/cinet_v2_test_predictions.npz",
        p_cal=p_test_cal,
        p_raw=p_test_blend,
        hm_max=hm_cnn_test,
        y=y_test,
        uid=f_te["uid"],
        row=f_te["row"],
        col=f_te["col"],
    )
    np.savez_compressed(
        D / "final/cinet_v2_dev_predictions.npz",
        p_cal=p_dev_cal,
        p_raw=p_dev_blend,
        hm_max=hm_cnn_dev,
        y=y_dev,
        uid=np.array([k[0] for k in dk]),
        row=np.array([k[1] for k in dk]),
        col=np.array([k[2] for k in dk]),
    )
    np.savez_compressed(
        D / "final/cinet_v2_bench_predictions.npz",
        p_cal=p_bench_cal,
        p_raw=p_bench_blend,
        hm_max=hm_cnn_bench,
        y=y_bench,
        uid=np.array([k[0] for k in bk]),
        row=np.array([k[1] for k in bk]),
        col=np.array([k[2] for k in bk]),
    )

    print("\n" + "=" * 50)
    print("FINAL BENCHMARK AND TEST EVALUATION RESULTS:")
    print("=" * 50)
    print(json.dumps(results, indent=2))
    print(f"\n[OUTPUT] Saved summary metrics to {out_json}")


if __name__ == "__main__":
    main()
