"""Fit and cache the ensemble weight, isotonic calibration and threshold (dev only)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import lightgbm as lgb
from sklearn.isotonic import IsotonicRegression

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))
from ewci import data, metrics as M  # noqa: E402

D = PROJECT / "experiments"
FULL = Path("/data1/group3/DATASAT/experiments/ci_det_v2/annotations_full")


def main():
    uids = set(json.loads((PROJECT / "splits_v1.json").read_text())["dev"]["early_stop_uids"])
    dev_patches = [p for p in data.load_patches_from(FULL, "val") if p["sample_uid"] in uids]
    keys = [(p["sample_uid"], int(p["row"]), int(p["col"])) for p in dev_patches]
    y = np.array([data.has_ci(p) for p in dev_patches])

    cnn = np.load(D / "p1_cnn/probs.npz")
    p_cnn = cnn["p_dev"]
    seed_files = [D / ("p1_seed%d_dev_tta.npz" % s) for s in range(1, 5)]
    if all(f.exists() for f in seed_files):
        arrs = []
        for f in [D / "p1_cnn/probs.npz"] + seed_files:
            z = np.load(f, allow_pickle=True)
            if "p" in z.files:  # TTA files
                pos = {(str(u), int(r), int(c)): i for i, (u, r, c) in enumerate(zip(z["uid"], z["row"], z["col"]))}
                arrs.append(z["p"][np.array([pos[k] for k in keys])])
            else:
                arrs.append(z["p_dev"])
        p_cnn = np.mean(arrs, 0)
        print("cnn dev probs from", len(arrs), "models")

    f_dev = np.load(D / "p0_gbdt/features_dev.npz", allow_pickle=True)
    posd = {(str(u), int(r), int(c)): i for i, (u, r, c) in enumerate(zip(f_dev["uid"], f_dev["row"], f_dev["col"]))}
    idx = np.array([posd[k] for k in keys])
    tr = np.load(D / "p0_gbdt/features_train.npz", allow_pickle=True)
    prm = dict(n_estimators=700, learning_rate=0.03, num_leaves=63, min_child_samples=40,
               subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
               class_weight="balanced", n_jobs=8, verbose=-1)
    gb = [lgb.LGBMClassifier(**prm, random_state=s).fit(tr["X"], tr["y"]) for s in (0, 1, 2)]
    joblib.dump(gb, D / "final/gbdt_cache.pkl")
    p_gbdt = np.mean([m.predict_proba(f_dev["X"][idx])[:, 1] for m in gb], 0)

    best = (0.75, -1)
    for w in np.linspace(0, 1, 21):
        a = M.ranking(y, w * p_cnn + (1 - w) * p_gbdt)["ROC_AUC"]
        if a > best[1]:
            best = (float(w), a)
    w = best[0]
    p = w * p_cnn + (1 - w) * p_gbdt
    iso = IsotonicRegression(out_of_bounds="clip").fit(p, y)
    tau, _ = M.best_threshold(y, iso.predict(p), "max_csi")
    json.dump({"x": [float(v) for v in iso.X_thresholds_],
               "y": [float(v) for v in iso.y_thresholds_],
               "w": w, "tau": float(tau), "dev_auc": float(M.ranking(y, p)["ROC_AUC"]),
               "n_models": int(len(np.atleast_1d(p_cnn)))},
              open(D / "final/calib.json", "w"), indent=2)
    print("w", w, "tau", round(tau, 3), "dev AUC", round(M.ranking(y, p)["ROC_AUC"], 4))


if __name__ == "__main__":
    main()
