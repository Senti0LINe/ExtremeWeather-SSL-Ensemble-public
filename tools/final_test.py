"""Final evaluation on the frozen test subset.

Two variants can be scored:
  * plain   : 51 physical features + CNN
  * era5    : the same plus 7 ERA5 environment features
The ensemble weight, calibration and threshold are always fitted on dev only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import lightgbm as lgb

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))
from ewci import data, metrics as M  # noqa: E402

FULL = Path("/data1/group3/DATASAT/experiments/ci_det_v2/annotations_full")
D = PROJECT / "experiments"


def keys_of(z):
    return [(str(u), int(r), int(c)) for u, r, c in zip(z["uid"], z["row"], z["col"])]


def align(base_path, era_path=None):
    b = np.load(base_path, allow_pickle=True)
    X = b["X"]
    if era_path is not None:
        e = np.load(era_path, allow_pickle=True)
        pos = {k: i for i, k in enumerate(keys_of(e))}
        idx = np.array([pos[k] for k in keys_of(b)])
        X = np.hstack([X, e["X"][idx]])
    return X, b["y"], keys_of(b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--era5", action="store_true")
    ap.add_argument("--cnn-dev", default="experiments/p1_cnn/probs.npz")
    ap.add_argument("--cnn-test", default="experiments/final/cnn_test_eval.npz")
    ap.add_argument("--out", default="experiments/final")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    tag = "_era5" if args.era5 else ""
    etr = str(D / "era5/era5_train.npz") if args.era5 else None
    edv = str(D / "era5/era5_dev.npz") if args.era5 else None
    ebk = str(D / "era5/era5_test_eval.npz") if args.era5 else None

    # dev
    sp = json.loads((PROJECT / "splits_v1.json").read_text())
    uids = set(sp["dev"]["early_stop_uids"])
    dev_patches = [p for p in data.load_patches_from(FULL, "val") if p["sample_uid"] in uids]
    dev_keys = [(p["sample_uid"], int(p["row"]), int(p["col"])) for p in dev_patches]
    y_dev = np.array([data.has_ci(p) for p in dev_patches])
    Xd, _, kd = align(D / "p0_gbdt/features_dev.npz", edv)
    pos = {k: i for i, k in enumerate(kd)}
    idx = np.array([pos[k] for k in dev_keys])
    Xd, y_dev = Xd[idx], y_dev

    Xtr, ytr, _ = align(D / "p0_gbdt/features_train.npz", etr)
    Xte, y_te, kte = align(D / "final/features_test_eval.npz", ebk)

    cnn = np.load(args.cnn_dev)
    cnn_te = np.load(args.cnn_test)
    pos_te = {k: i for i, k in enumerate(keys_of(cnn_te))}
    cnn_test = cnn_te["p"][np.array([pos_te[k] for k in kte])]
    cnn_dev = cnn["p_dev"]

    prm = dict(n_estimators=700, learning_rate=0.03, num_leaves=63, min_child_samples=40,
               subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
               class_weight="balanced", n_jobs=8, verbose=-1)
    pg_dev, pg_te = [], []
    for s in (0, 1, 2):
        m = lgb.LGBMClassifier(**prm, random_state=s).fit(Xtr, ytr)
        pg_dev.append(m.predict_proba(Xd)[:, 1]); pg_te.append(m.predict_proba(Xte)[:, 1])
    pg_dev, pg_te = np.mean(pg_dev, 0), np.mean(pg_te, 0)

    best = (0.75, -1.0)
    for w in np.linspace(0, 1, 21):
        a = M.ranking(y_dev, w * cnn_dev + (1 - w) * pg_dev)["ROC_AUC"]
        if a > best[1]:
            best = (float(w), a)
    w = best[0]
    p_dev = w * cnn_dev + (1 - w) * pg_dev
    p_te = w * cnn_test + (1 - w) * pg_te

    from sklearn.isotonic import IsotonicRegression
    iso = IsotonicRegression(out_of_bounds="clip").fit(p_dev, y_dev)
    p_te_cal = iso.predict(p_te)
    tau, _ = M.best_threshold(y_dev, iso.predict(p_dev), "max_csi")

    order = np.argsort(p_te_cal)[::-1]
    P = y_te[order]; tp = np.cumsum(P); fp = np.cumsum(1 - P)
    pos_n = float(y_te.sum()); neg_n = float(len(y_te) - y_te.sum())
    pofd = fp / neg_n
    at = {}
    for tgt in (0.10, 0.20):
        k = min(int(np.searchsorted(pofd, tgt)), len(tp) - 1)
        at["pofd_%d" % int(tgt * 100)] = {"POD": float(tp[k] / pos_n),
                                         "CSI": float(tp[k] / (tp[k] + fp[k] + (pos_n - tp[k])))}

    per_case = {}
    cases = np.load(D / "final/features_test_eval.npz", allow_pickle=True)["case"]
    for c in sorted(set(cases.tolist())):
        m = cases == c
        if m.sum() > 30 and len(np.unique(y_te[m])) > 1:
            per_case[str(c)] = {"n": int(m.sum()), "pos": int(y_te[m].sum()),
                                "AUC": M.ranking(y_te[m], p_te_cal[m])["ROC_AUC"],
                                "CSI": M.skill(y_te[m], p_te_cal[m], tau)["CSI"]}

    res = {"variant": "era5" if args.era5 else "plain", "weight_cnn": w,
           "test": {"n": int(len(y_te)), "pos": int(y_te.sum()),
                    "ROC_AUC": M.ranking(y_te, p_te_cal)["ROC_AUC"],
                    "PR_AUC": M.ranking(y_te, p_te_cal)["PR_AUC"],
                    "tau_dev": float(tau),
                    "at_tau": M.skill(y_te, p_te_cal, tau), "at": at,
                    "ECE_cal": M.reliability(y_te, p_te_cal)[3],
                    "Brier_cal": M.skill(y_te, p_te_cal, 0.5)["Brier"]},
           "dev": {"AUC": M.ranking(y_dev, p_dev)["ROC_AUC"]},
           "per_case": per_case}
    json.dump(res, open(out / ("test_metrics_%s.json" % res["variant"]), "w"), indent=2)
    print(json.dumps({k: v for k, v in res["test"].items() if k != "at_tau"}, indent=1))


if __name__ == "__main__":
    main()
