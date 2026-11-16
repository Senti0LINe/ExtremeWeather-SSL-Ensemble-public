"""Final evaluation of the 5-seed TTA ensemble on the frozen test subset."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import lightgbm as lgb
from sklearn.isotonic import IsotonicRegression

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))
from ewci import data, metrics as M  # noqa: E402

D = PROJECT / "experiments"
FULL = Path("/data1/group3/DATASAT/experiments/ci_det_v2/annotations_full")


def load(path):
    z = np.load(path, allow_pickle=True)
    return z["p"], {(str(u), int(r), int(c)): i for i, (u, r, c) in enumerate(zip(z["uid"], z["row"], z["col"]))}


def main():
    uids = set(json.loads((PROJECT / "splits_v1.json").read_text())["dev"]["early_stop_uids"])
    dev_patches = [p for p in data.load_patches_from(FULL, "val") if p["sample_uid"] in uids]
    dk = [(p["sample_uid"], int(p["row"]), int(p["col"])) for p in dev_patches]
    y_dev = np.array([data.has_ci(p) for p in dev_patches])

    devs = []
    for s in range(5):
        p, pos = load(D / ("p1_seed%d_dev_tta.npz" % s))
        devs.append(p[[pos[k] for k in dk]])
    p_cnn_dev = np.mean(devs, 0)

    f_te = np.load(D / "final/features_test_eval.npz", allow_pickle=True)
    te_keys = [(str(u), int(r), int(c)) for u, r, c in zip(f_te["uid"], f_te["row"], f_te["col"])]
    tests = []
    for s in range(5):
        p, pos = load(D / ("final/p1_seed%d_test_tta.npz" % s))
        tests.append(p[[pos[k] for k in te_keys]])
    p_cnn_te = np.mean(tests, 0)
    y_te = f_te["y"]

    f_dev = np.load(D / "p0_gbdt/features_dev.npz", allow_pickle=True)
    posd = {(str(u), int(r), int(c)): i for i, (u, r, c) in enumerate(zip(f_dev["uid"], f_dev["row"], f_dev["col"]))}
    idx = np.array([posd[k] for k in dk])
    tr = np.load(D / "p0_gbdt/features_train.npz", allow_pickle=True)
    prm = dict(n_estimators=700, learning_rate=0.03, num_leaves=63, min_child_samples=40,
               subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
               class_weight="balanced", n_jobs=8, verbose=-1)
    gb = [lgb.LGBMClassifier(**prm, random_state=s).fit(tr["X"], tr["y"]) for s in (0, 1, 2)]
    p_gbdt_dev = np.mean([m.predict_proba(f_dev["X"][idx])[:, 1] for m in gb], 0)
    p_gbdt_te = np.mean([m.predict_proba(f_te["X"])[:, 1] for m in gb], 0)

    best = (0.9, -1)
    for w in np.linspace(0, 1, 21):
        a = M.ranking(y_dev, w * p_cnn_dev + (1 - w) * p_gbdt_dev)["ROC_AUC"]
        if a > best[1]:
            best = (float(w), a)
    w = best[0]
    p_dev = w * p_cnn_dev + (1 - w) * p_gbdt_dev
    p_te = w * p_cnn_te + (1 - w) * p_gbdt_te
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
    cases = f_te["case"]
    for c in sorted(set(cases.tolist())):
        m = cases == c
        if m.sum() > 30 and len(np.unique(y_te[m])) > 1:
            per_case[str(c)] = {"n": int(m.sum()), "pos": int(y_te[m].sum()),
                                "AUC": M.ranking(y_te[m], p_te_cal[m])["ROC_AUC"],
                                "CSI": M.skill(y_te[m], p_te_cal[m], tau)["CSI"]}

    res = {"variant": "seed5_tta", "weight_cnn": w,
           "test": {"n": int(len(y_te)), "pos": int(y_te.sum()),
                    "ROC_AUC": M.ranking(y_te, p_te_cal)["ROC_AUC"],
                    "PR_AUC": M.ranking(y_te, p_te_cal)["PR_AUC"],
                    "tau_dev": float(tau), "at_tau": M.skill(y_te, p_te_cal, tau), "at": at,
                    "ECE_cal": M.reliability(y_te, p_te_cal)[3],
                    "Brier_cal": M.skill(y_te, p_te_cal, 0.5)["Brier"]},
           "dev": {"AUC": M.ranking(y_dev, p_dev)["ROC_AUC"]}, "per_case": per_case}
    json.dump(res, open(D / "final/test_metrics_seed5.json", "w"), indent=2)
    print(json.dumps({"AUC": res["test"]["ROC_AUC"], "PR": res["test"]["PR_AUC"],
                      "CSI": res["test"]["at_tau"]["CSI"], "POD": res["test"]["at_tau"]["POD"],
                      "FAR": res["test"]["at_tau"]["FAR"], "w": w, "tau": tau,
                      "ECE": res["test"]["ECE_cal"], "Brier": res["test"]["Brier_cal"],
                      "at": at}, indent=1))


if __name__ == "__main__":
    main()
