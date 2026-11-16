"""Does adding real ERA5 environment features help? (controlled ablation)"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import lightgbm as lgb

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))
from ewci import metrics as M  # noqa: E402

D = PROJECT / "experiments"


def keys(z):
    return [(str(u), int(r), int(c)) for u, r, c in zip(z["uid"], z["row"], z["col"])]


def load(base, era):
    b = np.load(base, allow_pickle=True)
    e = np.load(era, allow_pickle=True)
    pos = {k: i for i, k in enumerate(keys(e))}
    idx = np.array([pos[k] for k in keys(b)])
    return b["X"], e["X"][idx], b["y"]


def main():
    Xtr, Etr, ytr = load(D / "p0_gbdt/features_train.npz", D / "era5/era5_train.npz")
    Xdv, Edv, ydv = load(D / "p0_gbdt/features_dev.npz", D / "era5/era5_dev.npz")
    Xbk, Ebk, ybk = load(D / "p0_gbdt/features_val.npz", D / "era5/era5_val.npz")
    prm = dict(n_estimators=700, learning_rate=0.03, num_leaves=63, min_child_samples=40,
               subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
               class_weight="balanced", n_jobs=8, verbose=-1)

    res = {}
    for name, (A, B, C) in {
        "base51": (Xtr, Xdv, Xbk),
        "base+era5": (np.hstack([Xtr, Etr]), np.hstack([Xdv, Edv]), np.hstack([Xbk, Ebk])),
    }.items():
        pd_, pb = [], []
        for s in (0, 1, 2):
            m = lgb.LGBMClassifier(**prm, random_state=s).fit(A, ytr)
            pd_.append(m.predict_proba(B)[:, 1]); pb.append(m.predict_proba(C)[:, 1])
        pd_, pb = np.mean(pd_, 0), np.mean(pb, 0)
        tau, _ = M.best_threshold(ydv, pd_, "max_csi")
        res[name] = {
            "dev": {**M.ranking(ydv, pd_), "CSI": M.skill(ydv, pd_, tau)["CSI"]},
            "bench": {**M.ranking(ybk, pb), **{k: M.skill(ybk, pb, tau)[k] for k in ("CSI", "POD", "FAR")}},
            "tau": tau,
        }
        print(name, json.dumps(res[name], indent=1))

    out = D / "era5/ablation.json"
    json.dump(res, open(out, "w"), indent=2)
    d = res["base+era5"]["dev"]["ROC_AUC"] - res["base51"]["dev"]["ROC_AUC"]
    print("dev AUC delta %+.4f | bench AUC delta %+.4f | CSI delta %+.4f" % (
        d, res["base+era5"]["bench"]["ROC_AUC"] - res["base51"]["bench"]["ROC_AUC"],
        res["base+era5"]["bench"]["CSI"] - res["base51"]["bench"]["CSI"]))


if __name__ == "__main__":
    main()
