"""Combine 5 TTA seeds, compare with the single-seed ensemble on dev/benchmark."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import lightgbm as lgb

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))
from ewci import data, metrics as M  # noqa: E402

D = PROJECT / "experiments"
FULL = Path("/data1/group3/DATASAT/experiments/ci_det_v2/annotations_full")


def load_npz(p):
    z = np.load(p, allow_pickle=True)
    return z["p"], [(str(u), int(r), int(c)) for u, r, c in zip(z["uid"], z["row"], z["col"])]


def dev_keys():
    uids = set(json.loads((PROJECT / "splits_v1.json").read_text())["dev"]["early_stop_uids"])
    ps = [p for p in data.load_patches_from(FULL, "val") if p["sample_uid"] in uids]
    return [(p["sample_uid"], int(p["row"]), int(p["col"])) for p in ps], np.array([data.has_ci(p) for p in ps])


def align(probs, keys, order):
    pos = {k: i for i, k in enumerate(keys)}
    return probs[np.array([pos[k] for k in order])]


def main():
    dk, yd = dev_keys()
    seeds = range(5)
    dev_p, bench_p = [], []
    for s in seeds:
        p, k = load_npz(D / ("p1_seed%d_dev_tta.npz" % s)); dev_p.append(align(p, k, dk))
        p, k = load_npz(D / ("p1_seed%d_bench_tta.npz" % s)); bench_p.append((p, k))
    cnn5_dev = np.mean(dev_p, 0)
    bkeys = bench_p[0][1]
    cnn5_bench = np.mean([align(p, k, bkeys) for p, k in bench_p], 0)
    cnn1_bench = align(bench_p[0][0], bench_p[0][1], bkeys)

    f_dev = np.load(D / "p0_gbdt/features_dev.npz", allow_pickle=True)
    f_val = np.load(D / "p0_gbdt/features_val.npz", allow_pickle=True)
    tr = np.load(D / "p0_gbdt/features_train.npz", allow_pickle=True)
    posd = {(str(u), int(r), int(c)): i for i, (u, r, c) in enumerate(zip(f_dev["uid"], f_dev["row"], f_dev["col"]))}
    idx = np.array([posd[k] for k in dk])
    prm = dict(n_estimators=700, learning_rate=0.03, num_leaves=63, min_child_samples=40,
               subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
               class_weight="balanced", n_jobs=8, verbose=-1)
    gd, gb = [], []
    for s in (0, 1, 2):
        m = lgb.LGBMClassifier(**prm, random_state=s).fit(tr["X"], tr["y"])
        gd.append(m.predict_proba(f_dev["X"][idx])[:, 1]); gb.append(m.predict_proba(f_val["X"])[:, 1])
    gd, gb = np.mean(gd, 0), np.mean(gb, 0)
    yb = f_val["y"]

    out = {}
    for name, pc_d, pc_b in (("cnn1_tta", dev_p[0], cnn1_bench), ("cnn5_tta", cnn5_dev, cnn5_bench)):
        best = (0.75, -1)
        for w in np.linspace(0, 1, 21):
            a = M.ranking(yd, w * pc_d + (1 - w) * gd)["ROC_AUC"]
            if a > best[1]:
                best = (float(w), a)
        w = best[0]
        p_d = w * pc_d + (1 - w) * gd
        p_b = w * pc_b + (1 - w) * gb
        from sklearn.isotonic import IsotonicRegression
        iso = IsotonicRegression(out_of_bounds="clip").fit(p_d, yd)
        pb = iso.predict(p_b)
        tau, _ = M.best_threshold(yd, iso.predict(p_d), "max_csi")
        s = M.skill(yb, pb, tau)
        out[name] = {"w_cnn": w, "dev_auc": M.ranking(yd, p_d)["ROC_AUC"],
                     "bench_auc": M.ranking(yb, pb)["ROC_AUC"], "bench_pr": M.ranking(yb, pb)["PR_AUC"],
                     "csi": s["CSI"], "pod": s["POD"], "far": s["FAR"], "tau": tau,
                     "ece": M.reliability(yb, pb)[3], "brier": s["Brier"]}
        print(name, json.dumps({k: round(v, 4) for k, v in out[name].items()}))
    json.dump(out, open(D / "final/seeds_tta.json", "w"), indent=2)


if __name__ == "__main__":
    main()
