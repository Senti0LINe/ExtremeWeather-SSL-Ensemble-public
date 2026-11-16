"""Fuse the CNN and the physics-feature GBDT, calibrate on dev, score the benchmark."""

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cnn-dir", default="experiments/p1_cnn")
    ap.add_argument("--feat-dir", default="experiments/p0_gbdt")
    ap.add_argument("--out", default="experiments/final")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    sp = json.loads((PROJECT / "splits_v1.json").read_text())
    uids = set(sp["dev"]["early_stop_uids"])
    FULL = Path("/data1/group3/DATASAT/experiments/ci_det_v2/annotations_full")
    dev_patches = [p for p in data.load_patches_from(FULL, "val") if p["sample_uid"] in uids]
    keys = [(p["sample_uid"], int(p["row"]), int(p["col"])) for p in dev_patches]
    y_dev = np.array([data.has_ci(p) for p in dev_patches])

    cnn = np.load(Path(args.cnn_dir) / "probs.npz")
    f_dev = np.load(Path(args.feat_dir) / "features_dev.npz", allow_pickle=True)
    fk = {(str(u), int(r), int(c)): i for i, (u, r, c) in enumerate(zip(f_dev["uid"], f_dev["row"], f_dev["col"]))}
    idx = np.array([fk[k] for k in keys])

    f_val = np.load(Path(args.feat_dir) / "features_val.npz", allow_pickle=True)
    tr = np.load(Path(args.feat_dir) / "features_train.npz", allow_pickle=True)
    prm = dict(n_estimators=700, learning_rate=0.03, num_leaves=63, min_child_samples=40,
               subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
               class_weight="balanced", n_jobs=8, verbose=-1)
    p_gbdt_dev, p_gbdt_bench = [], []
    for s in (0, 1, 2):
        m = lgb.LGBMClassifier(**prm, random_state=s).fit(tr["X"], tr["y"])
        p_gbdt_dev.append(m.predict_proba(f_dev["X"][idx])[:, 1])
        p_gbdt_bench.append(m.predict_proba(f_val["X"])[:, 1])
    p_gbdt_dev = np.mean(p_gbdt_dev, 0)
    p_gbdt_bench = np.mean(p_gbdt_bench, 0)
    y_bench = f_val["y"]
    p_cnn_dev, p_cnn_bench = cnn["p_dev"], cnn["p_bench"]

    best = (0.5, -1.0)
    for w in np.linspace(0, 1, 21):
        a = M.ranking(y_dev, w * p_cnn_dev + (1 - w) * p_gbdt_dev)["ROC_AUC"]
        if a > best[1]:
            best = (float(w), a)
    w = best[0]
    p_dev = w * p_cnn_dev + (1 - w) * p_gbdt_dev
    p_bench = w * p_cnn_bench + (1 - w) * p_gbdt_bench

    from sklearn.isotonic import IsotonicRegression
    iso = IsotonicRegression(out_of_bounds="clip").fit(p_dev, y_dev)
    p_bench_cal = iso.predict(p_bench)
    tau, _ = M.best_threshold(y_dev, iso.predict(p_dev), "max_csi")
    s = M.skill(y_bench, p_bench_cal, tau)

    order = np.argsort(p_bench_cal)[::-1]
    P = y_bench[order]; tp = np.cumsum(P); fp = np.cumsum(1 - P)
    pos = float(y_bench.sum()); neg = float(len(y_bench) - y_bench.sum())
    pofd = fp / neg
    at = {}
    for tgt in (0.10, 0.20):
        k = min(int(np.searchsorted(pofd, tgt)), len(tp) - 1)
        at["pofd_%d" % int(tgt * 100)] = {"POD": float(tp[k] / pos),
                                         "CSI": float(tp[k] / (tp[k] + fp[k] + (pos - tp[k])))}

    per_case = {}
    key_pos = {(str(u), int(r), int(c)): i
               for i, (u, r, c) in enumerate(zip(f_val["uid"], f_val["row"], f_val["col"]))}
    bench_keys = [(p["sample_uid"], int(p["row"]), int(p["col"])) for p in data.benchmark_patches()]
    order_map = np.array([key_pos[k] for k in bench_keys])
    yb = y_bench[order_map]; pb = p_bench_cal[order_map]
    cases = f_val["case"][order_map]
    for c in sorted(set(cases.tolist())):
        msk = cases == c
        if msk.sum() > 30 and len(np.unique(yb[msk])) > 1:
            per_case[str(c)] = {"n": int(msk.sum()), "pos": int(yb[msk].sum()),
                                "AUC": M.ranking(yb[msk], pb[msk])["ROC_AUC"],
                                "CSI": M.skill(yb[msk], pb[msk], tau)["CSI"]}

    res = {"weight_cnn": w,
           "dev": {"cnn": M.ranking(y_dev, p_cnn_dev), "gbdt": M.ranking(y_dev, p_gbdt_dev),
                   "ensemble": M.ranking(y_dev, p_dev)},
           "bench": {"cnn": M.ranking(y_bench, p_cnn_bench), "gbdt": M.ranking(y_bench, p_gbdt_bench),
                     "ensemble": M.ranking(y_bench, p_bench),
                     "ECE_raw": M.reliability(y_bench, p_bench)[3],
                     "ECE_cal": M.reliability(y_bench, p_bench_cal)[3],
                     "Brier_cal": M.skill(y_bench, p_bench_cal, 0.5)["Brier"],
                     "tau": tau, "skill": s, "at": at},
           "per_case": per_case}
    json.dump(res, open(out / "metrics.json", "w"), indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
