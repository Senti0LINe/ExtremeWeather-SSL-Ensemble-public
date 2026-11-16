
"""Frozen-protocol evaluation of the physics-feature baseline.

train  : 2021-03-01..04-30 windows
dev    : 100 held-out May timestamps (threshold, model choice)
bench  : the official 30-timestamp benchmark (never used for tuning)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))
from ewci import metrics as M  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import lightgbm as lgb  # noqa: E402


def load(p):
    z = np.load(p, allow_pickle=True)
    return {k: z[k] for k in z.files}


def params(cls_weight="balanced", seed=0, n_jobs=8):
    return dict(n_estimators=700, learning_rate=0.03, num_leaves=63,
                min_child_samples=40, subsample=0.8, subsample_freq=1,
                colsample_bytree=0.8, reg_lambda=1.0, class_weight=cls_weight,
                random_state=seed, n_jobs=n_jobs, verbose=-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="experiments/p0_gbdt")
    ap.add_argument("--out", default="experiments/p0_frozen")
    args = ap.parse_args()
    d = Path(args.dir)
    tr, dev, bench = load(d / "features_train.npz"), load(d / "features_dev.npz"), load(d / "features_val.npz")
    names = [str(n) for n in tr["names"]]
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    # ---- model variants, chosen on dev AUC only ----
    variants = {}
    m = lgb.LGBMClassifier(**params("balanced")).fit(tr["X"], tr["y"])
    variants["balanced"] = m
    m2 = lgb.LGBMClassifier(**params(None)).fit(tr["X"], tr["y"])
    variants["plain"] = m2
    # 50/50 subsample
    r = np.random.RandomState(0)
    pos = np.where(tr["y"] == 1)[0]; neg = np.where(tr["y"] == 0)[0]
    n = min(len(pos), len(neg))
    idx = np.concatenate([r.choice(pos, n, replace=False), r.choice(neg, n, replace=False)])
    m3 = lgb.LGBMClassifier(**params(None)).fit(tr["X"][idx], tr["y"][idx])
    variants["sub50"] = m3

    dev_auc = {k: M.ranking(dev["y"], v.predict_proba(dev["X"])[:, 1])["ROC_AUC"] for k, v in variants.items()}
    print("dev AUC by variant:", {k: round(v, 4) for k, v in dev_auc.items()})
    best = max(dev_auc, key=dev_auc.get)
    print("selected:", best)

    # ---- 3-seed average of the selected variant ----
    probs_dev, probs_bench = [], []
    for s in (0, 1, 2):
        kw = params("balanced" if best == "balanced" else None, seed=s)
        if best == "sub50":
            mm = lgb.LGBMClassifier(**kw).fit(tr["X"][idx], tr["y"][idx])
        else:
            mm = lgb.LGBMClassifier(**kw).fit(tr["X"], tr["y"])
        probs_dev.append(mm.predict_proba(dev["X"])[:, 1])
        probs_bench.append(mm.predict_proba(bench["X"])[:, 1])
    p_dev = np.mean(probs_dev, axis=0)
    p_bench = np.mean(probs_bench, axis=0)

    # ---- thresholds from dev only ----
    tau_csi, dev_csi = M.best_threshold(dev["y"], p_dev, "max_csi")
    tau_f10, dev_f10 = M.threshold_at_far(dev["y"], p_dev, 0.10)
    tau_f20, dev_f20 = M.threshold_at_far(dev["y"], p_dev, 0.20)
    print("dev: tau_csi=%.3f FAR10=%.3f FAR20=%.3f" % (tau_csi, tau_f10, tau_f20))

    # ---- benchmark evaluation ----
    yb, pb = bench["y"], p_bench
    res = {"dev": {"n": int(len(dev["y"])), "pos": int(dev["y"].sum()), **M.ranking(dev["y"], p_dev)},
           "bench": {"n": int(len(yb)), "pos": int(yb.sum()), **M.ranking(yb, pb)}}
    for tag, tau in (("tau_csi", tau_csi), ("far10", tau_f10), ("far20", tau_f20)):
        res["bench"][tag] = M.skill(yb, pb, tau)
        res["dev"][tag] = M.skill(dev["y"], p_dev, tau)
    xs, ys, ns, ece = M.reliability(yb, pb, 10)
    res["bench"]["ECE"] = ece
    res["dev"]["ECE"] = M.reliability(dev["y"], p_dev, 10)[3]

    # bootstrap CI on the benchmark
    rng = np.random.RandomState(0)
    aucs, csis = [], []
    for _ in range(400):
        i = rng.randint(0, len(yb), len(yb))
        if len(np.unique(yb[i])) < 2:
            continue
        aucs.append(M.ranking(yb[i], pb[i])["ROC_AUC"])
        csis.append(M.skill(yb[i], pb[i], tau_csi)["CSI"])
    res["bench"]["AUC_CI95"] = [float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))]
    res["bench"]["CSI_CI95"] = [float(np.percentile(csis, 2.5)), float(np.percentile(csis, 97.5))]

    # per-case
    per_case = {}
    for c in sorted(set(bench["case"].tolist())):
        msk = bench["case"] == c
        if msk.sum() > 30 and len(np.unique(yb[msk])) > 1:
            per_case[str(c)] = {"n": int(msk.sum()),
                                "AUC": M.ranking(yb[msk], pb[msk])["ROC_AUC"],
                                "CSI": M.skill(yb[msk], pb[msk], tau_csi)["CSI"]}

    json.dump({"selected": best, "thresholds": {"csi": tau_csi, "far10": tau_f10, "far20": tau_f20},
               "results": res, "per_case": per_case}, open(out / "metrics.json", "w"), indent=2)

    # ---- figures ----
    from sklearn.metrics import roc_curve, precision_recall_curve
    fpr, tpr, _ = roc_curve(yb, pb)
    plt.figure(figsize=(6, 5)); plt.plot(fpr, tpr, lw=2, label="AUC %.3f" % res["bench"]["ROC_AUC"])
    plt.plot([0, 1], [0, 1], "k:"); plt.xlabel("False alarm rate"); plt.ylabel("POD")
    plt.title("Benchmark (30 timestamps, never tuned)"); plt.legend(); plt.tight_layout()
    plt.savefig(out / "roc_bench.png", dpi=160); plt.close()

    pr, rc, _ = precision_recall_curve(yb, pb)
    plt.figure(figsize=(6, 5)); plt.plot(rc, pr, lw=2, label="PR-AUC %.3f" % res["bench"]["PR_AUC"])
    plt.axhline(yb.mean(), color="k", ls=":", label="base rate %.3f" % yb.mean())
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.legend(); plt.tight_layout()
    plt.savefig(out / "pr_bench.png", dpi=160); plt.close()

    plt.figure(figsize=(6, 5)); plt.plot([0, 1], [0, 1], "k:"); plt.plot(xs, ys, "o-")
    plt.xlabel("Predicted probability"); plt.ylabel("Observed frequency")
    plt.title("Reliability, benchmark (ECE %.3f)" % ece); plt.tight_layout()
    plt.savefig(out / "reliability_bench.png", dpi=160); plt.close()

    lines = ["# Frozen-protocol baseline (physics features + LightGBM)", "",
             "train 2021-03-01..04-30 | dev %d windows / %d timestamps | benchmark %d windows" % (
                 res["dev"]["n"], 100, res["bench"]["n"]), "",
             "| split | n | pos | AUC | PR-AUC | CSI@tau | POD | FAR | Brier | ECE |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for tag, split in (("dev", "dev"), ("benchmark", "bench")):
        r = res[split]
        lines.append("| %s | %d | %d | %.4f | %.4f | %.1f%% | %.1f%% | %.1f%% | %.4f | %.3f |" % (
            tag, r["n"], r["pos"], r["ROC_AUC"], r["PR_AUC"], r["tau_csi"]["CSI"] * 100,
            r["tau_csi"]["POD"] * 100, r["tau_csi"]["FAR"] * 100, r["tau_csi"]["Brier"], r["ECE"]))
    lines += ["", "Benchmark AUC 95%% CI: [%.3f, %.3f] | CSI 95%% CI: [%.1f%%, %.1f%%]" % (
        res["bench"]["AUC_CI95"][0], res["bench"]["AUC_CI95"][1],
        res["bench"]["CSI_CI95"][0] * 100, res["bench"]["CSI_CI95"][1] * 100),
        "", "At FAR<=20%% on the benchmark: POD %.1f%%, CSI %.1f%%" % (
            res["bench"]["far20"]["POD"] * 100, res["bench"]["far20"]["CSI"] * 100),
        "", "Reference v0: AUC 0.6745 | PR-AUC 0.2474 | CSI 19.87%% @ tau 0.64", "",
        "## Per case (benchmark)"]
    for c, v in per_case.items():
        lines.append("- %s: n=%d AUC=%.3f CSI=%.3f" % (c, v["n"], v["AUC"], v["CSI"]))
    (out / "report.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()