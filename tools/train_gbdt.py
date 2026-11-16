"""Train and honestly evaluate the physics-feature GBDT baseline (P0).

Protocol
--------
* train  : 2021-03-01 .. 2021-04-30 (all annotated windows)
* val    : the official 30 May timestamps (9,180 windows)
* The 30 timestamps are split by *timestamp* into 6 for threshold tuning and
  24 for the primary test.  No window ever crosses the split.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from ewci import metrics as M  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _load(path: str):
    z = np.load(path, allow_pickle=True)
    return {k: z[k] for k in z.files}


def _params(n_jobs: int) -> dict:
    return dict(
        n_estimators=800,
        learning_rate=0.03,
        num_leaves=63,
        min_child_samples=40,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        class_weight="balanced",
        random_state=0,
        n_jobs=n_jobs,
        verbose=-1,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="experiments/p0_gbdt/features_train.npz")
    ap.add_argument("--val", default="experiments/p0_gbdt/features_val.npz")
    ap.add_argument("--out", default="experiments/p0_gbdt")
    ap.add_argument("--n-jobs", type=int, default=8)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tr, va = _load(args.train), _load(args.val)
    names = [str(n) for n in tr["names"]]

    uids = sorted(set(va["uid"].tolist()))
    rng = random.Random(7)
    rng.shuffle(uids)
    n_tune = max(1, int(round(0.2 * len(uids))))
    tune_uids = set(uids[:n_tune])
    is_tune = np.array([u in tune_uids for u in va["uid"]])
    print("val timestamps: %d total -> %d tune / %d test" % (len(uids), n_tune, len(uids) - n_tune))
    print("windows:        %d total -> %d tune / %d test" % (len(va["uid"]), is_tune.sum(), (~is_tune).sum()))

    import lightgbm as lgb

    # ---- variant A: use every training window (positives are over-represented)
    clf_all = lgb.LGBMClassifier(**_params(args.n_jobs))
    clf_all.fit(tr["X"], tr["y"])
    p_tune_all = clf_all.predict_proba(va["X"][is_tune])[:, 1]
    auc_all = M.ranking(va["y"][is_tune], p_tune_all)["ROC_AUC"]

    # ---- variant B: 50/50 subsample (matches the balanced evaluation prior)
    r = np.random.RandomState(0)
    pos = np.where(tr["y"] == 1)[0]
    neg = np.where(tr["y"] == 0)[0]
    n = min(len(pos), len(neg))
    idx = np.concatenate([r.choice(pos, n, replace=False), r.choice(neg, n, replace=False)])
    params_b = _params(args.n_jobs)
    params_b["class_weight"] = None
    clf_bal = lgb.LGBMClassifier(**params_b)
    clf_bal.fit(tr["X"][idx], tr["y"][idx])
    p_tune_bal = clf_bal.predict_proba(va["X"][is_tune])[:, 1]
    auc_bal = M.ranking(va["y"][is_tune], p_tune_bal)["ROC_AUC"]

    use_bal = auc_bal >= auc_all
    clf = clf_bal if use_bal else clf_all
    print("tune AUC  all-windows=%.4f  50/50=%.4f  -> using %s" % (auc_all, auc_bal, "50/50" if use_bal else "all"))

    # thresholds are chosen on the tune timestamps ONLY
    p_tune = clf.predict_proba(va["X"][is_tune])[:, 1]
    tau_csi, s_tune = M.best_threshold(va["y"][is_tune], p_tune, "max_csi")
    tau_far, _ = M.threshold_at_far(va["y"][is_tune], p_tune, 0.10)
    print("tuned thresholds: max-CSI tau=%.4f | FAR<=10%% tau=%.4f" % (tau_csi, tau_far))

    prob = clf.predict_proba(va["X"])[:, 1]
    y = va["y"]

    res = {}
    for label, mask in (("test_24", ~is_tune), ("official_30", np.ones(len(y), dtype=bool))):
        yy, pp = y[mask], prob[mask]
        res[label] = {
            "n": int(mask.sum()),
            "n_pos": int(yy.sum()),
            "n_timestamps": len(uids) if label == "official_30" else len(uids) - n_tune,
            **M.ranking(yy, pp),
            "at_tau_csi": M.skill(yy, pp, tau_csi),
            "at_far_10": M.skill(yy, pp, tau_far),
        }

    # reliability on the primary test split
    yy, pp = y[~is_tune], prob[~is_tune]
    xs, ys, ns, ece = M.reliability(yy, pp, 10)
    res["test_24"]["ECE"] = ece
    res["official_30"]["ECE"] = M.reliability(y, prob, 10)[3]

    # per-case breakdown on the primary split
    cases = va["case"][~is_tune]
    per_case = {}
    for c in sorted(set(cases.tolist())):
        m = cases == c
        if m.sum() >= 30 and len(np.unique(yy[m])) > 1:
            per_case[str(c)] = {
                "n": int(m.sum()),
                "AUC": M.ranking(yy[m], pp[m])["ROC_AUC"],
                "CSI": M.skill(yy[m], pp[m], tau_csi)["CSI"],
            }

    (out / "metrics.json").write_text(
        json.dumps({"thresholds": {"max_csi": tau_csi, "far10": tau_far},
                    "variant": "50/50" if use_bal else "all",
                    "results": res, "per_case": per_case}, indent=2)
    )

    # ---------------- figures ----------------
    plt.figure(figsize=(6, 5))
    from sklearn.metrics import roc_curve, precision_recall_curve
    fpr, tpr, _ = roc_curve(yy, pp)
    plt.plot(fpr, tpr, lw=2, label="ROC-AUC %.3f" % M.ranking(yy, pp)["ROC_AUC"])
    plt.plot([0, 1], [0, 1], "k:", lw=1)
    plt.xlabel("False alarm rate (background)")
    plt.ylabel("Hit rate (POD)")
    plt.title("Held-out timestamps (24)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "roc.png", dpi=160)
    plt.close()

    plt.figure(figsize=(6, 5))
    pr, rc, _ = precision_recall_curve(yy, pp)
    plt.plot(rc, pr, lw=2, label="PR-AUC %.3f" % M.ranking(yy, pp)["PR_AUC"])
    plt.axhline(yy.mean(), color="k", ls=":", lw=1, label="base rate %.3f" % yy.mean())
    plt.xlabel("Recall (POD)")
    plt.ylabel("Precision (1 - FAR)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "pr.png", dpi=160)
    plt.close()

    plt.figure(figsize=(6, 5))
    plt.plot([0, 1], [0, 1], "k:", lw=1)
    plt.plot(xs, ys, "o-", lw=2)
    plt.xlabel("Predicted probability")
    plt.ylabel("Observed frequency")
    plt.title("Reliability (ECE %.3f)" % ece)
    plt.tight_layout()
    plt.savefig(out / "reliability.png", dpi=160)
    plt.close()

    grid = np.unique(np.quantile(pp, np.linspace(0.5, 0.9995, 120)))
    csi = [M.skill(yy, pp, t)["CSI"] for t in grid]
    pod = [M.skill(yy, pp, t)["POD"] for t in grid]
    far = [M.skill(yy, pp, t)["FAR"] for t in grid]
    plt.figure(figsize=(7, 5))
    plt.plot(grid, pod, label="POD")
    plt.plot(grid, far, label="FAR", ls="--")
    plt.plot(grid, csi, label="CSI", lw=2.5)
    plt.axvline(tau_csi, color="k", ls=":", label="tuned tau %.2f" % tau_csi)
    plt.xlabel("Decision threshold")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "threshold_curve.png", dpi=160)
    plt.close()

    imp = clf.feature_importances_
    order = np.argsort(imp)[::-1][:20]
    plt.figure(figsize=(7, 6))
    plt.barh([names[i] for i in order][::-1], imp[order][::-1])
    plt.xlabel("split gain")
    plt.title("Top physical features")
    plt.tight_layout()
    plt.savefig(out / "importance.png", dpi=160)
    plt.close()

    # ---------------- report ----------------
    def row(label, d, key="at_tau_csi"):
        s = d[key]
        return ("| %s | %d | %.3f | %.3f | %.1f%% | %.1f%% | %.1f%% | %.3f |" %
                (label, d["n"], d["ROC_AUC"], d["PR_AUC"], s["CSI"] * 100,
                 s["POD"] * 100, s["FAR"] * 100, s["Brier"]))

    lines = []
    lines.append("# P0 physics-feature GBDT baseline\n")
    lines.append("Train: 2021-03-01..04-30 (all windows). Val: official 30 May timestamps.")
    lines.append("Threshold tuned on 6 held-out timestamps, reported on the other 24.\n")
    lines.append("| Split | Windows | ROC-AUC | PR-AUC | CSI | POD | FAR | Brier |")
    lines.append("|---|---|---|---|---|---|---|---|")
    lines.append(row("held-out 24 ts (primary)", res["test_24"]))
    lines.append(row("official 30 ts", res["official_30"]))
    lines.append("")
    lines.append("At FAR<=10%% (tau=%.3f): POD %.1f%% | CSI %.1f%%" % (
        tau_far, res["test_24"]["at_far_10"]["POD"] * 100, res["test_24"]["at_far_10"]["CSI"] * 100))
    lines.append("")
    lines.append("Reference - legacy v0 ensemble: ROC-AUC 0.6745 | PR-AUC 0.2474 | CSI 19.87% (tau 0.64)\n")
    lines.append("## Per case (held-out)")
    for c, d in per_case.items():
        lines.append("- %s: n=%d AUC=%.3f CSI=%.3f" % (c, d["n"], d["AUC"], d["CSI"]))
    (out / "report.md").write_text("\n".join(lines))

    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
