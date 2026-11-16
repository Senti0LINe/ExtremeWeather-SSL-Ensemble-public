"""Verification metrics for the CI benchmark (plain, dependency-light)."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def contingency(y: np.ndarray, pred: np.ndarray) -> Dict[str, int]:
    y = np.asarray(y).astype(int).ravel()
    pred = np.asarray(pred).astype(int).ravel()
    h = int(((y == 1) & (pred == 1)).sum())
    f = int(((y == 0) & (pred == 1)).sum())
    m = int(((y == 1) & (pred == 0)).sum())
    cn = int(((y == 0) & (pred == 0)).sum())
    return {"hits": h, "false_alarms": f, "misses": m, "correct_negatives": cn}


def skill(y: np.ndarray, prob: np.ndarray, tau: float, cost_fn: float = 5.0, cost_fp: float = 1.0) -> Dict[str, float]:
    c = contingency(y, (np.asarray(prob) >= tau).astype(int))
    h, f, m, cn = c["hits"], c["false_alarms"], c["misses"], c["correct_negatives"]
    n = h + f + m + cn
    eps = 1e-9
    h_rand = (h + m) * (h + f) / max(1, n)
    denom_hss = (h + m) * (m + cn) + (h + f) * (f + cn)
    out = {
        "POD": h / (h + m + eps),
        "FAR": f / (h + f + eps),
        "POFD": f / (f + cn + eps),
        "CSI": h / (h + f + m + eps),
        "ETS": (h - h_rand) / (h + f + m - h_rand + eps),
        "HSS": 2.0 * (h * cn - f * m) / (denom_hss + eps),
        "BIAS": (h + f) / (h + m + eps),
        "Brier": float(np.mean((np.asarray(prob) - y) ** 2)),
        "TotalCost": cost_fn * m + cost_fp * f,
        "threshold": float(tau),
        **c,
    }
    return out


def ranking(y: np.ndarray, prob: np.ndarray) -> Dict[str, float]:
    y = np.asarray(y).astype(int).ravel()
    if len(np.unique(y)) < 2:
        return {"ROC_AUC": float("nan"), "PR_AUC": float("nan")}
    return {
        "ROC_AUC": float(roc_auc_score(y, prob)),
        "PR_AUC": float(average_precision_score(y, prob)),
    }


def best_threshold(
    y: np.ndarray, prob: np.ndarray, criterion: str = "max_csi", cost_fn: float = 5.0, cost_fp: float = 1.0
) -> Tuple[float, Dict[str, float]]:
    prob = np.asarray(prob).ravel()
    # Search the full probability range: restricting to the upper half (the old
    # 0.5..0.9995 quantile grid) can miss the max-CSI threshold entirely at low
    # prevalence (E14 P8).
    grid = np.unique(np.quantile(prob, np.linspace(0.01, 0.9995, 800)))
    best_tau, best = 0.5, None
    best_val = -np.inf if criterion == "max_csi" else np.inf
    for tau in grid:
        s = skill(y, prob, float(tau), cost_fn, cost_fp)
        v = s["CSI"] if criterion == "max_csi" else s["TotalCost"]
        if (criterion == "max_csi" and v > best_val) or (criterion == "min_cost" and v < best_val):
            best_val, best_tau, best = v, float(tau), s
    return best_tau, best


def threshold_at_far(y: np.ndarray, prob: np.ndarray, far_target: float = 0.10) -> Tuple[float, Dict[str, float]]:
    """Highest POD threshold whose FAR stays at or below far_target."""
    prob = np.asarray(prob).ravel()
    grid = np.unique(np.quantile(prob, np.linspace(0.01, 0.9999, 1200)))
    best_tau, best = None, None
    for tau in grid:
        s = skill(y, prob, float(tau))
        if s["FAR"] <= far_target and (best is None or s["POD"] > best["POD"]):
            best_tau, best = float(tau), s
    if best is None:
        best_tau, best = 1.0, skill(y, prob, 1.0)
    return best_tau, best


def reliability(y: np.ndarray, prob: np.ndarray, n_bins: int = 10) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    y = np.asarray(y).astype(float).ravel()
    prob = np.asarray(prob).astype(float).ravel()
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(prob, edges) - 1, 0, n_bins - 1)
    xs, ys, ns = [], [], []
    for b in range(n_bins):
        m = idx == b
        if m.sum():
            xs.append(float(prob[m].mean()))
            ys.append(float(y[m].mean()))
            ns.append(int(m.sum()))
    xs, ys, ns = np.array(xs), np.array(ys), np.array(ns)
    ece = float(np.sum(ns * np.abs(xs - ys)) / max(1, ns.sum()))
    return xs, ys, ns, ece
