"""Physically interpretable features for CI windows.

Every column is a plain meteorological quantity: brightness temperature, its
spatial statistics, its change over the four frames, or a band difference.
Nothing is learned, so every feature can be explained and mapped.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from .data import BTD_PAIRS, IR7, FILL_VALUE, SCALE

IDX = {b: i for i, b in enumerate(IR7)}

FEATURE_NAMES: List[str] = []
for _b in ("B13", "B08", "B11", "B16"):
    FEATURE_NAMES += [_b + "_p01", _b + "_p50", _b + "_mean", _b + "_std"]
FEATURE_NAMES += ["BT13_min", "BT13_p05", "BT13_p25", "BT13_p75", "BT13_p95"]
FEATURE_NAMES += ["cold_frac_lt" + str(t) for t in (200, 210, 220, 225, 235)]
FEATURE_NAMES += [
    "d_min_bt13",
    "d_p05_bt13",
    "cold_area_growth_220",
    "frame_absdiff_bt13",
    "cold_centroid_shift_px",
    "new_cold_pixel_frac_220",
]
for _a, _b in BTD_PAIRS:
    FEATURE_NAMES.append("btd_" + _a + "_" + _b + "_at_min")
for _a, _b in BTD_PAIRS:
    FEATURE_NAMES.append("btd_" + _a + "_" + _b + "_mean")
FEATURE_NAMES += ["grad_p95", "grad_mean", "contrast_med_min", "cold_bbox_fill"]
FEATURE_NAMES += ["center_BT13_min", "center_BT13_std", "center_cold_frac_225"]


def _pct(a: np.ndarray, q: float) -> float:
    v = a[np.isfinite(a)]
    return float(np.percentile(v, q)) if v.size else np.nan


def _std(a: np.ndarray) -> float:
    v = a[np.isfinite(a)]
    return float(v.std()) if v.size else np.nan


def extract(raw: np.ndarray) -> np.ndarray:
    """Feature vector for one patch of raw int16 [4, 7, H, W]."""
    bt = np.asarray(raw).astype(np.float32) * SCALE
    bt[np.asarray(raw) == FILL_VALUE] = np.nan

    last = bt[-1]
    first = bt[0]
    b13_last = last[IDX["B13"]]
    b13_first = first[IDX["B13"]]
    fin = np.isfinite(b13_last)
    n_valid = max(1, int(fin.sum()))

    f: List[float] = []
    for b in ("B13", "B08", "B11", "B16"):
        plane = last[IDX[b]]
        f += [_pct(plane, 1), _pct(plane, 50), float(np.nanmean(plane)), _std(plane)]

    for q in (0, 5, 25, 75, 95):
        f.append(_pct(b13_last, q))

    for thr in (200, 210, 220, 225, 235):
        f.append(float(((b13_last < thr) & fin).sum()) / n_valid)

    f.append(_pct(b13_last, 0) - _pct(b13_first, 0))
    f.append(_pct(b13_last, 5) - _pct(b13_first, 5))

    cold_first = np.isfinite(b13_first) & (b13_first < 220)
    cold_last = np.isfinite(b13_last) & (b13_last < 220)
    a0 = int(cold_first.sum())
    a3 = int(cold_last.sum())
    f.append(float(a3 - a0) / float(max(a0, 1)))

    both = np.isfinite(b13_first) & np.isfinite(b13_last)
    f.append(float(np.abs(b13_last[both] - b13_first[both]).mean()) if both.any() else np.nan)

    def centroid(mask: np.ndarray) -> Tuple[float, float]:
        ys, xs = np.nonzero(mask)
        return (float(ys.mean()), float(xs.mean())) if ys.size else (np.nan, np.nan)

    c0, c3 = centroid(cold_first), centroid(cold_last)
    if np.isfinite(c0[0]) and np.isfinite(c3[0]):
        f.append(float(np.hypot(c3[0] - c0[0], c3[1] - c0[1])))
    else:
        f.append(0.0)
    f.append(float(int((cold_last & ~cold_first).sum())) / float(max(a3, 1)))

    if fin.any():
        yy, xx = np.unravel_index(np.nanargmin(b13_last), b13_last.shape)
        spec = last[:, yy, xx]
    else:
        spec = np.full(len(IR7), np.nan, dtype=np.float32)
    for a, b in BTD_PAIRS:
        f.append(float(spec[IDX[a]] - spec[IDX[b]]))
    for a, b in BTD_PAIRS:
        m = float(np.nanmean(last[IDX[a]] - last[IDX[b]]))
        f.append(m if np.isfinite(m) else np.nan)

    fill = np.nanmedian(b13_last) if fin.any() else 0.0
    gy, gx = np.gradient(np.where(fin, b13_last, fill))
    grad = np.hypot(gy, gx)
    f.append(float(np.percentile(grad, 95)))
    f.append(float(grad.mean()))
    f.append(_pct(b13_last, 50) - _pct(b13_last, 0))

    if cold_last.any():
        ys, xs = np.nonzero(cold_last)
        bbox = (ys.max() - ys.min() + 1) * (xs.max() - xs.min() + 1)
        f.append(float(cold_last.sum()) / float(max(bbox, 1)))
    else:
        f.append(0.0)

    h, w = b13_last.shape
    ch, cw, q = h // 2, w // 2, 32
    centre = b13_last[ch - q : ch + q, cw - q : cw + q]
    cf = np.isfinite(centre)
    f.append(_pct(centre, 0))
    f.append(_std(centre))
    f.append(float(((centre < 225) & cf).sum()) / max(1, int(cf.sum())))

    return np.asarray(f, dtype=np.float32)
