"""Extract ERA5 environment features at each window centre.

Features (7): CAPE, 2 m temperature, MSL pressure, 200-850 hPa wind shear,
850 hPa specific humidity, 850 hPa temperature, 200 hPa temperature.

Window centre: lat = 40.00448 - 0.01*(row+112), lon = 103.99570 + 0.01*(col+112)
(derived from the annotation contours, residual < 0.006 deg).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))
from ewci import data  # noqa: E402

FULL = Path("/data1/group3/DATASAT/experiments/ci_det_v2/annotations_full")
ERA5 = Path("/home/group3/group3_work/HYLiang/data/era5")
NAMES = ["cape", "t2m", "msl", "shear_200_850", "q850", "t850", "t200"]
LAT0, LON0 = 40.00448, 103.99570


def windows(split: str):
    if split == "train":
        return data.load_patches("train")
    if split == "val":
        return data.benchmark_patches()
    if split == "dev":
        uids = set(json.loads((PROJECT / "splits_v1.json").read_text())["dev"]["early_stop_uids"])
        return [p for p in data.load_patches_from(FULL, "val") if p["sample_uid"] in uids]
    if split == "test_eval":
        uids = set(json.loads((PROJECT / "splits_test_eval.json").read_text())["uids"])
        return [p for p in data.load_patches_from(FULL, "test") if p["sample_uid"] in uids]
    raise SystemExit("unknown split " + split)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    patches = windows(args.split)
    n = len(patches)
    X = np.full((n, len(NAMES)), np.nan, dtype=np.float32)
    t = pd.to_datetime([p["label_utc"] for p in patches])
    lat = LAT0 - 0.01 * (np.array([p["row"] for p in patches]) + 112)
    lon = LON0 + 0.01 * (np.array([p["col"] for p in patches]) + 112)
    months = pd.PeriodIndex(t, freq="M")
    for month in sorted(set(months)):
        idx = np.where(months == month)[0]
        tag = "%04d-%02d" % (month.year, month.month)
        pl_path = ERA5 / ("pl_%s.nc" % tag)
        sl_path = ERA5 / ("sl_%s.nc" % tag)
        if not pl_path.exists() or not sl_path.exists():
            print("missing files for", tag, "-> skipped")
            continue
        pl = xr.open_dataset(pl_path).drop_vars(["expver", "number"], errors="ignore")
        sl = xr.open_dataset(sl_path).drop_vars(["expver", "number"], errors="ignore")
        ti = xr.DataArray(t[idx].values, dims="p")
        la = xr.DataArray(lat[idx], dims="p")
        lo = xr.DataArray(lon[idx], dims="p")
        sel = dict(valid_time=ti, latitude=la, longitude=lo, method="nearest")
        spl = pl.sel(**sel)
        ssl = sl.sel(**sel)
        u850 = spl["u"].sel(pressure_level=850).values
        v850 = spl["v"].sel(pressure_level=850).values
        u200 = spl["u"].sel(pressure_level=200).values
        v200 = spl["v"].sel(pressure_level=200).values
        X[idx, 0] = ssl["cape"].values
        X[idx, 1] = ssl["t2m"].values - 273.15
        X[idx, 2] = ssl["msl"].values / 100.0
        X[idx, 3] = np.hypot(u200 - u850, v200 - v850)
        X[idx, 4] = spl["q"].sel(pressure_level=850).values * 1000.0
        X[idx, 5] = spl["t"].sel(pressure_level=850).values - 273.15
        X[idx, 6] = spl["t"].sel(pressure_level=200).values - 273.15
        print(tag, "filled", len(idx))
        pl.close(); sl.close()

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, X=X, names=np.array(NAMES),
                        uid=np.array([p["sample_uid"] for p in patches]),
                        row=np.array([int(p["row"]) for p in patches]),
                        col=np.array([int(p["col"]) for p in patches]))
    print("saved", out, X.shape, "nan", int(np.isnan(X).sum()),
          "ranges", {k: [round(float(np.nanmin(X[:, i])), 1), round(float(np.nanmax(X[:, i])), 1)]
                     for i, k in enumerate(NAMES)})


if __name__ == "__main__":
    main()
