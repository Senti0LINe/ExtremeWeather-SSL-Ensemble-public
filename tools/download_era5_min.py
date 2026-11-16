
"""Minimal ERA5 download for the CI benchmark (slow CDS queue, small requests).

Only the fields with a physical rationale for CI are fetched:
  * single levels : CAPE, 2 m temperature, MSL pressure
  * pressure levels: t, u, v, q at 200 and 850 hPa
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CDSAPI_RC", "/home/group3/group3_work/HYLiang/.cdsapirc")
import cdsapi

PL_VARS = ["temperature", "u_component_of_wind", "v_component_of_wind", "specific_humidity"]
PL_LEVELS = ["200", "850"]
SL_VARS = ["convective_available_potential_energy", "2m_temperature", "mean_sea_level_pressure"]
AREA = [41, 103, 19, 126]


def months(start, end):
    y0, m0 = map(int, start.split("-")[:2])
    y1, m1 = map(int, end.split("-")[:2])
    out, y, m = [], y0, m0
    while (y, m) <= (y1, m1):
        out.append((y, m))
        m += 1
        if m == 13:
            y, m = y + 1, 1
    return out


def fetch(client, kind, year, month, out_path, retries=5):
    if out_path.exists() and out_path.stat().st_size > 50_000:
        print("[skip] " + out_path.name, flush=True)
        return True
    common = {
        "product_type": ["reanalysis"],
        "year": [str(year)],
        "month": ["%02d" % month],
        "day": ["%02d" % d for d in range(1, 32)],
        "time": ["%02d:00" % h for h in range(24)],
        "data_format": "netcdf",
        "download_format": "unarchived",
        "area": AREA,
    }
    if kind == "pl":
        req = dict(common, variable=PL_VARS, pressure_level=PL_LEVELS)
        dataset = "reanalysis-era5-pressure-levels"
    else:
        req = dict(common, variable=SL_VARS)
        dataset = "reanalysis-era5-single-levels"
    for attempt in range(1, retries + 1):
        try:
            t0 = time.time()
            tmp = out_path.with_suffix(".part")
            client.retrieve(dataset, req, str(tmp))
            tmp.replace(out_path)
            print("[ok] %s %.1f MB %.0fs" % (out_path.name, out_path.stat().st_size / 1e6, time.time() - t0), flush=True)
            return True
        except Exception as exc:  # noqa: BLE001
            print("[retry %d] %s: %s" % (attempt, out_path.name, str(exc)[:200]), flush=True)
            time.sleep(30 * attempt)
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["pl", "sl"], required=True)
    ap.add_argument("--out", default="/home/group3/group3_work/HYLiang/data/era5")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    client = cdsapi.Client()
    for y, m in months("2021-03", "2021-05"):
        ok = fetch(client, args.kind, y, m, out / ("%s_%04d-%02d.nc" % (args.kind, y, m)))
        if not ok:
            print("[STOP] " + args.kind, flush=True)
            return 1
    print("[done] " + args.kind, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
