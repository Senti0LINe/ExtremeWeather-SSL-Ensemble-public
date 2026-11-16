#!/usr/bin/env bash
# Keep trying the May pressure-level download until the file exists.
PROJECT=/home/group3/group3_work/HYLiang/ExtremeWeather_SSL_Ensemble_Qoder
DATA=/home/group3/group3_work/HYLiang/data/era5
source /home/group3/miniconda3/etc/profile.d/conda.sh
conda activate ci_det
cd "$PROJECT"
export PYTHONPATH="$PROJECT"
for i in $(seq 1 200); do
  [ -s "$DATA/pl_2021-05.nc" ] && { echo "[sup] done $(date -Is)"; exit 0; }
  source /home/group3/group3_work/HYLiang/env_proxy.sh >/dev/null 2>&1
  /home/group3/miniconda3/envs/ci_det/bin/python3 tools/download_era5_min.py --kind pl >> /home/group3/group3_work/HYLiang/logs_era5_pl.log 2>&1
  sleep 300
done
