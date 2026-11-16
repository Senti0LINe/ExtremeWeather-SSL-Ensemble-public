#!/usr/bin/env bash
# Wait for the May pressure-level file, then run the ERA5 ablation end-to-end.
set -u
PROJECT=/home/group3/group3_work/HYLiang/ExtremeWeather_SSL_Ensemble_Qoder
DATA=/home/group3/group3_work/HYLiang/data/era5
source /home/group3/miniconda3/etc/profile.d/conda.sh
conda activate ci_det
cd "$PROJECT"
export PYTHONPATH="$PROJECT"
echo "[chain] waiting for pl_2021-05.nc at $(date -Is)"
for i in $(seq 1 144); do
  [ -s "$DATA/pl_2021-05.nc" ] && break
  sleep 300
done
if [ ! -s "$DATA/pl_2021-05.nc" ]; then echo "[chain] timeout, file missing"; exit 1; fi
sleep 10
echo "[chain] file present, extracting features at $(date -Is)"
python tools/era5_features.py --split dev --out experiments/era5/era5_dev.npz
python tools/era5_features.py --split val --out experiments/era5/era5_val.npz
python tools/era5_features.py --split test_eval --out experiments/era5/era5_test_eval.npz
echo "[chain] ablation at $(date -Is)"
python tools/era5_ablation.py
echo "[chain] done at $(date -Is)"
