#!/usr/bin/env bash
# Stage D: CPU-only fusion + calibration + per-case report.
#SBATCH --job-name=ew_report
#SBATCH --partition=debug
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/home/group3/group3_work/HYLiang/ExtremeWeather_SSL_Ensemble_Qoder/logs/report_%j.out
#SBATCH --error=/home/group3/group3_work/HYLiang/ExtremeWeather_SSL_Ensemble_Qoder/logs/report_%j.err
set -euo pipefail
PROJECT=/home/group3/group3_work/HYLiang/ExtremeWeather_SSL_Ensemble_Qoder
source /home/group3/miniconda3/etc/profile.d/conda.sh
conda activate ci_det
cd "$PROJECT"
export PYTHONPATH="$PROJECT"
python tools/ensemble_report.py --cnn-dir experiments/p2_ssl_ft --feat-dir experiments/p0_gbdt --out experiments/p2_final
python tools/ensemble_report.py --cnn-dir experiments/p1_cnn --feat-dir experiments/p0_gbdt --out experiments/p1_final
echo "[DONE] $(date -Is)"
