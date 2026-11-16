#!/usr/bin/env bash
#SBATCH --job-name=ew_final
#SBATCH --partition=debug
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:40:00
#SBATCH --output=/home/group3/group3_work/HYLiang/ExtremeWeather_SSL_Ensemble_Qoder/logs/final_%j.out
#SBATCH --error=/home/group3/group3_work/HYLiang/ExtremeWeather_SSL_Ensemble_Qoder/logs/final_%j.err
set -euo pipefail
PROJECT=/home/group3/group3_work/HYLiang/ExtremeWeather_SSL_Ensemble_Qoder
source /home/group3/miniconda3/etc/profile.d/conda.sh
conda activate ci_det
cd "$PROJECT"
export PYTHONPATH="$PROJECT"
python tools/final_test.py
echo "[DONE] $(date -Is)"
