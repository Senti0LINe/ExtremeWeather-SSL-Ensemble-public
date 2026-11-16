#!/usr/bin/env bash
#SBATCH --job-name=ew_test
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --requeue
#SBATCH --output=/home/group3/group3_work/HYLiang/ExtremeWeather_SSL_Ensemble_Qoder/logs/test_%j.out
#SBATCH --error=/home/group3/group3_work/HYLiang/ExtremeWeather_SSL_Ensemble_Qoder/logs/test_%j.err
set -euo pipefail
PROJECT=/home/group3/group3_work/HYLiang/ExtremeWeather_SSL_Ensemble_Qoder
source /home/group3/miniconda3/etc/profile.d/conda.sh
conda activate ci_det
cd "$PROJECT"
export PYTHONPATH="$PROJECT"
python tools/predict_cnn.py --ckpt experiments/p1_cnn/best.pt --out experiments/final/cnn_test_eval.npz
echo "[DONE] $(date -Is)"
