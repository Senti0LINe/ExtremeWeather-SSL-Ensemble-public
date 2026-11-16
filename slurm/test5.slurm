#!/usr/bin/env bash
#SBATCH --job-name=ew_test5
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=24G
#SBATCH --time=02:30:00
#SBATCH --requeue
#SBATCH --output=/home/group3/group3_work/HYLiang/ExtremeWeather_SSL_Ensemble_Qoder/logs/test5_%j.out
#SBATCH --error=/home/group3/group3_work/HYLiang/ExtremeWeather_SSL_Ensemble_Qoder/logs/test5_%j.err
set -euo pipefail
PROJECT=/home/group3/group3_work/HYLiang/ExtremeWeather_SSL_Ensemble_Qoder
source /home/group3/miniconda3/etc/profile.d/conda.sh
conda activate ci_det
cd "$PROJECT"
export PYTHONPATH="$PROJECT"
for s in 0 1 2 3 4; do
  if [ $s -eq 0 ]; then CK=experiments/p1_cnn/best.pt; else CK=experiments/p1_seed$s/best.pt; fi
  python tools/predict_cnn.py --ckpt $CK --split test_eval --tta --out experiments/final/p1_seed${s}_test_tta.npz
done
echo "[DONE] $(date -Is)"
