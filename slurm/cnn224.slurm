#!/usr/bin/env bash
#SBATCH --job-name=ew_cnn224
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --output=/home/group3/group3_work/HYLiang/ExtremeWeather_SSL_Ensemble_Qoder/logs/cnn224_%j.out
#SBATCH --error=/home/group3/group3_work/HYLiang/ExtremeWeather_SSL_Ensemble_Qoder/logs/cnn224_%j.err
set -euo pipefail
PROJECT=/home/group3/group3_work/HYLiang/ExtremeWeather_SSL_Ensemble_Qoder
source /home/group3/miniconda3/etc/profile.d/conda.sh
conda activate ci_det
cd "$PROJECT"
export PYTHONPATH="$PROJECT"
echo "[JOB] $SLURM_JOB_ID $(date -Is)"
python tools/train_cnn.py --size 224 --epochs 6 --batch 32 --workers 12 --val-ts 20 --minutes 150 --seed 0 --out experiments/p1_cnn224
echo "[DONE] $(date -Is)"
