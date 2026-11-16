#!/usr/bin/env bash
#SBATCH --job-name=ew_v2_seeds
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --array=0-4
#SBATCH --output=/home/group3/group3_work/HYLiang/ExtremeWeather_SSL_Ensemble_Qoder/logs/train_v2_%A_%a.out
#SBATCH --error=/home/group3/group3_work/HYLiang/ExtremeWeather_SSL_Ensemble_Qoder/logs/train_v2_%A_%a.err

set -euo pipefail

PROJECT=/home/group3/group3_work/HYLiang/ExtremeWeather_SSL_Ensemble_Qoder
source /home/group3/miniconda3/etc/profile.d/conda.sh
conda activate ci_det
cd "$PROJECT"
export PYTHONPATH="$PROJECT"

SEED=${SLURM_ARRAY_TASK_ID:-${1:-0}}

echo "=================================================="
echo "[JOB START] Slurm Job: ${SLURM_JOB_ID:-standalone} | Array Task (Seed): $SEED"
echo "[HOST] $(hostname) | Date: $(date -Is)"
echo "[GPU] CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-ALL}"
echo "=================================================="

mkdir -p "$PROJECT/logs"
mkdir -p "$PROJECT/experiments/p2_seed${SEED}"

python tools/train_cnn_v2.py \
    --size 224 \
    --heatmap-size 28 \
    --sigma 2.0 \
    --lambda-hm 1.0 \
    --balanced-sampling \
    --epochs 6 \
    --batch 32 \
    --workers 8 \
    --val-ts 20 \
    --minutes 60 \
    --seed "$SEED" \
    --out "experiments/p2_seed${SEED}"

echo "[JOB COMPLETED] Seed $SEED finished successfully at $(date -Is)"
