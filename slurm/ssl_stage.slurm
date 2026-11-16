#!/usr/bin/env bash
# ==============================================================================
# Stage C: self-supervised pretraining + fine-tuning (one GPU)
#   C1 contrastive pretraining on unlabelled 3-4 月 windows
#   C2 fine-tuning on the labelled training set
# ==============================================================================
#SBATCH --job-name=ew_ssl
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --requeue
#SBATCH --output=/home/group3/group3_work/HYLiang/ExtremeWeather_SSL_Ensemble_Qoder/logs/ssl_%j.out
#SBATCH --error=/home/group3/group3_work/HYLiang/ExtremeWeather_SSL_Ensemble_Qoder/logs/ssl_%j.err

set -euo pipefail
PROJECT=/home/group3/group3_work/HYLiang/ExtremeWeather_SSL_Ensemble_Qoder
source /home/group3/miniconda3/etc/profile.d/conda.sh
conda activate ci_det
cd "$PROJECT"
export PYTHONPATH="$PROJECT"

echo "[JOB] $SLURM_JOB_ID on $(hostname) at $(date -Is)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

python tools/ssl_pretrain.py --n 30000 --epochs 6 --batch 96 --workers 12 --minutes 150 --out experiments/p2_ssl

python tools/train_cnn.py --init experiments/p2_ssl/encoder.pt --epochs 4 --batch 64 \
    --workers 12 --val-ts 20 --minutes 70 --out experiments/p2_ssl_ft

echo "[DONE] $(date -Is)"
