#!/usr/bin/env bash
#SBATCH --job-name=ew_seeds
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --time=05:00:00
#SBATCH --requeue
#SBATCH --output=/home/group3/group3_work/HYLiang/ExtremeWeather_SSL_Ensemble_Qoder/logs/seeds_%j.out
#SBATCH --error=/home/group3/group3_work/HYLiang/ExtremeWeather_SSL_Ensemble_Qoder/logs/seeds_%j.err
set -euo pipefail
PROJECT=/home/group3/group3_work/HYLiang/ExtremeWeather_SSL_Ensemble_Qoder
source /home/group3/miniconda3/etc/profile.d/conda.sh
conda activate ci_det
cd "$PROJECT"
export PYTHONPATH="$PROJECT"
echo "[JOB] $SLURM_JOB_ID $(date -Is)"
for s in 1 2 3 4; do
  python tools/train_cnn.py --epochs 6 --batch 64 --workers 12 --val-ts 20 --minutes 45 --seed $s --out experiments/p1_seed$s
done
for s in 0 1 2 3 4; do
  if [ $s -eq 0 ]; then CK=experiments/p1_cnn/best.pt; else CK=experiments/p1_seed$s/best.pt; fi
  python tools/predict_cnn.py --ckpt $CK --split dev --tta --out experiments/p1_seed${s}_dev_tta.npz
  python tools/predict_cnn.py --ckpt $CK --split val --tta --out experiments/p1_seed${s}_bench_tta.npz
done
echo "[DONE] $(date -Is)"
