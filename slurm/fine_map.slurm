#!/usr/bin/env bash
#SBATCH --job-name=ew_finemap
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=/home/group3/group3_work/HYLiang/ExtremeWeather_SSL_Ensemble_Qoder/logs/finemap_%j.out
#SBATCH --error=/home/group3/group3_work/HYLiang/ExtremeWeather_SSL_Ensemble_Qoder/logs/finemap_%j.err
set -euo pipefail
PROJECT=/home/group3/group3_work/HYLiang/ExtremeWeather_SSL_Ensemble_Qoder
source /home/group3/miniconda3/etc/profile.d/conda.sh
conda activate ci_det
cd "$PROJECT"
export PYTHONPATH="$PROJECT"
python tools/fine_map.py --role test --stride 56 --out experiments/final/fine_map_test.png
python tools/fine_map.py --role val --uid val/140/SCW2021-0065 --stride 56 --out experiments/final/fine_map_bench.png
echo "[DONE] $(date -Is)"
