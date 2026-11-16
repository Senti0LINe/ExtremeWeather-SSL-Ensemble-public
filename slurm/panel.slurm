#!/usr/bin/env bash
#SBATCH --job-name=ew_panel
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=/home/group3/group3_work/HYLiang/ExtremeWeather_SSL_Ensemble_Qoder/logs/panel_%j.out
#SBATCH --error=/home/group3/group3_work/HYLiang/ExtremeWeather_SSL_Ensemble_Qoder/logs/panel_%j.err
set -euo pipefail
PROJECT=/home/group3/group3_work/HYLiang/ExtremeWeather_SSL_Ensemble_Qoder
source /home/group3/miniconda3/etc/profile.d/conda.sh
conda activate ci_det
cd "$PROJECT"
export PYTHONPATH="$PROJECT"
python tools/make_panel.py --role test --stride 56 --device cuda --out reports/panel_test5
python tools/make_panel.py --role val --uid val/140/SCW2021-0065 --stride 56 --device cuda --out reports/panel_bench5
echo "[DONE] $(date -Is)"
