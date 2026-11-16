#!/usr/bin/env bash
# Automated Post-Training Pipeline for CINet_v2

set -euo pipefail

PROJECT=/home/group3/group3_work/HYLiang/ExtremeWeather_SSL_Ensemble_Qoder
source /home/group3/miniconda3/etc/profile.d/conda.sh
conda activate ci_det
cd "$PROJECT"
export PYTHONPATH="$PROJECT"

echo "=================================================="
echo "[PIPELINE] Waiting for CINet_v2 training (seeds 0~4) to complete..."
echo "Start watching at: $(date -Is)"
echo "=================================================="

# Monitor training processes until all 5 complete
while true; do
    N_RUNNING=$(pgrep -f "tools/train_cnn_v2.py" | wc -l || true)
    if [ "$N_RUNNING" -eq 0 ]; then
        echo "[PIPELINE] All train_cnn_v2 processes have completed at $(date -Is)!"
        break
    fi
    echo "[PIPELINE] Training active ($N_RUNNING processes). Checking again in 30s..."
    sleep 30
done

# Verify all best.pt checkpoints exist
echo "[PIPELINE] Verifying checkpoints..."
for s in 0 1 2 3 4; do
    CKPT="$PROJECT/experiments/p2_seed${s}/best.pt"
    if [ ! -f "$CKPT" ]; then
        echo "[ERROR] Checkpoint $CKPT not found!"
        exit 1
    fi
    echo "  --> Found $CKPT ($(du -h "$CKPT" | cut -f1))"
done

# Step 1: Run 5-seed 4-flip TTA Inference in parallel
echo "=================================================="
echo "[PIPELINE] Launching 5-seed 4-flip TTA Inference across GPUs 1, 2, 4, 5, 6..."
echo "=================================================="
bash tools/predict_all_seeds_v2.sh

# Step 2: Run Ensemble Evaluation & Metrics Export
echo "=================================================="
echo "[PIPELINE] Running Heterogeneous Ensemble Evaluation (CNN + GBDT + Calibration)..."
echo "=================================================="
python tools/eval_v2_ensemble.py > "$PROJECT/logs/eval_v2_ensemble.log" 2>&1

echo "=================================================="
echo "[PIPELINE COMPLETED] End-to-end pipeline finished successfully at $(date -Is)!"
echo "Metrics written to: experiments/final/cinet_v2_ensemble_metrics.json"
echo "=================================================="
