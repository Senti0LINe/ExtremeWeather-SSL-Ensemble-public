#!/usr/bin/env bash
# Parallel 5-Seed TTA Inference for CINet_v2 across GPUs 1, 2, 4, 5, 6

set -euo pipefail

PROJECT=/home/group3/group3_work/HYLiang/ExtremeWeather_SSL_Ensemble_Qoder
source /home/group3/miniconda3/etc/profile.d/conda.sh
conda activate ci_det
cd "$PROJECT"
export PYTHONPATH="$PROJECT"

mkdir -p "$PROJECT/logs" "$PROJECT/experiments/final"

GPUS=(1 2 4 5 6)
SEEDS=(0 1 2 3 4)

echo "=================================================="
echo "Starting Parallel 5-Seed TTA Inference"
echo "Date: $(date -Is)"
echo "GPUs: ${GPUS[*]}"
echo "=================================================="

run_seed() {
    local s="$1"
    local gpu="$2"
    local ckpt="$PROJECT/experiments/p2_seed${s}/best.pt"
    
    echo "[SEED $s | GPU $gpu] Starting TTA inference on dev, bench, test_eval..."
    
    # 1. Dev TTA
    CUDA_VISIBLE_DEVICES="$gpu" python tools/predict_cnn_v2.py \
        --ckpt "$ckpt" \
        --split dev \
        --tta \
        --size 224 \
        --heatmap-size 28 \
        --batch 64 \
        --workers 6 \
        --out "$PROJECT/experiments/p2_seed${s}_dev_tta.npz"
        
    # 2. Bench TTA
    CUDA_VISIBLE_DEVICES="$gpu" python tools/predict_cnn_v2.py \
        --ckpt "$ckpt" \
        --split val \
        --tta \
        --size 224 \
        --heatmap-size 28 \
        --batch 64 \
        --workers 6 \
        --out "$PROJECT/experiments/p2_seed${s}_bench_tta.npz"
        
    # 3. Test TTA
    CUDA_VISIBLE_DEVICES="$gpu" python tools/predict_cnn_v2.py \
        --ckpt "$ckpt" \
        --split test_eval \
        --tta \
        --size 224 \
        --heatmap-size 28 \
        --batch 64 \
        --workers 6 \
        --out "$PROJECT/experiments/final/p2_seed${s}_test_tta.npz"
        
    echo "[SEED $s | GPU $gpu] Finished all splits successfully!"
}

PIDS=()
for i in "${!SEEDS[@]}"; do
    s="${SEEDS[$i]}"
    gpu="${GPUS[$i]}"
    run_seed "$s" "$gpu" > "$PROJECT/logs/infer_v2_seed${s}.log" 2>&1 &
    PIDS+=("$!")
done

echo "Launched 5 parallel inference processes with PIDs: ${PIDS[*]}"
for pid in "${PIDS[@]}"; do
    wait "$pid"
done

echo "All 5 seeds TTA inference completed at $(date -Is)!"
