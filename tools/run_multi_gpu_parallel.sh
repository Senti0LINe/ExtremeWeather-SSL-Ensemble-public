#!/usr/bin/env bash
# Direct Multi-GPU Parallel Launcher for CINet_v2 (Seeds 0~4 on GPUs 1~5)

set -euo pipefail

PROJECT=/home/group3/group3_work/HYLiang/ExtremeWeather_SSL_Ensemble_Qoder
source /home/group3/miniconda3/etc/profile.d/conda.sh
conda activate ci_det
cd "$PROJECT"
export PYTHONPATH="$PROJECT"

mkdir -p "$PROJECT/logs"

# Available idle GPUs on test-ai: 1, 2, 3, 4, 5
GPUS=(1 2 3 4 5)
SEEDS=(0 1 2 3 4)

echo "=================================================="
echo "Launching 5 Seeds in Parallel across GPUs: ${GPUS[*]}"
echo "Start time: $(date -Is)"
echo "=================================================="

PIDS=()
for i in "${!SEEDS[@]}"; do
    s="${SEEDS[$i]}"
    gpu="${GPUS[$i]}"
    log="$PROJECT/logs/direct_train_v2_seed${s}_gpu${gpu}.log"
    echo "Starting Seed $s on GPU $gpu -> logging to $log"
    
    CUDA_VISIBLE_DEVICES="$gpu" python tools/train_cnn_v2.py \
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
        --seed "$s" \
        --out "experiments/p2_seed${s}" > "$log" 2>&1 &
    
    PIDS+=("$!")
done

echo "All 5 training tasks launched in background with PIDs: ${PIDS[*]}"
echo "Use 'tail -f logs/direct_train_v2_seed*.log' or 'nvidia-smi' to monitor progress."
