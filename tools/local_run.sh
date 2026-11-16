#!/usr/bin/env bash
# Local (non-Slurm) runner — an ADDITIVE convenience entry point.
#
# The original slurm/ and tools/*.sh scripts are untouched. This script runs the
# same commands serially on a single device, so the pipeline works on a laptop
# or a single-GPU box without Slurm and without the hardcoded GPU arrays.
#
#   bash tools/local_run.sh            # auto: self-check + panel, then full chain if data present
#   bash tools/local_run.sh check      # self-check only (no data needed)
#   bash tools/local_run.sh panel      # rebuild the forecaster panel (no data needed)
#   bash tools/local_run.sh full       # full re-run (requires the dataset)
#
# Overrides: PYTHON, WORKERS, GPUS, DEVICE, MINUTES_V1, MINUTES_V2
set -euo pipefail

PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT"
export PYTHONPATH="$PROJECT"

PY="${PYTHON:-python3}"
WORKERS="${WORKERS:-4}"
GPUS="${GPUS:-0}"
MINUTES_V1="${MINUTES_V1:-600}"
MINUTES_V2="${MINUTES_V2:-600}"
STAGE="${1:-auto}"

mkdir -p logs experiments/final reports/panels reports/figures

if [ -z "${DEVICE:-}" ]; then
  DEVICE="$("$PY" -c 'import torch;print("cuda:0" if torch.cuda.is_available() else "cpu")' 2>/dev/null || echo cpu)"
fi

echo "=============================================================="
echo "repo    : $PROJECT"
echo "python  : $PY"
echo "device  : $DEVICE   workers: $WORKERS   gpus: $GPUS"
echo "stage   : $STAGE"
echo "=============================================================="

run_check() { "$PY" tools/selfcheck.py; }

run_panel() {
  echo "[panel] rebuilding the forecaster panel from the shipped dense sample"
  "$PY" tools/make_panel_v2.py
  echo "[panel] wrote reports/panels/panel_v2.html and panel_v2.png"
}

has_data() {
  "$PY" - <<'EOF' >/dev/null 2>&1
import os, sys
from ewci import data
sys.exit(0 if os.path.exists(data.STORE_PATH) else 1)
EOF
}

run_full() {
  if ! has_data; then
    echo "[full] dataset not found at the path configured in ewci/data.py."
    echo "[full] see docs/LOCAL_RUN.md for what is required. Skipping."
    return 0
  fi

  echo "[1/8] physical features"
  "$PY" tools/build_features.py --split train     --out experiments/p0_gbdt/features_train.npz    --workers "$WORKERS"
  "$PY" tools/build_features.py --split dev       --out experiments/p0_gbdt/features_dev.npz      --workers "$WORKERS"
  "$PY" tools/build_features.py --split val       --out experiments/p0_gbdt/features_val.npz      --workers "$WORKERS"
  "$PY" tools/build_features.py --split test_eval --out experiments/final/features_test_eval.npz  --workers "$WORKERS"
  "$PY" tools/eval_frozen.py

  echo "[2/8] v1 CNN, five seeds serially"
  "$PY" tools/train_cnn.py --epochs 6 --batch 64 --workers "$WORKERS" --val-ts 20 \
      --minutes "$MINUTES_V1" --seed 0 --out experiments/p1_cnn
  for s in 1 2 3 4; do
    "$PY" tools/train_cnn.py --epochs 6 --batch 64 --workers "$WORKERS" --val-ts 20 \
        --minutes "$MINUTES_V1" --seed "$s" --out "experiments/p1_seed$s"
  done

  echo "[3/8] v1 TTA predictions (dev + benchmark)"
  for s in 0 1 2 3 4; do
    if [ "$s" -eq 0 ]; then CK=experiments/p1_cnn/best.pt; else CK="experiments/p1_seed$s/best.pt"; fi
    "$PY" tools/predict_cnn.py --ckpt "$CK" --split dev --tta --out "experiments/p1_seed${s}_dev_tta.npz"
    "$PY" tools/predict_cnn.py --ckpt "$CK" --split val --tta --out "experiments/p1_seed${s}_bench_tta.npz"
  done
  "$PY" tools/fit_calibration.py

  echo "[4/8] v1 frozen test"
  for s in 0 1 2 3 4; do
    if [ "$s" -eq 0 ]; then CK=experiments/p1_cnn/best.pt; else CK="experiments/p1_seed$s/best.pt"; fi
    "$PY" tools/predict_cnn.py --ckpt "$CK" --split test_eval --tta --out "experiments/final/p1_seed${s}_test_tta.npz"
  done
  "$PY" tools/final_test5.py

  echo "[5/8] v2 training, five seeds serially"
  for s in 0 1 2 3 4; do
    "$PY" tools/train_cnn_v2.py --size 224 --heatmap-size 28 --sigma 2.0 --lambda-hm 1.0 \
        --balanced-sampling --epochs 6 --batch 32 --workers "$WORKERS" --val-ts 20 \
        --minutes "$MINUTES_V2" --seed "$s" --out "experiments/p2_seed$s"
  done

  echo "[6/8] v2 TTA predictions (dev + benchmark + test)"
  for s in 0 1 2 3 4; do
    ck="experiments/p2_seed$s/best.pt"
    "$PY" tools/predict_cnn_v2.py --ckpt "$ck" --split dev --tta --size 224 --heatmap-size 28 \
        --batch 64 --workers "$WORKERS" --out "experiments/p2_seed${s}_dev_tta.npz"
    "$PY" tools/predict_cnn_v2.py --ckpt "$ck" --split val --tta --size 224 --heatmap-size 28 \
        --batch 64 --workers "$WORKERS" --out "experiments/p2_seed${s}_bench_tta.npz"
    "$PY" tools/predict_cnn_v2.py --ckpt "$ck" --split test_eval --tta --size 224 --heatmap-size 28 \
        --batch 64 --workers "$WORKERS" --out "experiments/final/p2_seed${s}_test_tta.npz"
  done
  "$PY" tools/eval_v2_ensemble.py

  echo "[7/8] object level: localisation, dense perception, gating, tiers"
  "$PY" tools/eval_localization_dual.py --split all --gpus "$GPUS"
  "$PY" tools/dense_inference_v2.py --split both --gpus "$GPUS" --stride 56
  "$PY" tools/local_physics_gating.py
  "$PY" tools/eval_operational_tiers.py

  echo "[8/8] forecaster panel"
  "$PY" tools/extract_sample_dense.py --uid test/435/SCW2021-0072 --device "$DEVICE"
  "$PY" tools/make_panel_v2.py
  echo "[done] full chain finished"
}

case "$STAGE" in
  check) run_check ;;
  panel) run_panel ;;
  full)  run_full ;;
  auto)  run_check; run_panel; run_full ;;
  *) echo "usage: $0 [auto|check|panel|full]"; exit 2 ;;
esac