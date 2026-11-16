#!/usr/bin/env bash
# Fixed-pipeline orchestrator (EXPERIMENTS E14 repairs, stage 1+2).
#
# Retrains CINet_v2 with the corrected heatmap supervision, then re-runs the
# whole evaluation chain with operating points selected on dev only.
#
# Uses explicit GPU assignment rather than Slurm so that cards already busy
# with other users' work can be avoided. Override with EWCI_GPUS.
#
#   nohup bash tools/run_fixed_pipeline.sh > logs/fixed_pipeline.log 2>&1 &
#
# Stages: train -> TTA inference -> window ensemble -> dense -> localisation
#         -> gating -> tiers -> panel. Each stage logs separately and the
#         orchestrator continues past a failed stage so the rest is still usable.
set -uo pipefail

PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT"
export PYTHONPATH="$PROJECT"

read -r -a GPUS <<< "${EWCI_GPUS:-1 2 5 6 7}"
NG=${#GPUS[@]}
PY="${PYTHON:-python}"
WORKERS="${WORKERS:-8}"
MINUTES="${MINUTES:-120}"
FROM="${FROM:-1}"   # resume from this stage (1..8); earlier stages are skipped
GPU_LIST="$(IFS=,; echo "${GPUS[*]}")"

mkdir -p logs experiments/final reports/panels reports/figures
STAMP="$(date +%Y%m%d_%H%M%S)"
FAILED=()

say() { echo "[$(date +%H:%M:%S)] $*"; }

stage() {  # stage <name> <logfile> <cmd...>
  local name="$1" log="$2"; shift 2
  say "START $name  (log: $log)"
  local t0=$SECONDS
  if "$@" > "$log" 2>&1; then
    say "DONE  $name  in $(( (SECONDS-t0)/60 )) min"
  else
    say "FAIL  $name  (exit $?) — see $log"
    FAILED+=("$name")
  fi
}

say "=============================================================="
say "fixed pipeline | GPUs: ${GPUS[*]} | workers $WORKERS | budget ${MINUTES}min/seed | from stage $FROM"
say "=============================================================="

if [ "$FROM" -le 1 ]; then
# ---------------------------------------------------------------- 1. train
say "STAGE 1/8 — retrain v2, five seeds in parallel"
pids=()
for i in 0 1 2 3 4; do
  gpu="${GPUS[$((i % NG))]}"
  CUDA_VISIBLE_DEVICES="$gpu" $PY tools/train_cnn_v2.py \
      --size 224 --heatmap-size 28 --sigma 2.0 --lambda-hm 1.0 \
      --balanced-sampling --epochs 6 --batch 32 --workers "$WORKERS" \
      --val-ts 20 --minutes "$MINUTES" --seed "$i" \
      --out "experiments/p2_seed$i" \
      > "logs/fixed_train_seed${i}_${STAMP}.log" 2>&1 &
  pids+=($!)
  say "  seed $i -> GPU $gpu (pid $!)"
done
rc=0
for p in "${pids[@]}"; do wait "$p" || rc=1; done
[ "$rc" -eq 0 ] && say "STAGE 1 done in $((SECONDS/60)) min" || { say "STAGE 1 had failures"; FAILED+=("train"); }

for i in 0 1 2 3 4; do
  [ -f "experiments/p2_seed$i/best.pt" ] || { say "ABORT: seed $i checkpoint missing"; exit 1; }
done
fi

if [ "$FROM" -le 2 ]; then
# ------------------------------------------------------- 2. TTA inference
say "STAGE 2/8 — TTA inference, 15 jobs (5 seeds x 3 splits) over $NG GPUs"
jobs_list=()
for s in 0 1 2 3 4; do
  jobs_list+=("$s|dev|experiments/p2_seed${s}_dev_tta.npz")
  jobs_list+=("$s|val|experiments/p2_seed${s}_bench_tta.npz")
  jobs_list+=("$s|test_eval|experiments/final/p2_seed${s}_test_tta.npz")
done
idx=0
while [ "$idx" -lt "${#jobs_list[@]}" ]; do
  pids=()
  for k in $(seq 0 $((NG-1))); do
    j=$((idx+k))
    [ "$j" -ge "${#jobs_list[@]}" ] && break
    IFS='|' read -r s sp out <<< "${jobs_list[$j]}"
    gpu="${GPUS[$k]}"
    CUDA_VISIBLE_DEVICES="$gpu" $PY tools/predict_cnn_v2.py \
        --ckpt "experiments/p2_seed$s/best.pt" --split "$sp" --tta \
        --size 224 --heatmap-size 28 --batch 64 --workers 6 --out "$out" \
        > "logs/fixed_infer_s${s}_${sp}_${STAMP}.log" 2>&1 &
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait "$p" || FAILED+=("infer"); done
  idx=$((idx+NG))
  say "  inference progress: $idx / ${#jobs_list[@]}"
done
fi

if [ "$FROM" -le 3 ]; then
# ------------------------------------------------- 3. window-level ensemble
stage "window_ensemble" "logs/fixed_ensemble_${STAMP}.log" \
      $PY tools/eval_v2_ensemble.py
fi

# ------------------------------------------------------- 4. dense perception
stage "dense_all_splits" "logs/fixed_dense_${STAMP}.log" \
      $PY tools/dense_inference_v2.py --split all --gpus "$GPU_LIST" --stride 56

# --------------------------------------------------------- 5. localisation
stage "localisation_dual" "logs/fixed_localization_${STAMP}.log" \
      $PY tools/eval_localization_dual.py --split all --gpus "$GPU_LIST"

# -------------------------------------------------------------- 6. gating
stage "physics_gating" "logs/fixed_gating_${STAMP}.log" \
      $PY tools/local_physics_gating.py

# --------------------------------------------------------------- 7. tiers
stage "operational_tiers" "logs/fixed_tiers_${STAMP}.log" \
      $PY tools/eval_operational_tiers.py

# --------------------------------------------------------------- 8. panel
stage "dense_sample" "logs/fixed_sample_${STAMP}.log" \
      $PY tools/extract_sample_dense.py --uid test/435/SCW2021-0072 --device "cuda:${GPUS[0]}"
stage "panel_v2" "logs/fixed_panel_${STAMP}.log" \
      env MPLBACKEND=Agg $PY tools/make_panel_v2.py

say "=============================================================="
if [ "${#FAILED[@]}" -eq 0 ]; then
  say "ALL STAGES OK"
else
  say "FAILED STAGES: ${FAILED[*]}"
fi
say "total elapsed: $((SECONDS/60)) min"
say "=============================================================="