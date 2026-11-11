#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

TRAIN_SEED="${1:-1}"
TRAIN_STEPS="${2:-600000}"
PYTHON_BIN="${PYTHON_BIN:-python}"
LOG_DIR="logs_iicms_contention3"
TAG="rate_contention_neighbor3_${TRAIN_STEPS}"
MLP_RUN="results/IICMSMobility/PureMobility/mappo/${TAG}_mlp_s${TRAIN_SEED}/run1"
TF_RUN="results/IICMSMobility/PureMobility/mappo/${TAG}_tf_actor_s${TRAIN_SEED}/run1"

if [[ ! "$TRAIN_SEED" =~ ^[0-9]+$ || ! "$TRAIN_STEPS" =~ ^[1-9][0-9]*$ ]]; then
  echo "Usage: bash resume_contention_600k.sh [seed] [steps]" >&2
  exit 2
fi

mkdir -p "$LOG_DIR"

if [[ -f "$MLP_RUN/models/actor.pt" ]]; then
  if [[ ! -f "$MLP_RUN/checkpoint_selection.json" ]]; then
    echo "===== resume: select MLP checkpoint ====="
    "$PYTHON_BIN" select_best_checkpoint.py \
      --run_dir "$MLP_RUN" \
      --seed_start 91000 \
      --episodes 20 \
      --horizon 100 \
      --metric system_se \
      --max_qos_violation_rate 0.005 \
      --qos_penalty_weight 0 \
      2>&1 | tee -a "$LOG_DIR/seed${TRAIN_SEED}_resume.log"
  fi

  if [[ ! -f "$MLP_RUN/evaluation_deterministic/run1.json" ]]; then
    echo "===== resume: evaluate MLP ====="
    "$PYTHON_BIN" evaluate_iicms.py \
      --run_dir "$MLP_RUN" \
      --seed_start 93000 \
      --episodes 20 \
      --horizon 100 \
      --phase test \
      --checkpoint best \
      --out "$MLP_RUN/evaluation_deterministic" \
      2>&1 | tee -a "$LOG_DIR/seed${TRAIN_SEED}_resume.log"
    "$PYTHON_BIN" evaluate_iicms.py \
      --run_dir "$MLP_RUN" \
      --seed_start 93000 \
      --episodes 20 \
      --horizon 100 \
      --phase test \
      --checkpoint best \
      --stochastic \
      --out "$MLP_RUN/evaluation_stochastic" \
      2>&1 | tee -a "$LOG_DIR/seed${TRAIN_SEED}_resume.log"
    "$PYTHON_BIN" evaluate_iicms.py \
      --run_dir "$MLP_RUN" \
      --seed_start 93000 \
      --episodes 20 \
      --horizon 100 \
      --phase test \
      --fixed_action CB_L5_S00 \
      --policy_label fixed_cb_l5 \
      --out "$MLP_RUN/baseline_fixed_cb_l5_deterministic" \
      2>&1 | tee -a "$LOG_DIR/seed${TRAIN_SEED}_resume.log"
  fi
else
  echo "MLP checkpoint not found: $MLP_RUN/models/actor.pt" >&2
  echo "Run the normal MLP+TF script instead." >&2
  exit 1
fi

if [[ -f "$TF_RUN/evaluation_deterministic/run1.json" ]]; then
  echo "TF result already exists: $TF_RUN"
else
  echo "===== resume: train and evaluate Controlled TF ====="
  CONTENTION_ALGOS=tf \
  bash run_contention_600k.sh "$TRAIN_SEED" "$TRAIN_STEPS" \
    2>&1 | tee -a "$LOG_DIR/seed${TRAIN_SEED}_resume.log"
fi
