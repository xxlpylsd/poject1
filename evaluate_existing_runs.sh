#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
TAG="${TAG:-cap450_mlpopt_250k}"
SEEDS="${SEEDS:-1,2,3,4,5}"
FAMILIES="${FAMILIES:-mlp,tf_actor}"
RUN_SUFFIX="${RUN_SUFFIX:-run1}"
EVAL_SEED="${EVAL_SEED:-93000}"
EVAL_EPISODES="${EVAL_EPISODES:-20}"
EVAL_HORIZON="${EVAL_HORIZON:-100}"
EVAL_PHASE="${EVAL_PHASE:-test}"
CHECKPOINT="${CHECKPOINT:-actor}"
EVAL_BASELINES="${EVAL_BASELINES:-1}"
BASELINE_ACTION="${BASELINE_ACTION:-CB_L5_S00}"
BASELINE_LABEL="${BASELINE_LABEL:-fixed_cb_l5}"
PACKAGE_RESULTS="${PACKAGE_RESULTS:-1}"
PACKAGE_MODE="${PACKAGE_MODE:-results}"
PACKAGE_INCLUDE_TAGS="${PACKAGE_INCLUDE_TAGS:-}"
RESULT_ROOT="${RESULT_ROOT:-results/IICMSMobility/PureMobility/mappo}"

IFS=',' read -r -a seed_values <<< "$SEEDS"
IFS=',' read -r -a family_values <<< "$FAMILIES"

for seed in "${seed_values[@]}"; do
  seed="${seed//[[:space:]]/}"
  for family in "${family_values[@]}"; do
    family="${family//[[:space:]]/}"
    run_dir="${RESULT_ROOT}/${TAG}_${family}_s${seed}/${RUN_SUFFIX}"
    case "$CHECKPOINT" in
      actor) checkpoint_file="${run_dir}/models/actor.pt" ;;
      best) checkpoint_file="${run_dir}/models/actor_best.pt" ;;
      final) checkpoint_file="${run_dir}/models/actor_final.pt" ;;
      *)
        echo "[evaluate] invalid CHECKPOINT=$CHECKPOINT" >&2
        exit 2
        ;;
    esac
    if [[ ! -f "$checkpoint_file" ]]; then
      echo "[evaluate] missing checkpoint: $checkpoint_file" >&2
      echo "Set CHECKPOINT=actor for legacy runs without snapshots." >&2
      exit 2
    fi

    echo "===== evaluate ${family} seed ${seed}: deterministic ====="
    "$PYTHON_BIN" evaluate_iicms.py \
      --run_dir "$run_dir" \
      --seed_start "$EVAL_SEED" \
      --episodes "$EVAL_EPISODES" \
      --horizon "$EVAL_HORIZON" \
      --phase "$EVAL_PHASE" \
      --checkpoint "$CHECKPOINT" \
      --out "${run_dir}/evaluation_deterministic"

    echo "===== evaluate ${family} seed ${seed}: stochastic ====="
    "$PYTHON_BIN" evaluate_iicms.py \
      --run_dir "$run_dir" \
      --seed_start "$EVAL_SEED" \
      --episodes "$EVAL_EPISODES" \
      --horizon "$EVAL_HORIZON" \
      --phase "$EVAL_PHASE" \
      --checkpoint "$CHECKPOINT" \
      --stochastic \
      --out "${run_dir}/evaluation_stochastic"

    if [[ "$EVAL_BASELINES" == "1" ]]; then
      echo "===== evaluate ${family} seed ${seed}: ${BASELINE_LABEL} ====="
      "$PYTHON_BIN" evaluate_iicms.py \
        --run_dir "$run_dir" \
        --seed_start "$EVAL_SEED" \
        --episodes "$EVAL_EPISODES" \
        --horizon "$EVAL_HORIZON" \
        --phase "$EVAL_PHASE" \
        --fixed_action "$BASELINE_ACTION" \
        --policy_label "$BASELINE_LABEL" \
        --out "${run_dir}/baseline_${BASELINE_LABEL}_deterministic"
    fi
  done
done

if [[ "$PACKAGE_RESULTS" == "1" ]]; then
  include_args=()
  IFS=',' read -r -a extra_tags <<< "$PACKAGE_INCLUDE_TAGS"
  for extra_tag in "${extra_tags[@]}"; do
    if [[ -n "$extra_tag" ]]; then
      include_args+=(--include-tag "$extra_tag")
    fi
  done
  "$PYTHON_BIN" package_results.py \
    --tag "$TAG" \
    --log-dir "logs_iicms" \
    --mode "$PACKAGE_MODE" \
    "${include_args[@]}"
fi

echo "All evaluation runs completed."
