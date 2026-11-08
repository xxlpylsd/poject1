#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_DIR="${RUN_DIR:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-}"
EVAL_SEED="${EVAL_SEED:-93000}"
EVAL_EPISODES="${EVAL_EPISODES:-20}"
EVAL_HORIZON="${EVAL_HORIZON:-100}"
RSSI_CLUSTER_SIZE="${RSSI_CLUSTER_SIZE:-5}"
RSSI_THRESHOLD_HI="${RSSI_THRESHOLD_HI:-}"
RSSI_THRESHOLD_LO="${RSSI_THRESHOLD_LO:-}"
RSSI_THRESHOLDS_FILE="${RSSI_THRESHOLDS_FILE:-}"
PLOT_RESULTS="${PLOT_RESULTS:-1}"
SEEDS="${SEEDS:-2,3,4,5,6}"
FAMILIES="${FAMILIES:-mlp,tf_actor}"
RESULT_ROOT="${RESULT_ROOT:-results/IICMSMobility/PureMobility/mappo}"

if [[ -n "$RSSI_THRESHOLDS_FILE" ]]; then
  if [[ ! -f "$RSSI_THRESHOLDS_FILE" ]]; then
    echo "Missing RSSI thresholds file: $RSSI_THRESHOLDS_FILE" >&2
    exit 2
  fi
  # The calibration utility writes shell-compatible assignments.
  source "$RSSI_THRESHOLDS_FILE"
fi
if [[ -z "$RSSI_THRESHOLD_LO" ||
      -z "$RSSI_THRESHOLD_HI" ]]; then
  echo "Provide RSSI_THRESHOLDS_FILE or both RSSI_THRESHOLD_LO/HI" >&2
  exit 2
fi
SC_RSSI_THRESHOLD_HI="${SC_RSSI_THRESHOLD_HI:-$RSSI_THRESHOLD_HI}"
SC_RSSI_THRESHOLD_LO="${SC_RSSI_THRESHOLD_LO:-$RSSI_THRESHOLD_LO}"
DC_RSSI_THRESHOLD_HI="${DC_RSSI_THRESHOLD_HI:-$RSSI_THRESHOLD_HI}"
DC_RSSI_THRESHOLD_LO="${DC_RSSI_THRESHOLD_LO:-$RSSI_THRESHOLD_LO}"

if [[ -z "$RUN_DIR" ]]; then
  echo "RUN_DIR must point to a trained run containing config.json" >&2
  exit 2
fi
if [[ ! -f "$RUN_DIR/config.json" ]]; then
  echo "Missing config: $RUN_DIR/config.json" >&2
  exit 2
fi
if [[ -z "$OUTPUT_ROOT" ]]; then
  OUTPUT_ROOT="${RUN_DIR}/baseline_suite"
fi
mkdir -p "$OUTPUT_ROOT"

run_baseline() {
  local label=$1
  shift
  echo "===== baseline ${label} ====="
  "$PYTHON_BIN" evaluate_iicms.py \
    --run_dir "$RUN_DIR" \
    --seed_start "$EVAL_SEED" \
    --episodes "$EVAL_EPISODES" \
    --horizon "$EVAL_HORIZON" \
    --phase test \
    --policy_label "$label" \
    --out "${OUTPUT_ROOT}/${label}" \
    "$@" > "${OUTPUT_ROOT}/${label}.log" 2>&1
}

# Fixed coordination modes are intentionally evaluated with static
# clustering only. The entire physical AP ranking is frozen at reset.
run_baseline sc_nc \
  --baseline_policy fixed \
  --cluster_policy static \
  --fixed_action NC

for size in 2 3 4 5; do
  run_baseline "sc_cb_l${size}" \
    --baseline_policy fixed \
    --cluster_policy static \
    --fixed_action "CB_L${size}_S00"
  run_baseline "sc_jt_l${size}" \
    --baseline_policy fixed \
    --cluster_policy static \
    --fixed_action "JT_L${size}_S00"
done

# Fixed-mode dynamic-clustering controls. Only the best size per mode is
# displayed in the final CDF, but all sizes are evaluated for selection.
run_baseline dc_nc \
  --baseline_policy fixed \
  --cluster_policy dynamic \
  --fixed_action NC

for size in 2 3 4 5; do
  run_baseline "dc_cb_l${size}" \
    --baseline_policy fixed \
    --cluster_policy dynamic \
    --fixed_action "CB_L${size}_S00"
  run_baseline "dc_jt_l${size}" \
    --baseline_policy fixed \
    --cluster_policy dynamic \
    --fixed_action "JT_L${size}_S00"
done

# The RSSI-threshold reference is compared under identical static and
# dynamic clustering rules.
run_baseline sc_rssi \
  --baseline_policy rssi \
  --cluster_policy static \
  --rssi_cluster_size "$RSSI_CLUSTER_SIZE" \
  --rssi_threshold_hi "$SC_RSSI_THRESHOLD_HI" \
  --rssi_threshold_lo "$SC_RSSI_THRESHOLD_LO"

run_baseline dc_rssi \
  --baseline_policy rssi \
  --cluster_policy dynamic \
  --rssi_cluster_size "$RSSI_CLUSTER_SIZE" \
  --rssi_threshold_hi "$DC_RSSI_THRESHOLD_HI" \
  --rssi_threshold_lo "$DC_RSSI_THRESHOLD_LO"

"$PYTHON_BIN" summarize_baseline_results.py \
  --root "$OUTPUT_ROOT" \
  --out-prefix "${OUTPUT_ROOT}/baseline_summary"

if [[ "$PLOT_RESULTS" == "1" ]]; then
  RUN_PARENT="$(dirname "$RUN_DIR")"
  DERIVED_TAG="$(basename "$RUN_PARENT" | sed -E 's/_(mlp|tf_actor|orig_mappo)_s[0-9]+$//')"
  TAG="${TAG:-$DERIVED_TAG}"
  FIGURE_DIR="${FIGURE_DIR:-${RESULT_ROOT}/${TAG}_figures}"
  mkdir -p "$FIGURE_DIR"
  "$PYTHON_BIN" plot_strategy_cdf.py \
    --tag "$TAG" \
    --seeds "$SEEDS" \
    --families "$FAMILIES" \
    --result-root "$RESULT_ROOT" \
    --run run1 \
    --baseline-suite-dir "$OUTPUT_ROOT" \
    --out-prefix "${FIGURE_DIR}/${TAG}_strategy_rate_cdf"
fi

echo "BASELINE_OUTPUT=${OUTPUT_ROOT}"
