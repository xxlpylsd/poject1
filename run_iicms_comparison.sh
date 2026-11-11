#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
SEEDS="${SEEDS:-1}"
STEPS="${STEPS:-100000}"
EPISODE_LENGTH="${EPISODE_LENGTH:-100}"
NENV="${NENV:-4}"
VEC_ENV="${VEC_ENV:-subproc}"
TAG="${TAG:-iicms_mobility}"
SPEED="${SPEED:-15}"
QOS_PENALTY_WEIGHT="${QOS_PENALTY_WEIGHT:-20}"
MIN_UE_SE_BPS_HZ="${MIN_UE_SE_BPS_HZ:-0.0}"
SLOT_DT="${SLOT_DT:-0.5}"
BACKHAUL_LO="${BACKHAUL_LO:-5.5e8}"
BACKHAUL_HI="${BACKHAUL_HI:-6.5e8}"
FRONTHAUL_QUEUE="${FRONTHAUL_QUEUE:-1}"
FRONTHAUL_BUFFER_S="${FRONTHAUL_BUFFER_S:-0.5}"
CLUSTER_SIZES="${CLUSTER_SIZES:-2,3,4,5}"
CLUSTER_ACTION_MODE="${CLUSTER_ACTION_MODE:-subset}"
CSI_AGING="${CSI_AGING:-0}"
OBS_INCLUDE_XY="${OBS_INCLUDE_XY:-1}"
OBS_FRAMES="${OBS_FRAMES:-4}"
OBS_NEIGHBOR_UES="${OBS_NEIGHBOR_UES:-0}"
OBS_AP_COMPETITION="${OBS_AP_COMPETITION:-0}"
REWARD_MODE="${REWARD_MODE:-mean_se}"
SHARED_LR="${LR:-1e-4}"
SHARED_CRITIC_LR="${CRITIC_LR:-1e-4}"
SHARED_ENTROPY_COEF="${ENTROPY_COEF:-0.01}"
MLP_LR="${MLP_LR:-2e-4}"
MLP_CRITIC_LR="${MLP_CRITIC_LR:-1e-4}"
MLP_ENTROPY_COEF="${MLP_ENTROPY_COEF:-0.005}"
MLP_HIDDEN="${MLP_HIDDEN:-192}"
MLP_CRITIC_BACKBONE="${MLP_CRITIC_BACKBONE:-mlp}"
MLP_ACTOR_TEMPORAL_CONTEXT="${MLP_ACTOR_TEMPORAL_CONTEXT:-history_mean}"
TF_LR="${TF_LR:-$SHARED_LR}"
TF_CRITIC_LR="${TF_CRITIC_LR:-$SHARED_CRITIC_LR}"
TF_ENTROPY_COEF="${TF_ENTROPY_COEF:-$SHARED_ENTROPY_COEF}"
TF_HIDDEN="${TF_HIDDEN:-128}"
TF_D_MODEL="${TF_D_MODEL:-64}"
TF_CRITIC_BACKBONE="${TF_CRITIC_BACKBONE:-transformer}"
TF_POOL="${TF_POOL:-mean}"
TF_USE_CLS="${TF_USE_CLS:-0}"
TF_ACTOR_TEMPORAL_CONTEXT="${TF_ACTOR_TEMPORAL_CONTEXT:-history_mean}"
PAPER_LR="${PAPER_LR:-5e-5}"
PAPER_CRITIC_LR="${PAPER_CRITIC_LR:-5e-5}"
PAPER_ENTROPY_COEF="${PAPER_ENTROPY_COEF:-0.01}"
PAPER_HIDDEN="${PAPER_HIDDEN:-64}"
PAPER_PPO_EPOCH="${PAPER_PPO_EPOCH:-15}"
BACKHAUL_JT_MODEL="${BACKHAUL_JT_MODEL:-bottleneck}"
JOINT_ACTION_HEAD="${JOINT_ACTION_HEAD:-subset_conditional}"
ARCH_PROFILE="${ARCH_PROFILE:-controlled}"
TF_RELATION_DIM="${TF_RELATION_DIM:-32}"
USE_GPU="${USE_GPU:-1}"
PACKAGE_RESULTS="${PACKAGE_RESULTS:-1}"
PACKAGE_MODE="${PACKAGE_MODE:-results}"
PACKAGE_INCLUDE_TAGS="${PACKAGE_INCLUDE_TAGS:-}"
PLOT_RESULTS="${PLOT_RESULTS:-1}"
PLOT_METRIC="${PLOT_METRIC:-se}"
FIGURE_DIR="${FIGURE_DIR:-}"
SAVE_INTERVAL="${SAVE_INTERVAL:-10}"
AUTO_EVAL="${AUTO_EVAL:-1}"
ALGOS="${ALGOS:-both}"
TRAINING_VALIDATION="${TRAINING_VALIDATION:-1}"
VALIDATION_INTERVAL_STEPS="${VALIDATION_INTERVAL_STEPS:-25000}"
VALIDATION_EPISODES="${VALIDATION_EPISODES:-20}"
VALIDATION_SEED="${VALIDATION_SEED:-91000}"
VALIDATION_HORIZON="${VALIDATION_HORIZON:-0}"
VALIDATION_PLATEAU_RELATIVE_GAIN="${VALIDATION_PLATEAU_RELATIVE_GAIN:-0.005}"
VALIDATION_WINDOW="${VALIDATION_WINDOW:-3}"
SAVE_CHECKPOINT_SNAPSHOTS="${SAVE_CHECKPOINT_SNAPSHOTS:-1}"
SELECT_BEST_CHECKPOINT="${SELECT_BEST_CHECKPOINT:-1}"
BEST_CHECKPOINT_METRIC="${BEST_CHECKPOINT_METRIC:-auto}"
BEST_CHECKPOINT_MAX_QOS="${BEST_CHECKPOINT_MAX_QOS:-0.005}"
BEST_CHECKPOINT_EPISODES="${BEST_CHECKPOINT_EPISODES:-20}"
BEST_CHECKPOINT_SEED="${BEST_CHECKPOINT_SEED:-91000}"
EVAL_SEED="${EVAL_SEED:-93000}"
EVAL_EPISODES="${EVAL_EPISODES:-20}"
EVAL_HORIZON="${EVAL_HORIZON:-100}"
EVAL_PHASE="${EVAL_PHASE:-test}"
EVAL_BASELINES="${EVAL_BASELINES:-1}"
BASELINE_ACTION="${BASELINE_ACTION:-CB_L5_S00}"
BASELINE_LABEL="${BASELINE_LABEL:-fixed_cb_l5}"
LOG_DIR="${LOG_DIR:-logs_iicms}"
SEED_LIST="${SEEDS// /}"
SEEDS="${SEED_LIST//,/ }"
if [[ "$ARCH_PROFILE" != "controlled" &&
      "$ARCH_PROFILE" != "native" &&
      "$ARCH_PROFILE" != "direct" &&
      "$ARCH_PROFILE" != "main" ]]; then
  echo "[architecture] ARCH_PROFILE must be controlled, native, direct, or main" >&2
  exit 2
fi
echo "[architecture] profile=${ARCH_PROFILE}"
if [[ "$BEST_CHECKPOINT_METRIC" == "auto" ]]; then
  if [[ "$QOS_PENALTY_WEIGHT" =~ ^([+-]?)0([.]0*)?$ ]]; then
    BEST_CHECKPOINT_METRIC="system_se"
  else
    BEST_CHECKPOINT_METRIC="qos_constrained"
  fi
fi
echo "[validation] enabled=${TRAINING_VALIDATION} interval_steps=${VALIDATION_INTERVAL_STEPS} episodes=${VALIDATION_EPISODES} seed_start=${VALIDATION_SEED}"
echo "[checkpoint] metric=${BEST_CHECKPOINT_METRIC} episodes=${BEST_CHECKPOINT_EPISODES}"
if [[ -z "$FIGURE_DIR" ]]; then
  FIGURE_DIR="results/IICMSMobility/PureMobility/mappo/${TAG}_figures"
fi

selection_end=$((BEST_CHECKPOINT_SEED + BEST_CHECKPOINT_EPISODES - 1))
evaluation_end=$((EVAL_SEED + EVAL_EPISODES - 1))
if [[ "$AUTO_EVAL" == "1" &&
      "$SELECT_BEST_CHECKPOINT" == "1" &&
      "$EVAL_PHASE" != "diagnostic" ]] &&
   (( BEST_CHECKPOINT_SEED <= evaluation_end &&
      EVAL_SEED <= selection_end )); then
  echo "[evaluation] checkpoint-selection seeds overlap test seeds:" >&2
  echo "  selection=${BEST_CHECKPOINT_SEED}..${selection_end}" >&2
  echo "  evaluation=${EVAL_SEED}..${evaluation_end}" >&2
  echo "Set EVAL_SEED to a disjoint range." >&2
  exit 2
fi

export PYTHONIOENCODING=utf-8
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

package_on_exit() {
  local status=$?
  if [[ "$PACKAGE_RESULTS" == "1" ]]; then
    local include_args=()
    local extra_tag
    local old_ifs="$IFS"
    IFS=','
    for extra_tag in $PACKAGE_INCLUDE_TAGS; do
      if [[ -n "$extra_tag" ]]; then
        include_args+=(--include-tag "$extra_tag")
      fi
    done
    IFS="$old_ifs"
    "$PYTHON_BIN" package_results.py \
      --tag "$TAG" \
      --log-dir "$LOG_DIR" \
      --mode "$PACKAGE_MODE" \
      "${include_args[@]}" || \
      echo "[package] packaging failed; training status=${status}" >&2
  fi
  exit "$status"
}
trap package_on_exit EXIT

mkdir -p "$LOG_DIR"

evaluate_run() {
  local run_dir=$1
  local label=$2
  local allow_best_selection=${3:-1}
  local checkpoint_role=actor
  if [[ "$SELECT_BEST_CHECKPOINT" == "1" &&
        "$allow_best_selection" == "1" ]]; then
    checkpoint_role=best
    echo "===== select best checkpoint: ${label} ====="
    "$PYTHON_BIN" select_best_checkpoint.py \
      --run_dir "$run_dir" \
      --seed_start "$BEST_CHECKPOINT_SEED" \
      --episodes "$BEST_CHECKPOINT_EPISODES" \
      --horizon "$EVAL_HORIZON" \
      --metric "$BEST_CHECKPOINT_METRIC" \
      --max_qos_violation_rate "$BEST_CHECKPOINT_MAX_QOS" \
      --qos_penalty_weight "$QOS_PENALTY_WEIGHT"
  fi

  echo "===== evaluate ${label}: deterministic ====="
  "$PYTHON_BIN" evaluate_iicms.py \
    --run_dir "$run_dir" \
    --seed_start "$EVAL_SEED" \
    --episodes "$EVAL_EPISODES" \
    --horizon "$EVAL_HORIZON" \
    --phase "$EVAL_PHASE" \
    --checkpoint "$checkpoint_role" \
    --out "${run_dir}/evaluation_deterministic"

  echo "===== evaluate ${label}: stochastic ====="
  "$PYTHON_BIN" evaluate_iicms.py \
    --run_dir "$run_dir" \
    --seed_start "$EVAL_SEED" \
    --episodes "$EVAL_EPISODES" \
    --horizon "$EVAL_HORIZON" \
    --phase "$EVAL_PHASE" \
    --checkpoint "$checkpoint_role" \
    --stochastic \
    --out "${run_dir}/evaluation_stochastic"

  if [[ "$EVAL_BASELINES" == "1" ]]; then
    echo "===== evaluate ${label}: ${BASELINE_LABEL} ====="
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
}

common=(
  --env_name IICMSMobility
  --scenario_name PureMobility
  --mobility rwp
  --ue_speed "$SPEED"
  --slot_dt "$SLOT_DT"
  --pathloss_model legacy
  --csi_aging "$CSI_AGING"
  --obs_include_xy "$OBS_INCLUDE_XY"
  --obs_frames "$OBS_FRAMES"
  --obs_neighbor_ues "$OBS_NEIGHBOR_UES"
  --obs_ap_competition "$OBS_AP_COMPETITION"
  --cluster_sizes "$CLUSTER_SIZES"
  --cluster_action_mode "$CLUSTER_ACTION_MODE"
  --reward_mode "$REWARD_MODE"
  --objective_horizon "$EPISODE_LENGTH"
  --reward_phi1 0.5
  --reward_phi2 1.0
  --rate_req_lo_mbps 40
  --rate_req_hi_mbps 40
  --min_ue_se_bps_hz "$MIN_UE_SE_BPS_HZ"
  --qos_penalty_weight "$QOS_PENALTY_WEIGHT"
  --backhaul_jt_model "$BACKHAUL_JT_MODEL"
  --num_agents 30
  --num_aps 30
  --antennas_per_ap 4
  --map_size 1000
  --ap_min_distance 50
  --backhaul_lo "$BACKHAUL_LO"
  --backhaul_hi "$BACKHAUL_HI"
  --fronthaul_queue "$FRONTHAUL_QUEUE"
  --fronthaul_buffer_s "$FRONTHAUL_BUFFER_S"
  --episode_length "$EPISODE_LENGTH"
  --num_env_steps "$STEPS"
  --n_rollout_threads "$NENV"
  --vec_env "$VEC_ENV"
  --n_training_threads 4
  --ppo_epoch 5
  --save_interval "$SAVE_INTERVAL"
  --critic_hidden_size 128
  --tf_n_head 4
  --tf_n_layers 2
  --tf_ff_mult 4
  --tf_grad_checkpoint 1
)
if [[ "$TRAINING_VALIDATION" == "1" ]]; then
  common+=(
    --use_eval
    --validation_interval_steps "$VALIDATION_INTERVAL_STEPS"
    --validation_episodes "$VALIDATION_EPISODES"
    --validation_seed_start "$VALIDATION_SEED"
    --validation_horizon "$VALIDATION_HORIZON"
    --validation_plateau_relative_gain "$VALIDATION_PLATEAU_RELATIVE_GAIN"
    --validation_window "$VALIDATION_WINDOW"
  )
else
  common+=(--no_training_validation)
fi
if [[ "$SAVE_CHECKPOINT_SNAPSHOTS" == "1" ]]; then
  common+=(--save_checkpoint_snapshots)
fi

for seed in $SEEDS; do
  if [[ "$ALGOS" == "both" || "$ALGOS" == "mlp" ]]; then
    echo "===== seed ${seed}: MAPPO-MLP ====="
    mlp=(
      "${common[@]}"
      --actor_backbone mlp
      --critic_backbone "$MLP_CRITIC_BACKBONE"
      --hidden_size "$MLP_HIDDEN"
      --actor_temporal_context "$MLP_ACTOR_TEMPORAL_CONTEXT"
      --lr "$MLP_LR"
      --critic_lr "$MLP_CRITIC_LR"
      --entropy_coef "$MLP_ENTROPY_COEF"
      --seed "$seed"
      --experiment_name "${TAG}_mlp_s${seed}"
    )
    if [[ "$ARCH_PROFILE" == "native" ||
          "$ARCH_PROFILE" == "main" ]]; then
      mlp+=(
        --disable_actor_token_norm
        --joint_action_head mlp_factorized
      )
    elif [[ "$ARCH_PROFILE" == "direct" ]]; then
      mlp+=(
        --disable_actor_token_norm
        --joint_action_head flat
      )
    else
      mlp+=(--joint_action_head "$JOINT_ACTION_HEAD")
    fi
    if [[ "$USE_GPU" == "0" ]]; then
      mlp+=(--cuda)
    fi
    "$PYTHON_BIN" train_main.py "${mlp[@]}" \
      2>&1 | tee "${LOG_DIR}/${TAG}_mlp_s${seed}.log"

    if [[ "$AUTO_EVAL" == "1" ]]; then
      mlp_run_dir="results/IICMSMobility/PureMobility/mappo/${TAG}_mlp_s${seed}/run1"
      evaluate_run "$mlp_run_dir" "MAPPO-MLP seed ${seed}"
    fi
  fi

  if [[ "$ALGOS" == "both" || "$ALGOS" == "tf" ]]; then
    echo "===== seed ${seed}: MAPPO-Transformer Actor ====="
    tf=(
      "${common[@]}"
      --actor_backbone transformer
      --critic_backbone "$TF_CRITIC_BACKBONE"
      --hidden_size "$TF_HIDDEN"
      --tf_d_model "$TF_D_MODEL"
      --tf_pool "$TF_POOL"
      --actor_temporal_context "$TF_ACTOR_TEMPORAL_CONTEXT"
      --lr "$TF_LR"
      --critic_lr "$TF_CRITIC_LR"
      --entropy_coef "$TF_ENTROPY_COEF"
      --seed "$seed"
      --experiment_name "${TAG}_tf_actor_s${seed}"
    )
    if [[ "$ARCH_PROFILE" == "native" ||
          "$ARCH_PROFILE" == "direct" ]]; then
      tf+=(
        --joint_action_head relational_subset
        --tf_relation_dim "$TF_RELATION_DIM"
      )
    else
      tf+=(--joint_action_head "$JOINT_ACTION_HEAD")
    fi
    if [[ "$TF_USE_CLS" == "1" ]]; then
      tf+=(--tf_use_cls)
    fi
    if [[ "$USE_GPU" == "0" ]]; then
      tf+=(--cuda)
    fi
    "$PYTHON_BIN" train_main.py "${tf[@]}" \
      2>&1 | tee "${LOG_DIR}/${TAG}_tf_actor_s${seed}.log"

    if [[ "$AUTO_EVAL" == "1" ]]; then
      tf_run_dir="results/IICMSMobility/PureMobility/mappo/${TAG}_tf_actor_s${seed}/run1"
      evaluate_run "$tf_run_dir" "MAPPO-Transformer seed ${seed}"
    fi
  fi

  if [[ "$ALGOS" == "orig" ||
        "$ALGOS" == "original" ||
        "$ALGOS" == "all" ]]; then
    echo "===== seed ${seed}: Original-paper MAPPO ====="
    orig=(
      "${common[@]}"
      --actor_backbone mlp
      --critic_backbone mlp
      --disable_actor_token_norm
      --hidden_size "$PAPER_HIDDEN"
      --critic_hidden_size "$PAPER_HIDDEN"
      --lr "$PAPER_LR"
      --critic_lr "$PAPER_CRITIC_LR"
      --entropy_coef "$PAPER_ENTROPY_COEF"
      --ppo_epoch "$PAPER_PPO_EPOCH"
      --joint_action_head paper_factorized
      --separate_policy
      --original_mappo
      --seed "$seed"
      --experiment_name "${TAG}_orig_mappo_s${seed}"
    )
    if [[ "$USE_GPU" == "0" ]]; then
      orig+=(--cuda)
    fi
    "$PYTHON_BIN" train_main.py "${orig[@]}" \
      2>&1 | tee "${LOG_DIR}/${TAG}_orig_mappo_s${seed}.log"

    if [[ "$AUTO_EVAL" == "1" ]]; then
      orig_run_dir="results/IICMSMobility/PureMobility/mappo/${TAG}_orig_mappo_s${seed}/run1"
      evaluate_run \
        "$orig_run_dir" \
        "Original-paper MAPPO seed ${seed}" \
        0
    fi
  fi
done

if [[ "$PLOT_RESULTS" == "1" ]]; then
  mkdir -p "$FIGURE_DIR"
  plot_families=""
  if [[ "$ALGOS" == "both" || "$ALGOS" == "mlp" ]]; then
    plot_families="mlp"
  fi
  if [[ "$ALGOS" == "both" || "$ALGOS" == "tf" ]]; then
    if [[ -n "$plot_families" ]]; then
      plot_families="${plot_families},tf_actor"
    else
      plot_families="tf_actor"
    fi
  fi
  if [[ "$ALGOS" == "orig" ||
        "$ALGOS" == "original" ||
        "$ALGOS" == "all" ]]; then
    if [[ -n "$plot_families" ]]; then
      plot_families="${plot_families},orig_mappo"
    else
      plot_families="orig_mappo"
    fi
  fi

  if [[ -n "$plot_families" ]]; then
    echo "===== plotting training curves ====="
    "$PYTHON_BIN" plot_training_curves.py \
      --tag "$TAG" \
      --seeds "$SEED_LIST" \
      --families "$plot_families" \
      --metric "$PLOT_METRIC" \
      --source training \
      --run latest \
      --out "${FIGURE_DIR}/${TAG}_training_curves.png" || \
      echo "[plot] training-curve plotting failed" >&2

    echo "===== plotting reference-style reward convergence ====="
    "$PYTHON_BIN" plot_reward_convergence.py \
      --tag "$TAG" \
      --seeds "$SEED_LIST" \
      --families "$plot_families" \
      --episode-length "$EPISODE_LENGTH" \
      --source raw \
      --run latest \
      --out "${FIGURE_DIR}/${TAG}_reward_convergence.png" || \
      echo "[plot] reward-convergence plotting failed" >&2

    echo "===== plotting deterministic validation curves ====="
    "$PYTHON_BIN" plot_training_curves.py \
      --tag "$TAG" \
      --seeds "$SEED_LIST" \
      --families "$plot_families" \
      --metric "$PLOT_METRIC" \
      --source validation \
      --run latest \
      --out "${FIGURE_DIR}/${TAG}_validation_curves.png" || \
      echo "[plot] validation-curve plotting failed" >&2
    "$PYTHON_BIN" plot_training_curves.py \
      --tag "$TAG" \
      --seeds "$SEED_LIST" \
      --families "$plot_families" \
      --metric p05 \
      --source validation \
      --run latest \
      --out "${FIGURE_DIR}/${TAG}_validation_p05.png" || \
      echo "[plot] validation P05 plotting failed" >&2
    "$PYTHON_BIN" plot_training_curves.py \
      --tag "$TAG" \
      --seeds "$SEED_LIST" \
      --families "$plot_families" \
      --metric qos \
      --source validation \
      --run latest \
      --out "${FIGURE_DIR}/${TAG}_validation_qos.png" || \
      echo "[plot] validation QoS plotting failed" >&2

    if [[ "$AUTO_EVAL" == "1" ]]; then
      echo "===== plotting per-UE SE CDF ====="
      "$PYTHON_BIN" plot_seed_se_cdf.py \
        --tag "$TAG" \
        --seeds "$SEED_LIST" \
        --families "$plot_families" \
        --metric "$PLOT_METRIC" \
        --run latest \
        --out "${FIGURE_DIR}/${TAG}_${PLOT_METRIC}_cdf.png" || \
        echo "[plot] CDF plotting failed" >&2
    fi
  fi
fi

echo "All IICMS mobility runs completed."
