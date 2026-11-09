# Repository Guide

## Purpose

This repository trains and evaluates interference-coordination mode selection
for a clustered cell-free network. Each UE selects JT, CB, or NC, a cluster
size, and an AP subset. MAPPO is the core algorithm.

The public repository is:

```text
https://github.com/xxlpylsd/poject1
```

## Execution Flow

1. `train_main.py` parses `config.py`, creates train/eval environments, resolves
   observation and action metadata, writes provenance, and starts a Runner.
2. `envs/env_wrappers.py` creates Dummy or Subprocess vector environments.
3. `runner/shared/env_runner.py` is the main shared-policy MAPPO loop.
   `runner/separated/env_runner.py` supports one policy per UE.
4. Each rollout step records observations, actions, log-probabilities, rewards,
   values, masks, and RNN state in the replay buffer.
5. `algorithms/algorithm/r_mappo.py` computes GAE returns and performs PPO
   updates.
6. `utils/policy_evaluation.py` runs deterministic fixed-seed validation and
   writes `validation_curve.csv`, `validation_history.json`, and
   `validation_convergence.json`.
7. `select_best_checkpoint.py` evaluates snapshots and installs the selected
   pair as `actor_best.pt` / `critic_best.pt`.
8. `evaluate_iicms.py` evaluates learned policies or heuristic baselines on
   fixed test seeds.
9. Plotting and packaging scripts consume the saved CSV, JSON, NPZ, and model
   artifacts.

## Key Source Map

| Path | Responsibility |
|---|---|
| `config.py` | CLI parameters and defaults |
| `train_main.py` | Training entry point and provenance |
| `run_iicms_comparison.sh` / `.ps1` | End-to-end experiment orchestration |
| `algorithms/algorithm/r_mappo.py` | PPO/GAE update |
| `algorithms/algorithm/rMAPPOPolicy.py` | Actor/critic wrapper |
| `algorithms/algorithm/r_actor_critic.py` | Actor/critic assembly |
| `algorithms/utils/act.py` | Joint-action distributions and masks |
| `algorithms/utils/mlp.py` | MLP backbone |
| `algorithms/utils/transformer.py` | Transformer backbone |
| `envs/env_iicms_mobile.py` | Main mobility environment |
| `envs/env_core.py` | Channel and rate model |
| `runner/shared/` | Shared-policy MAPPO loop |
| `runner/separated/` | Independent-policy MAPPO loop |
| `utils/policy_evaluation.py` | Shared validation evaluator |
| `utils/shared_buffer.py` | Shared-policy replay buffer |
| `utils/separated_buffer.py` | Independent-policy replay buffer |
| `evaluate_iicms.py` | Final policy and baseline evaluation |
| `select_best_checkpoint.py` | Validation checkpoint selection |
| `tune_rssi_thresholds.py` | RSSI threshold calibration |
| `run_baseline_suite.sh` | Full heuristic baseline suite |
| `plot_*.py` | Training, validation, reward, and CDF figures |
| `package_project.py` | Source manifest and release archives |
| `package_results.py` | Review-result bundles |
| `tests/` | Unit and protocol tests |

## Modeling Invariants

- 30 APs, 30 UEs, 4 antennas per AP, 20 MHz bandwidth, 1.9 GHz carrier.
- AP placement is random in a 1 km x 1 km area with at least 50 m spacing.
- UE mobility uses RWP at 15 m/s with a 0.5 s decision interval.
- Shadowing is enabled; CSI aging is disabled in the main experiments.
- Backhaul capacity is 550-650 Mbps per AP with bottleneck JT accounting.
- The finite fronthaul queue is 0.5 s at the AP service rate.
- The full action space has 53 valid joint actions over JT/CB/NC and L=2..5.
- Observations contain four frames of one UE token and five AP tokens.
- Actor observation size is 168; centralized critic state size is 5040.
- The main reward is `team_rate`, the system sum rate in Mbps.
- A zero QoS weight means pure-rate training; P05 and QoS remain diagnostics.

## Evaluation Invariants

- Validation seeds: `91000-91019`.
- Hyperparameter validation seeds: `92000-92019`.
- Final test seeds: `93000-93019`.
- Do not overlap checkpoint selection, tuning, and test seed panels.
- Training validation uses 20 deterministic episodes every 25k steps.
- Checkpoint selection uses 20 deterministic episodes.
- Pure-rate checkpoint selection uses `system_se`; never use QoS-first
  selection unless a QoS penalty is part of the training objective.
- Report deterministic execution as the primary policy result.
- Report stochastic execution separately.
- Never compare curves from different seed panels or different environment
  configurations as if they were the same experiment.

## Artifact Rules

- `results/` contains generated run data; do not modify it by hand.
- Model weights and raw `reward_proposed.npy` files stay out of normal commits.
- Raw reward plotting requires `--source raw`; `curve.csv` is not a substitute.
- Checkpoint selection replaces `actor.pt` and `critic.pt` with the selected
  model while preserving `actor_final.pt` and `critic_final.pt`.
- Keep `config.json`, `source_sha256.json`, and test protocol IDs with results.
- Plot scripts must preserve the exact metric and execution-mode label.
- Do not silently mix deterministic and stochastic results.

## Required Checks

Run from the repository root:

```bash
python -m unittest discover -s tests -v
```

Before any training change, also run a short smoke experiment and inspect:

```text
logs/curve.csv
logs/validation_curve.csv
checkpoint_selection.json
evaluation_deterministic/run1.json
```

## Contribution Workflow

Changes to the shared repository should use a branch and Pull Request.

Suggested branch names:

```text
feature/<short-name>
fix/<short-name>
docs/<short-name>
```

Workflow:

1. Branch from the current `main`.
2. Make focused changes.
3. Run the full test suite.
4. Commit with a clear message.
5. Push the branch.
6. Open a Pull Request against `main`.
7. Merge only after tests pass and the change is reviewed.

Do not mix unrelated experiments, large generated artifacts, and source edits in
one Pull Request.

## Permission Model

The repository is public, so anyone can view, clone, fork, and open a Pull
Request. Only the owner and explicitly added collaborators with write access
can push to the original repository. The repository currently has no declared
license; adding a reuse license is a separate owner decision.
