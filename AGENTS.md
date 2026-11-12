# Repository Guide

## Purpose

Train and evaluate interference-coordination mode selection in a clustered
cell-free network. Each UE selects JT, CB, or NC and an AP subset. MAPPO is the
core algorithm.

Public repository:

```text
https://github.com/xxlpylsd/project1
```

## Execution Flow

1. `train_main.py` parses `config.py`, builds environments and policies, writes
   provenance, and starts a Runner.
2. `envs/env_wrappers.py` creates Dummy or Subprocess vector environments.
3. `runner/shared/env_runner.py` is the main shared-policy MAPPO loop;
   `runner/separated/env_runner.py` supports independent UE policies.
4. Rollouts record observations, joint actions, log-probabilities, rewards,
   values, masks, and RNN state.
5. `algorithms/algorithm/r_mappo.py` computes GAE returns and PPO updates.
6. `utils/policy_evaluation.py` runs fixed-seed validation and writes
   `validation_curve.csv`, `validation_history.json`, and
   `validation_convergence.json`.
7. `select_best_checkpoint.py` installs the selected model as
   `actor_best.pt` / `critic_best.pt` while preserving final weights.
8. `evaluate_iicms.py` evaluates learned policies and heuristic baselines.
9. Plot and packaging scripts consume generated CSV, JSON, NPZ, and model files.

## Source Map

| Path | Responsibility |
|---|---|
| `config.py` | CLI parameters and defaults |
| `train_main.py` | Training entry point and provenance |
| `run_contention_600k.sh` / `.ps1` | Current main MLP-vs-TF experiment on Linux/Windows |
| `resume_contention_600k.sh` | Resume after an interrupted run |
| `run_iicms_comparison.sh` / `.ps1` | Configurable end-to-end workflow |
| `algorithms/algorithm/r_mappo.py` | GAE and PPO update |
| `algorithms/algorithm/rMAPPOPolicy.py` | Actor/critic wrapper |
| `algorithms/algorithm/r_actor_critic.py` | Network assembly |
| `algorithms/utils/act.py` | Joint-action distributions and masks |
| `algorithms/utils/mlp.py` | MLP backbone |
| `algorithms/utils/transformer.py` | Transformer backbone |
| `envs/env_iicms_mobile.py` | Main mobility and radio environment |
| `envs/env_core.py` | Channel and rate model |
| `runner/shared/` | Shared-policy MAPPO loop |
| `runner/separated/` | Independent-policy MAPPO loop |
| `utils/policy_evaluation.py` | Validation evaluator |
| `utils/shared_buffer.py` | Shared-policy replay buffer |
| `utils/separated_buffer.py` | Independent-policy replay buffer |
| `evaluate_iicms.py` | Policy and baseline evaluation |
| `select_best_checkpoint.py` | Validation checkpoint selection |
| `tune_rssi_thresholds.py` | RSSI threshold calibration |
| `run_baseline_suite.sh` | Heuristic baseline suite |
| `plot_*.py` | Training, reward, and CDF figures |
| `package_project.py` | Source manifest and local archives |
| `package_results.py` | Result bundles |
| `tests/` | Unit and protocol tests |

## Modeling Invariants

- 30 APs, 30 UEs, 4 antennas per AP, 20 MHz bandwidth, 1.9 GHz carrier.
- APs are placed in 1 km x 1 km with at least 50 m spacing.
- RWP mobility uses 15 m/s and a 0.5 s decision interval.
- Shadowing is enabled; CSI aging is disabled in the main experiment.
- Backhaul is 550-650 Mbps per AP with bottleneck JT accounting.
- The finite fronthaul queue is 0.5 s at the AP service rate.
- The action space has 53 valid joint actions over JT/CB/NC and L=2..5.
- Main observations contain four frames of 1 self UE, 3 competing neighbors,
  and 5 candidate AP tokens.
- Each token is 9-dimensional, so actor input is 324 and centralized critic
  input is 30 x 324 = 9720.
- The main reward is `team_rate` in Mbps.
- QoS weight is zero in pure-rate training; P05 and QoS are diagnostics.

## Evaluation Invariants

- Validation seeds: `91000-91019`.
- Hyperparameter-validation seeds: `92000-92019`.
- Final test seeds: `93000-93019`.
- Checkpoint selection, tuning, and final testing must use disjoint seeds.
- Training validation runs 20 deterministic episodes every 25k steps.
- Checkpoint selection uses 20 deterministic episodes.
- Pure-rate checkpoint selection uses `system_se`, not a QoS-first objective.
- Deterministic execution is the primary policy result; stochastic execution is
  reported separately.
- Never compare runs from different seed panels or environment settings as the
  same experiment.

## Artifact Rules

- `results/`, `analysis_outputs/`, figures, logs, archives, and model weights
  are generated artifacts and stay out of Git.
- Raw reward plotting requires `--source raw`; `curve.csv` is not a substitute.
- Preserve `config.json`, `source_sha256.json`, and protocol identifiers with
  experiment outputs.
- Plot scripts must retain the exact metric and execution-mode label.
- Do not mix deterministic and stochastic results silently.

## Required Checks

From the repository root:

```bash
python -m unittest discover -s tests -v
```

Before a training-related change, also run a short smoke experiment and inspect
`logs/curve.csv`, `logs/validation_curve.csv`,
`checkpoint_selection.json`, and `evaluation_deterministic/run1.json`.

## Contribution Workflow

1. Branch from current `main`.
2. Make one focused change.
3. Run the full test suite.
4. Commit with a clear message.
5. Push the branch and open a Pull Request.
6. Merge after review and passing tests.

Do not combine unrelated source changes with generated artifacts.

## Permission Model

The repository is public: anyone may view, clone, fork, and open a Pull
Request. Direct writes require ownership or collaborator permission. The
repository currently has no declared license; licensing is a separate owner
decision.
