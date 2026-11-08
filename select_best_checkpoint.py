# -*- coding: utf-8 -*-
"""Select a trained checkpoint using deterministic validation trajectories."""
from __future__ import annotations

import argparse
import json
import re
import shutil
from argparse import Namespace
from pathlib import Path

from utils.runtime_env import normalize_thread_environment

normalize_thread_environment()

import numpy as np
import torch

from algorithms.algorithm.rMAPPOPolicy import RMAPPOPolicy
from envs.env_iicms_mobile import IICMSMobileDiscreteActionEnv


def encode_action(env, action: np.ndarray) -> np.ndarray:
    action_index = int(np.asarray(action).reshape(-1)[0])
    encoded = np.zeros(len(env.env.joint_actions), dtype=np.float32)
    encoded[action_index] = 1.0
    return encoded


def snapshot_steps(models_dir: Path) -> list[int]:
    steps = []
    for path in models_dir.glob("actor_step_*.pt"):
        match = re.fullmatch(r"actor_step_(\d+)\.pt", path.name)
        if match:
            steps.append(int(match.group(1)))
    return sorted(set(steps))


def load_config(run_dir: Path) -> Namespace:
    config = json.loads((run_dir / "config.json").read_text(
        encoding="utf-8"
    ))
    config.setdefault("cluster_sizes", "2,3,4,5")
    config.setdefault("obs_include_xy", 0)
    config.setdefault("obs_frames", 1)
    return Namespace(**config)


def make_policy(all_args: Namespace, env, device: torch.device):
    all_args.actor_token_dim = env.obs_token_dim
    all_args.critic_token_dim = env.signal_obs_dim
    all_args.joint_num_modes = env.joint_num_modes
    all_args.joint_num_sizes = env.joint_num_sizes
    all_args.joint_num_actions = env.joint_num_actions
    all_args.joint_valid_action_indices = list(env.joint_action_indices)
    all_args.joint_num_cluster_choices = (
        env.env.joint_num_cluster_choices
    )
    all_args.joint_cluster_choice_ranks = [
        list(ranks) for ranks in env.env.cluster_choice_ranks
    ]
    all_args.joint_cluster_choice_sizes = list(
        env.env.cluster_choice_sizes
    )
    all_args.joint_action_cluster_choice = list(
        env.env.joint_action_cluster_choice
    )
    all_args.joint_num_ap_tokens = env.obs_tokens_per_frame - 1
    return RMAPPOPolicy(
        all_args,
        env.observation_space[0],
        env.share_observation_space[0],
        env.action_space[0],
        device=device,
    )


@torch.no_grad()
def evaluate_candidate(
    policy,
    env,
    all_args: Namespace,
    actor_path: Path,
    critic_path: Path,
    seeds: list[int],
    horizon: int,
    qos_weight: float,
) -> dict:
    policy.actor.load_state_dict(
        torch.load(actor_path, map_location=policy.device)
    )
    if critic_path.exists():
        policy.critic.load_state_dict(
            torch.load(critic_path, map_location=policy.device)
        )
    policy.actor.eval()
    policy.critic.eval()

    step_system_se = []
    per_ue_rates = []
    qos_violation = []
    qos_shortfall = []
    overload = []
    fronthaul_drop = []
    fronthaul_queue = []
    for seed in seeds:
        episode_args = Namespace(**vars(all_args))
        episode_args.seed = seed
        env.configure(episode_args)
        obs = env.reset()
        rnn = np.zeros((env.num_agent, 1, 1), dtype=np.float32)
        masks = np.ones((env.num_agent, 1), dtype=np.float32)
        for _ in range(horizon):
            action, rnn = policy.act(
                torch.as_tensor(
                    obs, device=policy.device
                ),
                torch.as_tensor(
                    rnn, device=policy.device
                ),
                torch.as_tensor(
                    masks, device=policy.device
                ),
                deterministic=True,
            )
            action = action.detach().cpu().numpy().astype(int)
            env_actions = [
                encode_action(env, row) for row in action
            ]
            obs, _, _, infos = env.step(env_actions)
            rates = np.asarray([
                item["inst_rate"] for item in infos
            ]) / 1e6
            per_ue_rates.extend(rates.tolist())
            step_system_se.append(
                float(rates.sum()) * 1e6 / env.env.bandwidth_hz
            )
            qos_violation.append(float(np.mean([
                item["qos_violation"] for item in infos
            ])))
            qos_shortfall.append(float(np.mean([
                item["qos_shortfall_mbps"] for item in infos
            ])))
            overload.append(float(np.mean([
                item["backhaul_scale"] < 0.999999
                for item in infos
            ])))
            fronthaul_drop.append(float(np.mean([
                item["fronthaul_drop_ratio"] for item in infos
            ])))
            fronthaul_queue.append(float(np.mean([
                item["fronthaul_queue_ratio"] for item in infos
            ])))

    system_se = float(np.mean(step_system_se))
    mean_ue_se = system_se / env.num_agent
    violation_rate = float(np.mean(qos_violation))
    shortfall_mbps = float(np.mean(qos_shortfall))
    requirement = max(
        float(env.env.rate_req_lo_mbps), 1e-12
    )
    normalized_shortfall = shortfall_mbps / requirement
    qos_aware_score = (
        mean_ue_se
        - qos_weight
        * (violation_rate + normalized_shortfall)
    )
    return {
        "system_se_bps_hz": system_se,
        "mean_ue_se_bps_hz": mean_ue_se,
        "p05_mbps": float(np.percentile(per_ue_rates, 5)),
        "p50_mbps": float(np.percentile(per_ue_rates, 50)),
        "qos_violation_rate": violation_rate,
        "qos_shortfall_mbps": shortfall_mbps,
        "overload_ratio": float(np.mean(overload)),
        "fronthaul_drop_ratio": float(np.mean(fronthaul_drop)),
        "fronthaul_queue_ratio": float(np.mean(fronthaul_queue)),
        "qos_aware_score": float(qos_aware_score),
    }


def select_candidate(
    candidates: list[dict],
    metric: str,
    max_qos_violation_rate: float,
) -> tuple[dict, str]:
    """Select a checkpoint according to the configured comparison rule."""
    if not candidates:
        raise ValueError("No checkpoint candidates were provided")
    if metric == "system_se":
        best = max(
            candidates,
            key=lambda item: (
                item["system_se_bps_hz"],
                -item["qos_violation_rate"],
            ),
        )
        return best, "system_se"
    if metric == "qos_aware":
        best = max(
            candidates,
            key=lambda item: (
                item["qos_aware_score"],
                item["system_se_bps_hz"],
                -item["qos_violation_rate"],
            ),
        )
        return best, "qos_aware"
    if metric != "qos_constrained":
        raise ValueError(f"Unknown selection metric: {metric}")

    feasible = [
        item
        for item in candidates
        if item["qos_violation_rate"]
        <= max_qos_violation_rate + 1e-12
    ]
    if feasible:
        return max(
            feasible,
            key=lambda item: (
                item["system_se_bps_hz"],
                -item["qos_shortfall_mbps"],
            ),
        ), "qos_constrained"

    minimum_violation = min(
        item["qos_violation_rate"] for item in candidates
    )
    least_infeasible = [
        item
        for item in candidates
        if item["qos_violation_rate"]
        <= minimum_violation + 1e-12
    ]
    return max(
        least_infeasible,
        key=lambda item: item["system_se_bps_hz"],
    ), "qos_constrained_fallback"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--seed_start", type=int, default=91000)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--horizon", type=int, default=100)
    parser.add_argument(
        "--metric",
        choices=("qos_constrained", "qos_aware", "system_se"),
        default="qos_constrained",
    )
    parser.add_argument(
        "--max_qos_violation_rate",
        type=float,
        default=0.005,
        help=(
            "maximum validation QoS violation rate for "
            "qos_constrained checkpoint selection"
        ),
    )
    parser.add_argument("--qos_penalty_weight", type=float, default=20.0)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    models_dir = run_dir / "models"
    all_args = load_config(run_dir)
    env = IICMSMobileDiscreteActionEnv()
    env.configure(all_args)
    device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu"
    )
    policy = make_policy(all_args, env, device)
    seeds = [
        args.seed_start + index
        for index in range(args.episodes)
    ]

    steps = snapshot_steps(models_dir)
    candidates = []
    if steps:
        for step in steps:
            actor_path = models_dir / f"actor_step_{step:08d}.pt"
            critic_path = models_dir / f"critic_step_{step:08d}.pt"
            metrics = evaluate_candidate(
                policy, env, all_args, actor_path, critic_path,
                seeds, args.horizon, args.qos_penalty_weight,
            )
            candidates.append({
                "step": step,
                "actor": actor_path.name,
                "critic": critic_path.name,
                **metrics,
            })
    else:
        candidates.append({
            "step": None,
            "actor": "actor.pt",
            "critic": "critic.pt",
            **evaluate_candidate(
                policy,
                env,
                all_args,
                models_dir / "actor.pt",
                models_dir / "critic.pt",
                seeds,
                args.horizon,
                args.qos_penalty_weight,
            ),
        })

    # Preserve the final model, then install the validation-selected pair.
    final_actor = models_dir / "actor_final.pt"
    final_critic = models_dir / "critic_final.pt"
    if not final_actor.exists():
        shutil.copyfile(models_dir / "actor.pt", final_actor)
    if not final_critic.exists():
        shutil.copyfile(models_dir / "critic.pt", final_critic)

    if not 0.0 <= args.max_qos_violation_rate <= 1.0:
        parser.error(
            "--max_qos_violation_rate must be in [0, 1]"
        )
    best, selection_status = select_candidate(
        candidates,
        args.metric,
        args.max_qos_violation_rate,
    )
    best_actor = models_dir / best["actor"]
    best_critic = models_dir / best["critic"]
    shutil.copyfile(best_actor, models_dir / "actor_best.pt")
    shutil.copyfile(best_critic, models_dir / "critic_best.pt")
    shutil.copyfile(models_dir / "actor_best.pt", models_dir / "actor.pt")
    shutil.copyfile(models_dir / "critic_best.pt", models_dir / "critic.pt")

    result = {
        "run_dir": str(run_dir),
        "selection_seed_start": args.seed_start,
        "selection_episodes": args.episodes,
        "selection_horizon": args.horizon,
        "selection_metric": args.metric,
        "max_qos_violation_rate": (
            args.max_qos_violation_rate
        ),
        "selection_status": selection_status,
        "scenario_seed_policy": "selection_episode_seed",
        "selected_step": best["step"],
        "selected": best,
        "candidates": candidates,
    }
    (run_dir / "checkpoint_selection.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
