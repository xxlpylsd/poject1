# -*- coding: utf-8 -*-
"""Shared deterministic policy evaluation and validation-curve utilities."""
from __future__ import annotations

import csv
import json
from argparse import Namespace
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import torch


def encode_action(env, action: np.ndarray) -> np.ndarray:
    """Encode a compact joint-action index for the IICMS environment."""
    action_index = int(np.asarray(action).reshape(-1)[0])
    encoded = np.zeros(len(env.env.joint_actions), dtype=np.float32)
    encoded[action_index] = 1.0
    return encoded


class SharedPolicyActor:
    """Adapt one shared recurrent/non-recurrent policy for episode rollout."""

    def __init__(self, policy, device: torch.device, num_agents: int):
        self.policy = policy
        self.device = device
        self.num_agents = int(num_agents)
        self.rnn = None
        self.reset()

    def reset(self) -> None:
        self.rnn = np.zeros(
            (self.num_agents, 1, 1), dtype=np.float32
        )

    @torch.no_grad()
    def act(self, obs: np.ndarray) -> np.ndarray:
        masks = np.ones((self.num_agents, 1), dtype=np.float32)
        action, self.rnn = self.policy.act(
            torch.as_tensor(obs, device=self.device),
            torch.as_tensor(self.rnn, device=self.device),
            torch.as_tensor(masks, device=self.device),
            deterministic=True,
        )
        return action.detach().cpu().numpy().astype(int)


class SeparatePolicyActor:
    """Adapt independent per-UE policies for episode rollout."""

    def __init__(
        self,
        policies: Sequence,
        device: torch.device,
        num_agents: int,
    ):
        self.policies = list(policies)
        self.device = device
        self.num_agents = int(num_agents)
        self.rnn = None
        self.reset()

    def reset(self) -> None:
        self.rnn = [
            np.zeros((1, 1, 1), dtype=np.float32)
            for _ in range(self.num_agents)
        ]

    @torch.no_grad()
    def act(self, obs: np.ndarray) -> np.ndarray:
        rows = []
        for agent_id, policy in enumerate(self.policies):
            action, rnn = policy.act(
                torch.as_tensor(
                    obs[agent_id:agent_id + 1],
                    device=self.device,
                ),
                torch.as_tensor(
                    self.rnn[agent_id],
                    device=self.device,
                ),
                torch.ones(
                    (1, 1), dtype=torch.float32, device=self.device
                ),
                deterministic=True,
            )
            self.rnn[agent_id] = rnn.detach().cpu().numpy()
            rows.append(action.detach().cpu().numpy())
        return np.concatenate(rows, axis=0).astype(int)


def _mean(values: Sequence[float]) -> float:
    return float(np.mean(values)) if len(values) else 0.0


def _std(values: Sequence[float]) -> float:
    if len(values) <= 1:
        return 0.0
    return float(np.std(values, ddof=1))


@torch.no_grad()
def evaluate_episodes(
    env,
    all_args: Namespace,
    seeds: Sequence[int],
    horizon: int,
    actor: Callable[[np.ndarray], np.ndarray] | object,
) -> dict:
    """Evaluate deterministic episodes on a fixed seed panel.

    ``actor`` may be any object with ``reset()`` and ``act(obs)`` methods.
    The function owns environment scenario setup so training validation,
    checkpoint selection and later evaluation share identical semantics.
    """
    if not seeds:
        raise ValueError("At least one evaluation seed is required")
    if int(horizon) <= 0:
        raise ValueError("Evaluation horizon must be positive")

    step_system_se = []
    per_ue_rates = []
    qos_violation = []
    qos_shortfall = []
    overload = []
    backhaul_scale = []
    fronthaul_drop = []
    fronthaul_queue = []
    cluster_change = []
    mode_counts = np.zeros(3, dtype=float)
    size_values = []
    episode_sum_rates = []
    episode_system_se = []
    episode_p05_rates = []
    episode_qos_violation = []
    total_decisions = 0

    for seed in seeds:
        episode_args = Namespace(**vars(all_args))
        episode_args.seed = int(seed)
        env.configure(episode_args)
        obs = env.reset()
        actor.reset()
        current_rates = []
        current_sum_rates = []
        current_qos = []
        current_system_se = []

        for _ in range(int(horizon)):
            action = actor.act(obs)
            env_actions = [
                encode_action(env, row) for row in action
            ]
            obs, _rewards, _dones, infos = env.step(env_actions)
            rates = np.asarray(
                [item["inst_rate"] for item in infos],
                dtype=float,
            ) / 1e6
            current_rates.extend(rates.tolist())
            per_ue_rates.extend(rates.tolist())

            sum_rate = float(rates.sum())
            current_sum_rates.append(sum_rate)
            system_se = (
                sum_rate * 1e6
                / float(env.env.bandwidth_hz)
            )
            current_system_se.append(system_se)
            step_system_se.append(system_se)

            qos_values = [
                float(item.get("qos_violation", 0.0))
                for item in infos
            ]
            current_qos.append(float(np.mean(qos_values)))
            qos_violation.append(float(np.mean(qos_values)))
            qos_shortfall.append(float(np.mean([
                item.get("qos_shortfall_mbps", 0.0)
                for item in infos
            ])))
            overload.append(float(np.mean([
                item.get("backhaul_scale", 1.0) < 0.999999
                for item in infos
            ])))
            backhaul_scale.append(float(np.mean([
                item.get("backhaul_scale", 1.0)
                for item in infos
            ])))
            fronthaul_drop.append(float(np.mean([
                item.get("fronthaul_drop_ratio", 0.0)
                for item in infos
            ])))
            fronthaul_queue.append(float(np.mean([
                item.get("fronthaul_queue_ratio", 0.0)
                for item in infos
            ])))
            cluster_change.append(float(np.mean([
                item.get("cluster_changed", 0.0)
                for item in infos
            ])))
            for item in infos:
                mode = int(item.get("mode", 2))
                if 0 <= mode < 3:
                    mode_counts[mode] += 1.0
                if mode != 2 and "cluster_k" in item:
                    size_values.append(float(item["cluster_k"]))
            total_decisions += len(infos)

        episode_sum_rates.append(float(np.mean(current_sum_rates)))
        episode_system_se.append(float(np.mean(current_system_se)))
        episode_p05_rates.append(float(np.percentile(current_rates, 5)))
        episode_qos_violation.append(float(np.mean(current_qos)))

    system_se = float(np.mean(step_system_se))
    mean_ue_se = system_se / max(1, int(env.num_agent))
    mode_share = mode_counts / max(1.0, total_decisions)
    return {
        "episodes": int(len(seeds)),
        "horizon": int(horizon),
        "seeds": [int(seed) for seed in seeds],
        "system_se_bps_hz": system_se,
        "mean_ue_se_bps_hz": mean_ue_se,
        "sum_rate_mbps": float(np.mean(episode_sum_rates)),
        "sum_rate_std_mbps": _std(episode_sum_rates),
        "system_se_std_bps_hz": _std(episode_system_se),
        "p05_rate_mbps": float(np.percentile(per_ue_rates, 5)),
        "p50_rate_mbps": float(np.percentile(per_ue_rates, 50)),
        "qos_violation_rate": _mean(qos_violation),
        "qos_shortfall_mbps": _mean(qos_shortfall),
        "overload_ratio": _mean(overload),
        "backhaul_scale": _mean(backhaul_scale),
        "fronthaul_drop_ratio": _mean(fronthaul_drop),
        "fronthaul_queue_ratio": _mean(fronthaul_queue),
        "cluster_change_rate": _mean(cluster_change),
        "cluster_size_mean": _mean(size_values),
        "mode_jt": float(mode_share[0]),
        "mode_cb": float(mode_share[1]),
        "mode_nc": float(mode_share[2]),
        "episode_sum_rate_mbps": episode_sum_rates,
        "episode_system_se_bps_hz": episode_system_se,
        "episode_p05_rate_mbps": episode_p05_rates,
        "episode_qos_violation_rate": episode_qos_violation,
    }


VALIDATION_SCALAR_FIELDS = (
    "validation_index",
    "environment_steps",
    "episodes",
    "horizon",
    "system_se_bps_hz",
    "sum_rate_mbps",
    "sum_rate_std_mbps",
    "system_se_std_bps_hz",
    "p05_rate_mbps",
    "p50_rate_mbps",
    "qos_violation_rate",
    "qos_shortfall_mbps",
    "overload_ratio",
    "backhaul_scale",
    "fronthaul_drop_ratio",
    "fronthaul_queue_ratio",
    "cluster_change_rate",
    "cluster_size_mean",
    "mode_jt",
    "mode_cb",
    "mode_nc",
)


def validation_plateau_report(
    history: Sequence[dict],
    relative_gain_tolerance: float = 0.005,
    window_points: int = 3,
) -> dict:
    """Estimate recent validation slope with fixed scenario effects.

    The slope is expressed as relative system-rate gain per 100k
    environment steps. A plateau is only reported when the upper end of
    the slope confidence interval is within the requested tolerance.
    """
    if len(history) < 2:
        return {
            "status": "insufficient_validation_points",
            "plateau_detected": False,
            "recommendation": "continue_training",
            "validation_points": len(history),
        }

    points = list(history[-max(2, int(window_points)):])
    episode_key = "episode_sum_rate_mbps"
    if not points or episode_key not in points[-1]:
        return {
            "status": "missing_episode_history",
            "plateau_detected": False,
            "recommendation": "continue_training",
            "validation_points": len(points),
        }

    widths = [len(point[episode_key]) for point in points]
    if min(widths) < 2 or len(set(widths)) != 1:
        return {
            "status": "insufficient_episode_samples",
            "plateau_detected": False,
            "recommendation": "continue_training",
            "validation_points": len(points),
        }

    steps = np.asarray(
        [float(point["environment_steps"]) for point in points],
        dtype=float,
    )
    values = np.asarray(
        [point[episode_key] for point in points],
        dtype=float,
    )
    x = steps / 100000.0
    x_centered = x - np.mean(x)
    y_centered = values - np.mean(values, axis=0, keepdims=True)
    denominator = (
        float(np.sum(x_centered ** 2)) * values.shape[1]
    )
    if denominator <= 0.0:
        return {
            "status": "zero_step_variance",
            "plateau_detected": False,
            "recommendation": "continue_training",
            "validation_points": len(points),
        }

    slope = float(
        np.sum(x_centered[:, None] * y_centered) / denominator
    )
    residuals = y_centered - x_centered[:, None] * slope
    degrees_freedom = (
        (values.shape[0] - 1) * (values.shape[1] - 1)
    )
    if degrees_freedom > 0:
        residual_variance = (
            float(np.sum(residuals ** 2)) / degrees_freedom
        )
        slope_se = float(
            np.sqrt(residual_variance / denominator)
        )
    else:
        slope_se = float("inf")

    current_mean = float(np.mean(values[-1]))
    scale = max(abs(current_mean), 1e-12)
    relative_slope = slope / scale
    confidence_low = slope - 1.96 * slope_se
    confidence_high = slope + 1.96 * slope_se
    plateau = (
        len(points) >= max(3, int(window_points))
        and (confidence_high / scale) <= relative_gain_tolerance
    )
    return {
        "status": (
            "plateau_candidate" if plateau
            else "still_improving_or_uncertain"
        ),
        "plateau_detected": bool(plateau),
        "recommendation": (
            "plateau_candidate" if plateau
            else "continue_training"
        ),
        "validation_points": len(points),
        "first_step": int(steps[0]),
        "last_step": int(steps[-1]),
        "current_mean_sum_rate_mbps": current_mean,
        "slope_mbps_per_100k": slope,
        "slope_relative_per_100k": relative_slope,
        "slope_ci95_mbps_per_100k": [
            confidence_low,
            confidence_high,
        ],
        "relative_gain_tolerance_per_100k": float(
            relative_gain_tolerance
        ),
    }


class ValidationRecorder:
    """Write validation data in CSV and JSON and update convergence state."""

    def __init__(self, run_dir: Path, plateau_tolerance: float, window: int):
        self.log_dir = Path(run_dir) / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.log_dir / "validation_curve.csv"
        self.history_path = self.log_dir / "validation_history.json"
        self.report_path = self.log_dir / "validation_convergence.json"
        self.plateau_tolerance = float(plateau_tolerance)
        self.window = int(window)
        self.history = []
        if self.history_path.exists():
            try:
                self.history = json.loads(
                    self.history_path.read_text(encoding="utf-8")
                )
            except (OSError, ValueError, TypeError):
                self.history = []

    def append(
        self,
        environment_steps: int,
        metrics: dict,
    ) -> tuple[dict, dict]:
        row = {
            "validation_index": len(self.history) + 1,
            "environment_steps": int(environment_steps),
            **metrics,
        }
        self.history.append(row)
        self.history_path.write_text(
            json.dumps(self.history, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        scalar_row = {
            key: row.get(key)
            for key in VALIDATION_SCALAR_FIELDS
        }
        write_header = not self.csv_path.exists()
        with self.csv_path.open(
            "a", newline="", encoding="utf-8-sig"
        ) as stream:
            writer = csv.DictWriter(
                stream, fieldnames=list(VALIDATION_SCALAR_FIELDS)
            )
            if write_header:
                writer.writeheader()
            writer.writerow(scalar_row)

        report = validation_plateau_report(
            self.history,
            self.plateau_tolerance,
            self.window,
        )
        self.report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return row, report
