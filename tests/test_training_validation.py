from __future__ import annotations

import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import get_config  # noqa: E402
from select_best_checkpoint import resolve_selection_metric  # noqa: E402
from utils.policy_evaluation import (  # noqa: E402
    ValidationRecorder,
    evaluate_episodes,
    validation_plateau_report,
)


class _FakeInnerEnv:
    bandwidth_hz = 20e6
    joint_actions = [0, 1]


class _FakeEnv:
    def __init__(self):
        self.num_agent = 2
        self.env = _FakeInnerEnv()
        self.seed = None

    def configure(self, args):
        self.seed = int(args.seed)

    def reset(self):
        return np.zeros((2, 1), dtype=np.float32)

    def step(self, actions):
        del actions
        rates = [1e7, 2e7] if self.seed == 91000 else [2e7, 4e7]
        infos = [
            {
                "inst_rate": rate,
                "qos_violation": 0.0,
                "qos_shortfall_mbps": 0.0,
                "backhaul_scale": 1.0,
                "fronthaul_drop_ratio": 0.0,
                "fronthaul_queue_ratio": 0.0,
                "cluster_changed": 1.0,
                "mode": index,
                "cluster_k": 2,
            }
            for index, rate in enumerate(rates)
        ]
        return (
            np.zeros((2, 1), dtype=np.float32),
            np.zeros((2, 1), dtype=np.float32),
            np.zeros(2, dtype=bool),
            infos,
        )


class _FakeActor:
    def reset(self):
        return None

    def act(self, obs):
        return np.zeros((len(obs), 1), dtype=int)


def _history(values):
    return [
        {
            "environment_steps": 25000 * (index + 1),
            "episode_sum_rate_mbps": [
                value + offset for offset in (-1.0, 0.0, 1.0)
            ],
        }
        for index, value in enumerate(values)
    ]


class TrainingValidationTest(unittest.TestCase):
    def test_shared_evaluator_reconfigures_each_validation_seed(self):
        metrics = evaluate_episodes(
            _FakeEnv(),
            Namespace(seed=1),
            [91000, 91001],
            horizon=2,
            actor=_FakeActor(),
        )
        self.assertEqual(metrics["episodes"], 2)
        self.assertEqual(metrics["seeds"], [91000, 91001])
        self.assertAlmostEqual(metrics["sum_rate_mbps"], 45.0)
        np.testing.assert_allclose(
            metrics["episode_sum_rate_mbps"], [30.0, 60.0]
        )
        self.assertGreater(metrics["p05_rate_mbps"], 0.0)

    def test_plateau_requires_a_confident_flat_recent_slope(self):
        flat = validation_plateau_report(
            _history([3000.0, 3001.0, 3000.0]),
            relative_gain_tolerance=0.005,
            window_points=3,
        )
        rising = validation_plateau_report(
            _history([2800.0, 2950.0, 3100.0]),
            relative_gain_tolerance=0.005,
            window_points=3,
        )
        self.assertTrue(flat["plateau_detected"])
        self.assertEqual(flat["recommendation"], "plateau_candidate")
        self.assertFalse(rising["plateau_detected"])
        self.assertEqual(rising["recommendation"], "continue_training")

    def test_validation_recorder_writes_curve_and_convergence_files(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = ValidationRecorder(
                Path(directory), plateau_tolerance=0.005, window=3
            )
            for index in range(3):
                recorder.append(
                    25000 * (index + 1),
                    {
                        "episodes": 2,
                        "horizon": 100,
                        "system_se_bps_hz": 150.0,
                        "sum_rate_mbps": 3000.0 + index,
                        "p05_rate_mbps": 50.0,
                        "qos_violation_rate": 0.01,
                        "episode_sum_rate_mbps": [2999.0, 3000.0],
                    },
                )
            logs = Path(directory) / "logs"
            self.assertTrue((logs / "validation_curve.csv").exists())
            self.assertTrue((logs / "validation_history.json").exists())
            self.assertTrue((logs / "validation_convergence.json").exists())

    def test_auto_checkpoint_metric_matches_training_objective(self):
        self.assertEqual(
            resolve_selection_metric("auto", 0.0), "system_se"
        )
        self.assertEqual(
            resolve_selection_metric("auto", 20.0),
            "qos_constrained",
        )
        self.assertEqual(
            resolve_selection_metric("system_se", 20.0),
            "system_se",
        )

    def test_validation_defaults_are_twenty_fixed_seed_episodes(self):
        args = get_config().parse_args([])
        self.assertTrue(args.use_eval)
        self.assertEqual(args.validation_interval_steps, 25000)
        self.assertEqual(args.validation_episodes, 20)
        self.assertEqual(args.validation_seed_start, 91000)


if __name__ == "__main__":
    unittest.main()
