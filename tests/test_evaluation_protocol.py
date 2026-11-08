from __future__ import annotations

import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluate_iicms import (  # noqa: E402
    canonical_env_config,
    checkpoint_path,
    empirical_cdf,
    initial_state_digest,
    protocol_id,
    ranges_overlap,
    resolve_fixed_action,
    seed_range,
)
from envs.env_iicms_mobile import IICMSMobileDiscreteActionEnv  # noqa: E402
from select_best_checkpoint import select_candidate  # noqa: E402


def evaluation_args(seed: int) -> Namespace:
    return Namespace(
        num_aps=12,
        num_agents=6,
        map_size=1000.0,
        ap_min_distance=50.0,
        antennas_per_ap=4,
        channel_file=None,
        seed=seed,
        backhaul_lo=4e8,
        backhaul_hi=5e8,
        cluster_sizes="2,3,4,5",
        mobility="rwp",
        ue_speed=10.0,
        slot_dt=0.5,
        pathloss_model="legacy",
        csi_aging=0,
        reward_mode="mean_se",
        objective_horizon=100,
        rate_req_lo_mbps=40.0,
        rate_req_hi_mbps=40.0,
        min_ue_se_bps_hz=0.5,
        qos_penalty_weight=20.0,
        backhaul_jt_model="bottleneck",
        fronthaul_queue=1,
        fronthaul_buffer_s=0.5,
        reset_ues=1,
        cluster_action_mode="subset",
        obs_include_xy=1,
        obs_frames=4,
    )


class EvaluationProtocolTest(unittest.TestCase):
    def test_empirical_cdf_is_monotone_and_reaches_one(self):
        grid, cdf = empirical_cdf(
            np.asarray([0.0, 1.0, 1.0, 2.0, 4.0]),
            points=5,
        )
        np.testing.assert_allclose(grid, [0.0, 1.0, 2.0, 3.0, 4.0])
        np.testing.assert_allclose(cdf, [0.2, 0.6, 0.8, 0.8, 1.0])
        self.assertTrue(np.all(np.diff(cdf) >= 0.0))

    def test_qos_constrained_checkpoint_maximizes_se_after_qos_filter(self):
        candidates = [
            {
                "step": 1,
                "system_se_bps_hz": 170.0,
                "qos_violation_rate": 0.003,
                "qos_shortfall_mbps": 0.02,
                "qos_aware_score": 1.0,
            },
            {
                "step": 2,
                "system_se_bps_hz": 172.0,
                "qos_violation_rate": 0.004,
                "qos_shortfall_mbps": 0.03,
                "qos_aware_score": 2.0,
            },
            {
                "step": 3,
                "system_se_bps_hz": 180.0,
                "qos_violation_rate": 0.008,
                "qos_shortfall_mbps": 0.05,
                "qos_aware_score": 3.0,
            },
        ]
        best, status = select_candidate(
            candidates,
            "qos_constrained",
            0.005,
        )
        self.assertEqual(best["step"], 2)
        self.assertEqual(status, "qos_constrained")

        best, status = select_candidate(
            candidates[:2] + [{
                **candidates[2],
                "qos_violation_rate": 0.009,
            }],
            "qos_constrained",
            0.001,
        )
        self.assertEqual(best["step"], 1)
        self.assertEqual(status, "qos_constrained_fallback")

    def test_seed_ranges_are_inclusive_and_disjointness_is_explicit(self):
        self.assertEqual(seed_range(93000, 20), (93000, 93019))
        self.assertTrue(ranges_overlap((93000, 93019), (93019, 93025)))
        self.assertFalse(ranges_overlap((93000, 93019), (93020, 93039)))
        self.assertFalse(ranges_overlap((93000, 93019), (91000, 91004)))

    def test_fixed_cb_l5_alias_resolves_rank_zero_subset(self):
        env = SimpleNamespace(
            joint_action_names=[
                "JT_L2_S00",
                "JT_L5_S00",
                "CB_L2_S00",
                "CB_L5_S00",
                "NC",
            ]
        )
        name, index = resolve_fixed_action(env, "CB-L5")
        self.assertEqual(name, "CB_L5_S00")
        self.assertEqual(index, 3)
        self.assertEqual(
            resolve_fixed_action(env, "NC"),
            ("NC", 4),
        )

    def test_protocol_id_ignores_algorithm_but_tracks_trajectory(self):
        common = {
            "num_agents": 30,
            "num_aps": 30,
            "map_size": 1000,
            "ue_speed": 10,
            "actor_backbone": "mlp",
        }
        tf = {**common, "actor_backbone": "transformer"}
        self.assertEqual(
            canonical_env_config(common),
            canonical_env_config(tf),
        )
        digests = ["a" * 64, "b" * 64]
        base_id = protocol_id(
            common, [93000, 93001], 100, "test", digests
        )
        self.assertEqual(
            base_id,
            protocol_id(
                tf, [93000, 93001], 100, "test", digests
            ),
        )
        self.assertNotEqual(
            base_id,
            protocol_id(
                common, [93000, 93002], 100, "test", digests
            ),
        )
        self.assertNotEqual(
            base_id,
            protocol_id(
                common, [93000, 93001], 200, "test", digests
            ),
        )
        self.assertNotEqual(
            base_id,
            protocol_id(
                common,
                [93000, 93001],
                100,
                "test",
                ["c" * 64, "b" * 64],
            ),
        )
        queued = {
            **common,
            "fronthaul_queue": 1,
            "fronthaul_buffer_s": 0.5,
        }
        changed_buffer = {
            **queued,
            "fronthaul_buffer_s": 1.0,
        }
        queued_id = protocol_id(
            queued, [93000, 93001], 100, "test", digests
        )
        self.assertNotEqual(
            queued_id,
            protocol_id(
                changed_buffer,
                [93000, 93001],
                100,
                "test",
                digests,
            ),
        )

    def test_checkpoint_role_requires_the_requested_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            models = Path(directory) / "models"
            models.mkdir()
            (models / "actor_best.pt").write_bytes(b"best")
            self.assertEqual(
                checkpoint_path(Path(directory), "best"),
                models / "actor_best.pt",
            )
            with self.assertRaises(FileNotFoundError):
                checkpoint_path(Path(directory), "final")

    def test_eval_seed_rebuilds_the_same_scenario_for_any_training_seed(self):
        digests = []
        for training_seed in (1, 2):
            env = IICMSMobileDiscreteActionEnv()
            env.configure(evaluation_args(training_seed))
            env.configure(evaluation_args(93000))
            env.reset()
            digests.append(initial_state_digest(env))
        self.assertEqual(digests[0], digests[1])

        env_a = IICMSMobileDiscreteActionEnv()
        env_b = IICMSMobileDiscreteActionEnv()
        env_a.configure(evaluation_args(1))
        env_a.configure(evaluation_args(93000))
        env_a.reset()
        env_b.configure(evaluation_args(2))
        env_b.configure(evaluation_args(93000))
        env_b.reset()
        np.testing.assert_array_equal(
            env_a.env.users_xy,
            env_b.env.users_xy,
        )
        np.testing.assert_array_equal(
            env_a.env.aps_xy,
            env_b.env.aps_xy,
        )


if __name__ == "__main__":
    unittest.main()
