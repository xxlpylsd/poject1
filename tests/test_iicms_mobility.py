from __future__ import annotations
import sys
import unittest
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from envs.env_iicms_mobile import IICMSMobileEnvCore, encode_iicms_action

def make_env(**overrides) -> IICMSMobileEnvCore:
    sizes = overrides.pop('sizes', (2, 3, 4, 5, 6))
    env = IICMSMobileEnvCore(size_options=sizes, mobility=overrides.pop('mobility', 'rwp'), ue_speed=overrides.pop('ue_speed', 10.0), slot_dt=overrides.pop('slot_dt', 0.5), csi_aging=overrides.pop('csi_aging', False), pathloss_model=overrides.pop('pathloss_model', 'legacy'), reward_mode=overrides.pop('reward_mode', 'mean_se'), objective_horizon=overrides.pop('objective_horizon', 100), backhaul_jt_model=overrides.pop('backhaul_jt_model', 'product'))
    env.configure(num_agents=12, num_aps=18, map_size=1000.0, seed=overrides.pop('seed', 7), size_options=sizes, **overrides)
    return env

class IICMSMobilityTest(unittest.TestCase):

    def test_spaces_follow_mode_and_size_heads(self):
        env = make_env(sizes=(2, 3, 4, 5, 6))
        self.assertEqual(env.action_dim, [11])
        self.assertEqual(len(env.joint_actions), 11)
        self.assertEqual(env.joint_actions[0], (0, 2))
        self.assertEqual(env.joint_actions[-1], (2, 1))
        self.assertEqual(env.obs_token_dim, 4)
        self.assertEqual(env.obs_tokens_per_frame, 7)
        self.assertEqual(env.obs_dim, 28)
        obs = env.reset()
        self.assertEqual(np.stack(obs).shape, (12, 28))

    def test_observation_tokens_are_normalized_and_informative(self):
        env = make_env(sizes=(2, 3, 5))
        env.reset()
        actions = [encode_iicms_action(env, 0, 2) for _ in range(env.agent_num)]
        (obs, _, _, _) = env.step(actions)
        matrix = np.stack(obs)
        self.assertEqual(matrix.shape, (12, 24))
        self.assertTrue(np.all(np.isfinite(matrix)))
        self.assertGreaterEqual(float(matrix[:, 1].min()), 0.0)
        self.assertGreaterEqual(float(matrix[:, 3].min()), 0.0)
        ap_tokens = matrix[:, 4:].reshape(env.agent_num, env.cluster_max, env.obs_token_dim)
        self.assertLessEqual(float(ap_tokens[:, :, 0].max()), 0.0)
        self.assertGreaterEqual(float(ap_tokens[:, :, 0].min()), -1.0)
        self.assertGreaterEqual(float(ap_tokens[:, :, 1:3].min()), 0.0)
        self.assertLessEqual(float(ap_tokens[:, :, 1:3].max()), 1.0)
        self.assertGreater(float(ap_tokens[:, :, 0].std()), 0.0)

    def test_xy_features_and_frame_history(self):
        env = make_env(sizes=(2, 3, 4, 5), cluster_action_mode='subset', obs_include_xy=1, obs_frames=4)
        self.assertEqual(env.joint_num_cluster_choices, 26)
        self.assertEqual(len(env.joint_actions), 53)
        self.assertEqual(env.obs_token_dim, 6)
        self.assertEqual(env.obs_tokens_per_frame, 6)
        self.assertEqual(env.obs_frame_dim, 36)
        self.assertEqual(env.obs_dim, 144)
        obs = env.reset()
        matrix = np.stack(obs)
        self.assertEqual(matrix.shape, (env.agent_num, 144))
        self.assertTrue(np.allclose(matrix, 0.0))
        action = encode_iicms_action(env, 0, 2)
        (obs, _, _, _) = env.step([action.copy() for _ in range(env.agent_num)])
        matrix = np.stack(obs)
        self.assertEqual(matrix.shape, (env.agent_num, 144))
        self.assertTrue(np.allclose(matrix[:, :108], 0.0))
        self.assertGreater(float(np.abs(matrix[:, 108:]).max()), 0.0)
        frames = matrix.reshape(env.agent_num, env.obs_frames, env.obs_frame_dim)
        ap_tokens = frames[:, :, 6:].reshape(env.agent_num, env.obs_frames, env.cluster_max, 6)
        self.assertTrue(np.all(ap_tokens[:, :, :, 4:] >= -1.0))
        self.assertTrue(np.all(ap_tokens[:, :, :, 4:] <= 1.0))

    def test_static_cluster_policy_freezes_candidate_ap_order(self):
        env = make_env(sizes=(2, 3, 4, 5), cluster_action_mode='subset', cluster_policy='static')
        env.reset()
        frozen = env._static_candidate_order.copy()
        env.mu = np.tile(np.arange(env.ap_num, dtype=float), (env.agent_num, 1))
        np.testing.assert_array_equal(env._candidate_ap_order(), frozen)

    def test_static_cluster_has_no_cluster_changes(self):
        env = make_env(sizes=(2, 3, 4, 5), cluster_action_mode='subset', cluster_policy='static')
        env.reset()
        action = encode_iicms_action(env, 1, 5)
        actions = [action.copy() for _ in range(env.agent_num)]
        (_, _, _, infos) = env.step(actions)
        self.assertEqual(np.mean([item['cluster_changed'] for item in infos]), 0.0)
        (_, _, _, infos) = env.step(actions)
        self.assertEqual(np.mean([item['cluster_changed'] for item in infos]), 0.0)

    def test_fronthaul_queue_adds_ap_observation_feature(self):
        env = make_env(sizes=(2, 3, 4, 5), cluster_action_mode='subset', obs_include_xy=1, obs_frames=4, fronthaul_queue=1, fronthaul_buffer_s=1.0)
        self.assertEqual(env.obs_token_dim, 7)
        self.assertEqual(env.obs_tokens_per_frame, 6)
        self.assertEqual(env.obs_frame_dim, 42)
        self.assertEqual(env.obs_dim, 168)
        obs = env.reset()
        self.assertEqual(np.stack(obs).shape, (env.agent_num, 168))
        action = encode_iicms_action(env, 0, 2)
        (obs, _, _, infos) = env.step([action.copy() for _ in range(env.agent_num)])
        matrix = np.stack(obs)
        self.assertEqual(matrix.shape, (env.agent_num, 168))
        self.assertTrue(np.all(np.isfinite(matrix)))
        self.assertIn('fronthaul_queue_ratio', infos[0])
        self.assertIn('fronthaul_drop_ratio', infos[0])

    def test_finite_fronthaul_queue_builds_overflows_and_drains(self):
        env = make_env(sizes=(2, 3), slot_dt=0.5, fronthaul_queue=1, fronthaul_buffer_s=1.0)
        env.backhaul_caps = np.full(env.ap_num, 10.0)
        env._fronthaul_queue_bits = np.zeros(env.ap_num, dtype=float)
        offered = np.zeros(env.ap_num, dtype=float)
        offered[0] = 4.0
        drained = env._apply_fronthaul_constraint(offered, advance_queue=True)
        self.assertAlmostEqual(float(drained['fronthaul_queue_bits'][0]), 0.0)
        self.assertAlmostEqual(float(drained['ap_scale'][0]), 1.0)
        offered[0] = 20.0
        first = env._apply_fronthaul_constraint(offered, advance_queue=True)
        self.assertAlmostEqual(float(first['fronthaul_queue_bits'][0]), 5.0)
        self.assertAlmostEqual(float(first['ap_scale'][0]), 0.5)
        self.assertAlmostEqual(float(first['fronthaul_overflow_bits'][0]), 0.0)
        second = env._apply_fronthaul_constraint(offered, advance_queue=True)
        self.assertAlmostEqual(float(second['fronthaul_queue_bits'][0]), 10.0)
        self.assertAlmostEqual(float(second['ap_scale'][0]), 5.0 / 15.0)
        self.assertAlmostEqual(float(second['fronthaul_overflow_bits'][0]), 0.0)
        third = env._apply_fronthaul_constraint(offered, advance_queue=True)
        self.assertAlmostEqual(float(third['fronthaul_queue_bits'][0]), 10.0)
        self.assertAlmostEqual(float(third['ap_scale'][0]), 5.0 / 20.0)
        self.assertAlmostEqual(float(third['fronthaul_overflow_bits'][0]), 5.0)
        offered[0] = 0.0
        draining = env._apply_fronthaul_constraint(offered, advance_queue=True)
        self.assertAlmostEqual(float(draining['fronthaul_queue_bits'][0]), 5.0)

    def test_evaluate_current_does_not_advance_fronthaul_queue(self):
        env = make_env(sizes=(2, 3, 5), fronthaul_queue=1, fronthaul_buffer_s=1.0)
        env.reset()
        env._fronthaul_queue_bits[:] = 123.0
        before = env._fronthaul_queue_bits.copy()
        env.evaluate_current(0, 2)
        np.testing.assert_array_equal(env._fronthaul_queue_bits, before)

    def test_reset_clears_fronthaul_queue(self):
        env = make_env(sizes=(2, 3, 5), fronthaul_queue=1, fronthaul_buffer_s=1.0)
        env.reset()
        env._fronthaul_queue_bits[:] = 123.0
        env.reset()
        self.assertTrue(np.all(env._fronthaul_queue_bits == 0.0))

    def test_encoded_action_decodes_exact_size(self):
        env = make_env(sizes=(2, 3, 4, 5))
        for size in (2, 3, 4, 5):
            action = encode_iicms_action(env, 1, size)
            (mode, decoded, _action_index) = env._decode_iicms_actions([action])
            self.assertEqual(int(mode[0]), 1)
            self.assertEqual(int(decoded[0]), size)
            self.assertEqual(float(np.sum(action)), 1.0)

    def test_subset_mode_covers_ap_membership_patterns(self):
        env = make_env(sizes=(2, 3, 5), cluster_action_mode='subset')
        self.assertEqual(env.cluster_action_mode, 'subset')
        self.assertEqual(env.joint_num_cluster_choices, 21)
        self.assertEqual(len(env.joint_actions), 43)
        self.assertEqual(env.action_dim, [43])
        self.assertEqual(env.action_names[0], 'JT_L2_S00')
        self.assertEqual(env.action_names[20], 'JT_L5_S00')
        self.assertEqual(env.action_names[-1], 'NC')
        env.reset()
        top_l = encode_iicms_action(env, 0, 2, subset_index=0)
        alternate = encode_iicms_action(env, 0, 2, subset_index=1)
        (_mode, _size, action_idx) = env._decode_iicms_actions([top_l, alternate])
        clusters = env._clusters_for_actions(action_idx)
        self.assertEqual(env.joint_action_cluster_ranks[int(action_idx[0])], (0, 1))
        self.assertEqual(env.joint_action_cluster_ranks[int(action_idx[1])], (0, 2))
        self.assertNotEqual(set(clusters[0].tolist()), set(clusters[1].tolist()))

    def test_topk_mode_has_only_strongest_cluster_patterns(self):
        env = make_env(sizes=(2, 3, 5))
        self.assertEqual(env.cluster_action_mode, 'topk')
        self.assertEqual(env.joint_num_cluster_choices, 3)
        self.assertEqual(len(env.joint_actions), 7)
        for ranks in env.joint_action_cluster_ranks[:-1]:
            self.assertEqual(ranks, tuple(range(len(ranks))))

    def test_nc_has_one_joint_action_without_cluster_options(self):
        env = make_env(sizes=(2, 3, 5))
        nc_actions = [action for action in env.joint_actions if action[0] == 2]
        self.assertEqual(nc_actions, [(2, 1)])
        self.assertEqual(len(env.joint_actions), 7)
        self.assertNotIn('NC_L3', env.action_names)
        self.assertNotIn('NC_L5', env.action_names)

    def test_nc_ignores_cluster_size(self):
        env = make_env(sizes=(2, 3, 4, 5), csi_aging=False)
        env.reset()
        low = env.evaluate_current(2, 2)
        high = env.evaluate_current(2, 5)
        self.assertAlmostEqual(low['sum_rate_mbps'], high['sum_rate_mbps'], places=10)
        np.testing.assert_allclose(low['per_ue_rate_mbps'], high['per_ue_rate_mbps'], rtol=0.0, atol=1e-10)

    def test_cb_cluster_size_changes_interference(self):
        env = make_env(sizes=(2, 3, 4, 5), csi_aging=False)
        env.reset()
        low = env.evaluate_current(1, 2)
        high = env.evaluate_current(1, 5)
        self.assertNotAlmostEqual(low['sum_rate_mbps'], high['sum_rate_mbps'], places=4)

    def test_mobility_updates_large_scale_channels(self):
        env = make_env(csi_aging=False)
        env.reset()
        distance_before = env.distance.copy()
        channels_before = env.mu.copy()
        action = encode_iicms_action(env, 0, 4)
        for _ in range(20):
            env.step([action.copy() for _ in range(env.agent_num)])
        self.assertGreater(float(np.mean(np.abs(env.distance - distance_before))), 0.0)
        self.assertGreater(float(np.mean(np.abs(env.mu - channels_before))), 0.0)

    def test_csi_aging_ab(self):
        clean = make_env(csi_aging=False)
        aged = make_env(csi_aging=True)
        self.assertTrue(np.allclose(clean._rho2, 1.0))
        self.assertTrue(np.all(aged._rho2 < 1.0))
        self.assertGreater(float(np.mean(clean._rho2 - aged._rho2)), 0.0)

    def test_paper_rewards_are_not_identical_by_construction(self):
        env = make_env(reward_mode='paper')
        env.reset()
        actions = [encode_iicms_action(env, mode, size) for (mode, size) in zip(np.resize(np.arange(3), env.agent_num), np.resize(np.asarray(env.size_options), env.agent_num))]
        (_, rewards, _, infos) = env.step(actions)
        reward_values = np.asarray(rewards).reshape(-1)
        self.assertGreater(float(np.std(reward_values)), 0.0)
        self.assertIn('cluster_change_rate', env._last_step_summary)
        self.assertNotIn('ho', infos[0])

    def test_mean_se_reward_matches_time_objective(self):
        env = make_env(reward_mode='mean_se', objective_horizon=50)
        env.reset()
        actions = [encode_iicms_action(env, 1, 5) for _ in range(env.agent_num)]
        (_, rewards, _, infos) = env.step(actions)
        expected = float(np.mean([item['spectral_efficiency_bps_hz'] for item in infos]))
        np.testing.assert_allclose(np.asarray(rewards).reshape(-1), expected, rtol=0.0, atol=1e-07)
        self.assertEqual(env.objective_horizon, 50)

    def test_team_rate_reward_matches_system_rate_mbps(self):
        env = make_env(reward_mode='team_rate', qos_penalty_weight=0.0)
        env.reset()
        actions = [encode_iicms_action(env, 1, 3) for _ in range(env.agent_num)]
        (_, rewards, _, infos) = env.step(actions)
        expected = sum((item['inst_rate'] for item in infos)) / 1000000.0
        for reward in rewards:
            self.assertAlmostEqual(float(reward[0]), expected, places=3)

    def test_ap_minimum_spacing_is_enforced(self):
        env = make_env()
        pair_distance = np.linalg.norm(env.aps_xy[:, None, :] - env.aps_xy[None, :, :], axis=2)
        pair_distance[np.diag_indices(env.ap_num)] = np.inf
        self.assertGreaterEqual(float(np.min(pair_distance)), env.ap_min_distance - 1e-09)
        self.assertGreaterEqual(env.ap_min_distance, 50.0)

    def test_qos_penalty_activates_on_rate_shortfall(self):
        env = make_env(reward_mode='mean_se', rate_req_lo_mbps=1000.0, rate_req_hi_mbps=1000.0, qos_penalty_weight=20.0)
        env.reset()
        actions = [encode_iicms_action(env, 2, 2) for _ in range(env.agent_num)]
        (_, rewards, _, infos) = env.step(actions)
        mean_se = float(np.mean([item['spectral_efficiency_bps_hz'] for item in infos]))
        self.assertAlmostEqual(env._last_step_summary['qos_violation_rate'], 1.0)
        self.assertLess(float(rewards[0][0]), mean_se)
        self.assertGreater(env._last_step_summary['qos_penalty'], 0.0)

    def test_se_floor_is_reported_without_changing_reward(self):
        env = make_env(reward_mode='mean_se', min_ue_se_bps_hz=1000.0, qos_penalty_weight=0.0)
        env.reset()
        actions = [encode_iicms_action(env, 2, 1) for _ in range(env.agent_num)]
        (_, rewards, _, infos) = env.step(actions)
        expected = float(np.mean([item['spectral_efficiency_bps_hz'] for item in infos]))
        self.assertAlmostEqual(float(rewards[0][0]), expected)
        self.assertEqual(env._last_step_summary['se_floor_violation_rate'], 1.0)
        self.assertTrue(all((item['se_floor_violation'] for item in infos)))

    def test_jt_bottleneck_is_not_more_pessimistic_than_product(self):
        product = make_env(backhaul_jt_model='product', backhaul_cap_range=(1000000.0, 1000000.0))
        bottleneck = make_env(backhaul_jt_model='bottleneck', backhaul_cap_range=(1000000.0, 1000000.0))
        product.reset()
        bottleneck.reset()
        mode = np.zeros(product.agent_num, dtype=np.int32)
        sizes = np.full(product.agent_num, 2, dtype=np.int32)
        product_result = product._compute_metrics(mode, sizes, product._clusters_for_size(sizes))
        bottleneck_result = bottleneck._compute_metrics(mode, sizes, bottleneck._clusters_for_size(sizes))
        self.assertTrue(np.all(product_result['backhaul_scale'] <= bottleneck_result['backhaul_scale'] + 1e-12))
        self.assertTrue(np.any(product_result['backhaul_scale'] < bottleneck_result['backhaul_scale'] - 1e-09))

    def test_cluster_size_action_has_observable_effect(self):
        env = make_env(sizes=(2, 3, 4, 5), slot_dt=1.0)
        env.reset()
        action = encode_iicms_action(env, 1, 5)
        changed = False
        for _ in range(40):
            (_, _, _, infos) = env.step([action.copy() for _ in range(env.agent_num)])
            changed |= any((item['cluster_changed'] for item in infos))
        self.assertTrue(changed)
if __name__ == '__main__':
    unittest.main()
