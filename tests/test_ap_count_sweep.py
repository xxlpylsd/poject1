from __future__ import annotations
import sys
import unittest
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from evaluate_ap_count_sweep import aggregate_rows, parse_int_list, run_dir_for_family
from plot_ap_count_results import grouped_rows, paired_differences, relative_paired_gain
from config import get_config
from envs.env_iicms_mobile import IICMSMobileDiscreteActionEnv
from train_main import parse_args

class ApCountSweepTest(unittest.TestCase):

    def test_ap_count_list_is_sorted_and_unique(self):
        self.assertEqual(parse_int_list('50,10,30,10', 'AP counts'), [10, 30, 50])

    def test_learning_seeds_are_aggregated_at_each_ap_count(self):
        base = {'strategy': 'mlp', 'strategy_label': 'MLP-native', 'ap_count': 10, 'sum_rate_mbps': 100.0, 'rate_per_ap_mbps': 10.0, 'per_ue_rate_p05_mbps': 20.0, 'qos_violation_rate': 0.1, 'overload_ratio': 0.2, 'fronthaul_queue_delay_ms': 3.0, 'cluster_change_rate': 0.4}
        rows = [{**base, 'model_seed': 1}, {**base, 'model_seed': 2, 'sum_rate_mbps': 140.0, 'rate_per_ap_mbps': 14.0, 'per_ue_rate_p05_mbps': 24.0, 'qos_violation_rate': 0.2}]
        result = aggregate_rows(rows)
        self.assertEqual(len(result), 1)
        row = result[0]
        self.assertEqual(row['model_seed_count'], 2)
        self.assertAlmostEqual(row['sum_rate_mbps_mean'], 120.0)
        self.assertAlmostEqual(row['rate_per_ap_mbps_mean'], 12.0)
        self.assertAlmostEqual(row['per_ue_rate_p05_mbps_mean'], 22.0)

    def test_ap_count_preserves_policy_input_dimensions(self):
        args = parse_args(['--num_agents', '30', '--num_aps', '10', '--cluster_sizes', '2,3,4,5', '--obs_frames', '4'], get_config())
        env = IICMSMobileDiscreteActionEnv()
        env.configure(args)
        low_ap_shape = env.observation_space[0].shape
        args.num_aps = 50
        env.configure(args)
        high_ap_shape = env.observation_space[0].shape
        self.assertEqual(low_ap_shape, high_ap_shape)

    def test_old_and_new_mlp_use_mlp_run_suffix(self):
        root = Path('results')
        self.assertEqual(run_dir_for_family(root, 'old_tag', 'old_mlp', 2), root / 'old_tag_mlp_s2' / 'run1')
        self.assertEqual(run_dir_for_family(root, 'new_tag', 'mlp', 2), root / 'new_tag_mlp_s2' / 'run1')

    def test_tf_mlp_paired_gain_tracks_ap_count(self):
        rows = []
        for (ap_count, values) in ((10, (90.0, 80.0)), (20, (110.0, 220.0))):
            for (seed, (mlp_rate, tf_rate)) in enumerate(zip((100.0, 200.0), values), 1):
                rows.extend([{'strategy': 'mlp', 'ap_count': ap_count, 'model_seed': seed, 'sum_rate_mbps': mlp_rate, 'per_ue_rate_p05_mbps': 20.0, 'qos_violation_rate': 0.1, 'rate_per_ap_mbps': mlp_rate / ap_count}, {'strategy': 'tf', 'ap_count': ap_count, 'model_seed': seed, 'sum_rate_mbps': tf_rate * (2.0 if seed == 2 else 1.0), 'per_ue_rate_p05_mbps': 22.0, 'qos_violation_rate': 0.08, 'rate_per_ap_mbps': tf_rate / ap_count}])
        grouped = grouped_rows(rows)
        (_x, delta, _ci, wins) = paired_differences(grouped, [10, 20], 'sum_rate_mbps')
        (gain, _gain_ci) = relative_paired_gain(grouped, [10, 20])
        self.assertLess(delta[0], 0.0)
        self.assertGreater(delta[1], 0.0)
        self.assertEqual(wins, [0, 2])
        self.assertLess(gain[0], 0.0)
        self.assertGreater(gain[1], 0.0)
if __name__ == '__main__':
    unittest.main()
