from __future__ import annotations
import json
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from plot_reward_convergence import flatten_reward_array, group_reward_episodes, load_reward_curve, reward_column

class RewardConvergencePlotTest(unittest.TestCase):

    def test_flatten_and_group_reward_by_episode(self):
        values = np.asarray([[[1.0], [3.0]], [[2.0], [4.0]], [[5.0], [7.0]], [[6.0], [8.0]]])
        step_mean = flatten_reward_array(values)
        np.testing.assert_allclose(step_mean, [2.0, 3.0, 6.0, 7.0])
        (means, stds) = group_reward_episodes(step_mean, 2)
        np.testing.assert_allclose(means, [2.5, 6.5])
        np.testing.assert_allclose(stds, [np.std([2, 3], ddof=1), np.std([6, 7], ddof=1)])

    def test_raw_reward_file_takes_precedence_over_logged_curve(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            np.save(run_dir / 'reward_proposed.npy', np.asarray([[[1.0], [2.0]], [[3.0], [4.0]], [[5.0], [6.0]], [[7.0], [8.0]]]))
            (run_dir / 'config.json').write_text(json.dumps({'reward_mode': 'team_rate'}), encoding='utf-8')
            (episodes, means, _stds, source, _label) = load_reward_curve(run_dir, episode_length=2)
            np.testing.assert_array_equal(episodes, [1, 2])
            np.testing.assert_allclose(means, [2.5, 6.5])
            self.assertEqual(source, 'raw reward_proposed')

    def test_raw_mode_fails_instead_of_silently_using_curve_csv(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            (run_dir / 'logs').mkdir()
            (run_dir / 'config.json').write_text(json.dumps({'reward_mode': 'team_rate'}), encoding='utf-8')
            (run_dir / 'logs' / 'curve.csv').write_text('episode,sum_rate_mbps\n1,1000\n', encoding='utf-8')
            with self.assertRaises(FileNotFoundError):
                load_reward_curve(run_dir, episode_length=1, source='raw')
            (_x, means, _std, source, _label) = load_reward_curve(run_dir, episode_length=1, source='auto')
            np.testing.assert_allclose(means, [1000.0])
            self.assertEqual(source, 'curve.csv fallback')

    def test_reward_mode_selects_matching_logged_column(self):
        self.assertEqual(reward_column('team_rate'), 'sum_rate_mbps')
        self.assertEqual(reward_column('mean_se'), 'mean_ue_se_bps_hz')
        self.assertEqual(reward_column('team_sum'), 'system_sum_se_bps_hz')
if __name__ == '__main__':
    unittest.main()
