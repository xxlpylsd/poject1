from __future__ import annotations
import sys
import unittest
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from package_results import is_useful_result, require_tag

class PackageResultsTest(unittest.TestCase):

    def test_tag_must_not_be_empty(self):
        self.assertEqual(require_tag(' demo_tag '), 'demo_tag')
        with self.assertRaises(ValueError):
            require_tag(' ')

    def test_results_mode_keeps_summaries_and_omits_large_artifacts(self):
        kept = ['logs_iicms/demo_tf_s1.log', 'results/run1/config.json', 'results/run1/checkpoint_selection.json', 'results/run1/evaluation_deterministic/run1.json', 'results/run1/evaluation_deterministic/run1_rate_cdf.npz', 'results/run1/evaluation_stochastic/run1.json', 'results/run1/baseline_fixed_cb_l5_deterministic/run1.json', 'results/run1/baseline_suite/sc_cb_l5/run1.json', 'results/run1/baseline_suite/sc_cb_l5/run1_rate_cdf.npz', 'results/run1/baseline_suite/baseline_summary.csv', 'results/run1/baseline_suite/baseline_summary.json', 'results/run1/rssi_threshold_calibration/rssi_threshold_calibration.json', 'results/run1/rssi_threshold_calibration/rssi_threshold_calibration.csv', 'results/run1/rssi_threshold_calibration/selected_rssi_thresholds.env', 'results/run1/logs/curve.csv', 'results/run1/logs/metrics_history.json', 'results/run1/logs/summary.json', 'results/IICMSMobility/PureMobility/mappo/demo_figures/demo_se_cdf.png', 'results/IICMSMobility/PureMobility/mappo/demo_figures/demo_se_cdf.json', 'results/IICMSMobility/PureMobility/mappo/demo_figures/demo_training_curves.png']
        omitted = ['results/run1/models/actor_best.pt', 'results/run1/models/actor_step_00000800.pt', 'results/run1/reward_proposed.npy', 'results/run1/logs/events.out.tfevents.demo']
        for relative in kept:
            self.assertTrue(is_useful_result(Path(relative), 'results'), relative)
        for relative in omitted:
            self.assertFalse(is_useful_result(Path(relative), 'results'), relative)

    def test_best_mode_keeps_best_weights_but_not_step_snapshots(self):
        self.assertTrue(is_useful_result(Path('results/run1/models/actor_best.pt'), 'best'))
        self.assertTrue(is_useful_result(Path('results/run1/models/critic_best.pt'), 'best'))
        self.assertTrue(is_useful_result(Path('results/run1/models/actor_agent0.pt'), 'best'))
        self.assertTrue(is_useful_result(Path('results/run1/models/critic_agent29.pt'), 'best'))
        self.assertFalse(is_useful_result(Path('results/run1/models/actor_step_00000800.pt'), 'best'))
        self.assertFalse(is_useful_result(Path('results/run1/models/actor_final.pt'), 'best'))

    def test_full_mode_keeps_all_files(self):
        self.assertTrue(is_useful_result(Path('results/run1/models/actor_step_00000800.pt'), 'full'))
        self.assertTrue(is_useful_result(Path('results/run1/reward_proposed.npy'), 'full'))
if __name__ == '__main__':
    unittest.main()
