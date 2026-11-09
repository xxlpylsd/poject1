import io
import unittest
from contextlib import redirect_stdout
from runner.shared.env_runner import _TrainingProgress, _objective_metric

class TrainingProgressTest(unittest.TestCase):

    def test_team_rate_uses_system_rate_display(self):
        (label, value, unit) = _objective_metric('team_rate', {'sum_rate_mbps': 3210.5, 'system_sum_se_bps_hz': 160.525})
        self.assertEqual(('SysRate', 3210.5, 'Mbps'), (label, value, unit))

    def test_mean_se_uses_mean_ue_se_display(self):
        (label, value, unit) = _objective_metric('mean_se', {'mean_ue_se_bps_hz': 5.5, 'system_sum_se_bps_hz': 165.0})
        self.assertEqual(('MeanSE', 5.5, 'bps/Hz'), (label, value, unit))

    def test_progress_line_does_not_report_se_for_team_rate(self):
        output = io.StringIO()
        progress = _TrainingProgress(10, 1000, reward_mode='team_rate', refresh_s=0.0)
        with redirect_stdout(output):
            progress.update(episode_index=0, env_steps=100, elapsed_s=1.0, metrics={'sum_rate_mbps': 2800.0, 'system_sum_se_bps_hz': 140.0, 'qos_violation_rate': 0.01, 'joint_gate_mean': 0.5, 'overload_ratio': 0.2, 'cluster_size_mean': 3.0, 'mode_jt': 0.4, 'mode_cb': 0.4, 'mode_nc': 0.2}, average_reward=2800.0, force=True)
        line = output.getvalue()
        self.assertIn('objective=team_rate', line)
        self.assertIn('SysRate=2800.00Mbps', line)
        self.assertNotIn('SE=', line)
if __name__ == '__main__':
    unittest.main()
