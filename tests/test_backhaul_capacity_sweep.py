from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluate_backhaul_capacity_sweep import (  # noqa: E402
    add_eval_args,
    aggregate_rows,
    parse_float_list,
)


class BackhaulCapacitySweepTest(unittest.TestCase):
    def test_capacity_list_is_sorted_and_unique(self):
        self.assertEqual(
            parse_float_list(
                "1.5,0.3,0.6,0.3", "capacities"
            ),
            [0.3, 0.6, 1.5],
        )

    def test_reference_capacity_matches_current_environment(self):
        command = ["python", "evaluate_iicms.py"]
        low, high = add_eval_args(
            command,
            Path("run"),
            Path("out"),
            center_gbps=0.6,
            variation_gbps=0.05,
            seed_start=93000,
            episodes=20,
            horizon=100,
        )
        self.assertEqual(low, 550_000_000)
        self.assertEqual(high, 650_000_000)

    def test_capacity_rows_are_averaged_across_seeds(self):
        base = {
            "strategy": "tf",
            "strategy_label": "TF-native",
            "capacity_gbps": 0.6,
            "sum_rate_mbps": 100.0,
            "per_ue_rate_p05_mbps": 20.0,
            "qos_violation_rate": 0.1,
            "overload_ratio": 0.2,
            "fronthaul_queue_delay_ms": 3.0,
            "cluster_change_rate": 0.4,
        }
        result = aggregate_rows([
            {**base, "model_seed": 1},
            {
                **base,
                "model_seed": 2,
                "sum_rate_mbps": 140.0,
            },
        ])
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["model_seed_count"], 2)
        self.assertAlmostEqual(
            result[0]["sum_rate_mbps_mean"], 120.0
        )


if __name__ == "__main__":
    unittest.main()
