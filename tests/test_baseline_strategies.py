from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluate_iicms import rssi_modes  # noqa: E402


class _FakeCore:
    def __init__(self, gains: np.ndarray):
        self.mu = np.asarray(gains, dtype=float)

    def _candidate_ap_order(self) -> np.ndarray:
        return np.argsort(self.mu, axis=1)[:, ::-1]


class RssiBaselineTest(unittest.TestCase):
    def test_rssi_threshold_selects_jt_cb_and_nc(self):
        wrapper = SimpleNamespace(
            num_agent=3,
            env=_FakeCore(np.asarray([
                [10.0, 0.1, 0.1],
                [5.0, 1.0, 1.0],
                [0.5, 5.0, 5.0],
            ])),
        )
        modes = rssi_modes(
            wrapper,
            threshold_hi=20.0,
            threshold_lo=1.0,
        )
        np.testing.assert_array_equal(modes, [2, 1, 0])

    def test_rssi_thresholds_are_validated(self):
        wrapper = SimpleNamespace(
            num_agent=1,
            env=_FakeCore(np.asarray([[1.0, 0.5]])),
        )
        with self.assertRaises(ValueError):
            rssi_modes(wrapper, threshold_hi=1.0, threshold_lo=20.0)


if __name__ == "__main__":
    unittest.main()
