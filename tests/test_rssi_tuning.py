from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tune_rssi_thresholds import parse_grid, resolve_run_dir


class RssiTuningTest(unittest.TestCase):
    def test_parse_grid_sorts_and_deduplicates(self):
        self.assertEqual(
            parse_grid("1, 0.1,1, 0.2"),
            [0.1, 0.2, 1.0],
        )
        with self.assertRaises(ValueError):
            parse_grid("0,1")

    def test_resolve_run_dir_accepts_experiment_or_run_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            experiment = Path(directory)
            (experiment / "run1").mkdir()
            (experiment / "run1" / "config.json").write_text(
                "{}", encoding="utf-8"
            )
            (experiment / "run2").mkdir()
            (experiment / "run2" / "config.json").write_text(
                "{}", encoding="utf-8"
            )
            self.assertEqual(
                resolve_run_dir(experiment),
                experiment / "run2",
            )
            self.assertEqual(
                resolve_run_dir(experiment / "run1"),
                experiment / "run1",
            )

    def test_resolve_run_dir_reports_missing_config(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                resolve_run_dir(Path(directory))


if __name__ == "__main__":
    unittest.main()
