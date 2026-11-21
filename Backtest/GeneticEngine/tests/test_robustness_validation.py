import unittest

from Backtest.GeneticEngine.robustness_validation import split_oos_windows


class RobustnessWindowTests(unittest.TestCase):
    def test_windows_are_disjoint_exhaustive_and_ordered(self):
        timestamps = list(range(100, 112))
        windows = split_oos_windows(timestamps, 3)
        self.assertEqual(windows, [(100, 103), (104, 107), (108, 111)])
        rebuilt = []
        for start, end in windows:
            rebuilt.extend(value for value in timestamps if start <= value <= end)
        self.assertEqual(rebuilt, timestamps)

    def test_rejects_too_few_timestamps(self):
        with self.assertRaises(ValueError):
            split_oos_windows([1, 2], 3)

    def test_rejects_non_increasing_timestamps(self):
        with self.assertRaises(ValueError):
            split_oos_windows([1, 2, 2, 4], 2)


if __name__ == "__main__":
    unittest.main()
