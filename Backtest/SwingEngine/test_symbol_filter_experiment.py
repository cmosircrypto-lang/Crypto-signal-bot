import unittest

import pandas as pd

from backtest_runner import FILTER_MODES
from symbol_filter_experiment import classify_symbols, _subset_signals


class SymbolFilterExperimentTests(unittest.TestCase):
    def test_classification_uses_consistent_training_performance(self):
        training = {}
        for mode in FILTER_MODES:
            rows = []
            for symbol, values in (
                ("GOOD", [2.0, 2.0, 2.0, 2.0, -1.0, -1.0]),
                ("BAD", [1.0, 1.0, -2.0, -2.0, -2.0, -2.0]),
                ("MIXED", [1.0, 1.0, 1.0, -1.0, -1.0, -1.0]),
            ):
                rows.extend({"symbol": symbol, "pnl": value} for value in values)
            training[mode] = pd.DataFrame(rows)
        strong, weak, details = classify_symbols(training, min_trades=5)
        self.assertEqual(strong, ["GOOD"])
        self.assertEqual(weak, ["BAD"])
        self.assertEqual(
            details.set_index("symbol").loc["MIXED", "classification"],
            "mixed",
        )

    def test_classification_excludes_insufficient_samples(self):
        training = {}
        for mode in FILTER_MODES:
            training[mode] = pd.DataFrame([
                {"symbol": "TOO_FEW", "pnl": 1.0},
                {"symbol": "TOO_FEW", "pnl": -1.0},
            ])
        strong, weak, details = classify_symbols(training, min_trades=5)
        self.assertEqual(strong, [])
        self.assertEqual(weak, [])
        self.assertTrue(details.empty)

    def test_signals_are_split_without_boundary_overlap(self):
        signals = {"X": pd.DataFrame({"signal_ts": [10, 20, 30]})}
        train = _subset_signals(signals, 20, True)["X"]
        test = _subset_signals(signals, 20, False)["X"]
        self.assertEqual(train["signal_ts"].tolist(), [10, 20])
        self.assertEqual(test["signal_ts"].tolist(), [30])


if __name__ == "__main__":
    unittest.main()
