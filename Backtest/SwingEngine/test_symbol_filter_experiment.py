import unittest

import pandas as pd

from symbol_filter_experiment import FILTER_MODES, classify_symbols, _subset_signals


class SymbolFilterExperimentTests(unittest.TestCase):
    def test_classification_requires_minimum_sample_in_every_filter(self):
        training = {}
        for mode in FILTER_MODES:
            rows = []
            for symbol, pf in (("GOOD", 1.5), ("BAD", 0.5), ("MIXED", 1.2)):
                for index in range(6):
                    rows.append({
                        "symbol": symbol,
                        "pnl": 1.0 if index < int(4 * pf / (1 + pf)) else -1.0,
                    })
            training[mode] = pd.DataFrame(rows)
        strong, weak, details = classify_symbols(training, min_trades=5)
        self.assertIn("GOOD", strong + weak + details["symbol"].tolist())
        self.assertEqual(set(strong).intersection(weak), set())

    def test_signals_are_split_strictly_after_cutoff(self):
        signals = {"X": pd.DataFrame({"signal_ts": [10, 20, 30]})}
        train = _subset_signals(signals, 20, True)["X"]
        test = _subset_signals(signals, 20, False)["X"]
        self.assertEqual(train["signal_ts"].tolist(), [10, 20])
        self.assertEqual(test["signal_ts"].tolist(), [30])


if __name__ == "__main__":
    unittest.main()
