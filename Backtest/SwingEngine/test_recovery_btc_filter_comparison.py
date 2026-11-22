import unittest
from recovery_cycle_resumable import _btc_filter_matches

class BTCFilterComparisonTests(unittest.TestCase):
    def test_strict_requires_both_timeframes(self):
        self.assertTrue(_btc_filter_matches({"1d":"BUY","4h":"BUY"},"BUY","1d_4h"))
        self.assertFalse(_btc_filter_matches({"1d":"NEUTRAL","4h":"BUY"},"BUY","1d_4h"))

    def test_daily_neutral_falls_back_to_four_hour(self):
        self.assertTrue(_btc_filter_matches({"1d":"NEUTRAL","4h":"BUY"},"BUY","1d_neutral_fallback"))
        self.assertFalse(_btc_filter_matches({"1d":"SELL","4h":"BUY"},"BUY","1d_neutral_fallback"))
        self.assertFalse(_btc_filter_matches({"1d":"BUY","4h":"NEUTRAL"},"BUY","1d_neutral_fallback"))

    def test_either_timeframe_can_agree_but_other_cannot_oppose(self):
        self.assertTrue(_btc_filter_matches({"1d":"NEUTRAL","4h":"BUY"},"BUY","either_nonopposing"))
        self.assertTrue(_btc_filter_matches({"1d":"BUY","4h":"NEUTRAL"},"BUY","either_nonopposing"))
        self.assertFalse(_btc_filter_matches({"1d":"SELL","4h":"BUY"},"BUY","either_nonopposing"))
        self.assertFalse(_btc_filter_matches({"1d":"NEUTRAL","4h":"NEUTRAL"},"BUY","either_nonopposing"))

if __name__ == "__main__":
    unittest.main()
