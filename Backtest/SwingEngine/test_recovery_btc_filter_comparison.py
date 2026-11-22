import unittest
from types import SimpleNamespace
from recovery_cycle_resumable import _btc_filter_matches, _unrealized_position_pnl

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


    def test_linear_buy_unrealized_pnl_uses_latest_close(self):
        position = SimpleNamespace(direction="BUY", entry_price=100.0, quantity=2.5)
        self.assertAlmostEqual(_unrealized_position_pnl(position, 104.0), 10.0)
        self.assertAlmostEqual(_unrealized_position_pnl(position, 96.0), -10.0)

    def test_linear_sell_unrealized_pnl_uses_latest_close(self):
        position = SimpleNamespace(direction="SELL", entry_price=100.0, quantity=2.5)
        self.assertAlmostEqual(_unrealized_position_pnl(position, 96.0), 10.0)
        self.assertAlmostEqual(_unrealized_position_pnl(position, 104.0), -10.0)

if __name__ == "__main__":
    unittest.main()
